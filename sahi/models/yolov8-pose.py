import logging
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list


class Yolov8PoseModel(Yolov8DetectionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_keypoints = None

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to
        self._original_predictions and _original_keypoints.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        if self.image_size is not None:  # ADDED IMAGE SIZE OPTION FOR YOLOV8 MODELS:
            prediction_result = self.model(
                image[:, :, ::-1], imgsz=self.image_size, verbose=False, device=self.device
            )  # YOLOv8 expects numpy arrays to have BGR
        else:
            prediction_result = self.model(
                image[:, :, ::-1], verbose=False, device=self.device
            )  # YOLOv8 expects numpy arrays to have BGR

        prediction_result_bboxes, prediction_result_keypoints = [], []
        for result in prediction_result:
            mask = result.boxes.data[:, 4] >= self.confidence_threshold
            prediction_result_bboxes.append(result.boxes.data[mask])

            if not mask.numel():
                mask = torch.tensor([0], dtype=torch.bool)

            prediction_result_keypoints.append(result.keypoints.data[mask])

        self._original_predictions = prediction_result_bboxes
        self._original_keypoints = prediction_result_keypoints

    @property
    def category_names(self):
        return self.model.names.values()

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model.names)

    @property
    def num_keypoints(self):
        """
        Returns number of keypoints
        """
        last_layer = list(self.model.modules())[-1]
        return int(last_layer.out_channels / 3)

    @staticmethod
    def clip(value, shape=0, clip_type="max"):
        if clip_type == "min":
            return min(shape, value)
        elif clip_type == "max":
            return max(shape, value)

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions
        original_keypoints = self._original_keypoints

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, (image_predictions_in_xyxy_format, image_keypoints) in (
                enumerate(zip(original_predictions, original_keypoints))
        ):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for prediction, keypoints in zip(
                    image_predictions_in_xyxy_format.cpu().detach().numpy(),
                    image_keypoints.cpu().numpy()
            ):
                x1, y1, x2, y2 = prediction[0: 4]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                for i in range(4):
                    bbox[i] = self.clip(bbox[i])

                # fix out of image box coords
                if full_shape is not None:
                    for i in range(4):
                        bbox[i] = self.clip(bbox[i], full_shape[~i % 2], clip_type="min")

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                points = []
                scores = []
                for kpt in keypoints:
                    x, y, conf = kpt
                    point = [x, y]
                    for i in range(2):
                        point[i] = self.clip(point[i])

                    if full_shape is not None:
                        for i in range(2):
                            point[i] = self.clip(point[i], full_shape[~i % 2], clip_type="min")

                    points.append(point)
                    scores.append(conf)

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    keypoints=points,
                    kpts_scores=scores,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
