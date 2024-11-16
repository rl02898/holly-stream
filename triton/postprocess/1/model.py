import json
import os
from ast import literal_eval
from typing import Any, Dict, List, Type, TypeVar, Union

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from c_python_backend_utils import InferenceRequest, InferenceResponse
from dotenv import load_dotenv
from numpy.typing import NDArray
from torch.utils.dlpack import from_dlpack, to_dlpack
from torchvision.ops import nms

T = TypeVar("T")


class EnvArgumentParser:
    """
    A parser for environment variables that supports type casting and default values.

    This class provides functionality similar to argparse.ArgumentParser but for
    environment variables, allowing type specification and default values.
    """

    def __init__(self):
        """
        Initialize an empty environment argument parser.
        """

        self.dict: Dict[str, Any] = {}

    class _define_dict(dict):
        """
        A custom dictionary class that allows attribute-style access to dictionary items.

        This enables accessing dictionary values using both square bracket notation
        and dot notation (e.g., dict['key'] or dict.key).
        """

        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def add_arg(
        self,
        variable: str,
        default: T = None,
        type: Union[Type[T], type] = str,
    ) -> None:
        """
        Add an environment variable argument with optional default value and type.

        Args:
            variable: The name of the environment variable to parse
            default: The default value to use if the environment variable is not set
            type: The type to cast the environment variable value to. Can be a basic type
                 (like str, int) or a complex type (list, tuple, bool)

        Raises:
            ValueError: If the environment variable value cannot be cast to the specified type
        """

        env = os.environ.get(variable)

        if env is None:
            value = default
        else:
            value = self._cast_type(env, type)

        self.dict[variable] = value

    @staticmethod
    def _cast_type(arg: str, d_type: Union[Type[T], type]) -> Any:
        """
        Cast a string argument to the specified type.

        Args:
            arg: The string value to cast
            d_type: The type to cast to. Can be a basic type (like str, int) or
                   a complex type (list, tuple, bool)

        Returns:
            The cast value

        Raises:
            ValueError: If the value cannot be cast to the specified type or
                       if the type is not supported
        """

        if isinstance(type, (list, tuple, bool)):
            try:
                cast_value = literal_eval(arg)
                return cast_value
            except (ValueError, SyntaxError):
                raise ValueError(
                    f"Argument {arg} does not match given data type or is not supported."
                )
        else:
            try:
                cast_value = d_type(arg)
                return cast_value
            except (ValueError, SyntaxError):
                raise ValueError(
                    f"Argument {arg} does not match given data type or is not supported."
                )

    def parse_args(self) -> _define_dict:
        """
        Parse all added arguments and return them in a dictionary with attribute access.

        Returns:
            A dictionary-like object that supports both dictionary access (dict['key'])
            and attribute access (dict.key) for all parsed environment variables
        """

        return self._define_dict(self.dict)


class TritonPythonModel:
    """
    A Triton Inference Server backend Python model for post-processing object detection results.

    This class implements the required interface for Triton backend Python models,
    handling initialization, execution, and cleanup of the post-processing model.
    It supports both CPU and GPU (CUDA) execution modes and handles configuration
    through environment variables.
    """

    def initialize(self, args: Dict[str, Any]) -> None:
        """
        Initialize the model with configuration parameters and environment settings.

        This method sets up logging, model configuration, and loads parameters from
        environment variables. It configures the model for either CPU or GPU execution
        based on the Triton model configuration.

        Args:
            args: Dictionary containing initialization arguments including:
                - model_name: Name of the model
                - model_config: JSON string containing model configuration

        Note:
            Environment variables are used to configure:
                - CAMERA_WIDTH: Width of input image (default: 1280)
                - CAMERA_HEIGHT: Height of input image (default: 720)
                - MODEL_DIMS: Model input dimensions (default: (640, 640))
                - CONFIDENCE_THRESHOLD: Detection confidence threshold (default: 0.3)
                - IOU_THRESHOLD: IoU threshold for NMS (default: 0.25)
                - CLASSES: List of classes to detect (default: None)
                - SANTA_HAT_PLUGIN: Enable Santa hat detection (default: False)
        """

        self.logger = pb_utils.Logger
        self.model_name = args["model_name"]
        model_config = json.loads(args["model_config"])
        self.inputs: List[str] = [
            input["name"] for input in model_config["input"]
        ]
        self.outputs: List[str] = [
            output["name"] for output in model_config["output"]
        ]
        self.cuda: bool = (
            model_config["instance_group"][0]["kind"] == "KIND_GPU"
        )

        load_dotenv()
        parser = EnvArgumentParser()
        parser.add_arg("CAMERA_WIDTH", default=1280, type=int)
        parser.add_arg("CAMERA_HEIGHT", default=720, type=int)
        parser.add_arg("MODEL_DIMS", default=(640, 640), type=tuple)
        parser.add_arg("CONFIDENCE_THRESHOLD", default=0.3, type=float)
        parser.add_arg("IOU_THRESHOLD", default=0.25, type=float)
        parser.add_arg("CLASSES", default=None, type=list)
        parser.add_arg("SANTA_HAT_PLUGIN", default=False, type=bool)
        args = parser.parse_args()

        self.camera_width = args.CAMERA_WIDTH
        self.camera_height = args.CAMERA_HEIGHT
        self.model_dims = args.MODEL_DIMS
        self.conf_thres = args.CONFIDENCE_THRESHOLD
        self.iou_thres = args.IOU_THRESHOLD
        self.classes = args.CLASSES
        self.santa_hat_plugin = args.SANTA_HAT_PLUGIN

    def execute(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """
        Execute the model on the given inference requests.

        This method handles the post-processing of detection results, including
        non-maximum suppression. It supports both CPU and GPU execution paths.

        Args:
            requests: List of inference requests, each containing input tensors
                     with detection results to be post-processed.

        Returns:
            List of inference responses containing the processed bounding boxes
            after applying non-maximum suppression.

        Note:
            The method automatically handles conversion between DLPack tensors
            and PyTorch/NumPy arrays based on the execution device (CPU/GPU).
        """

        if self.cuda:
            detections = from_dlpack(
                pb_utils.get_input_tensor_by_name(
                    requests[0], self.inputs[0]
                ).to_dlpack()
            )

            bounding_boxes = self.non_max_suppression(detections=detections)

            outputs_tensors = [
                pb_utils.Tensor.from_dlpack(
                    self.outputs[0], to_dlpack(bounding_boxes)
                )
            ]

        else:
            detections = pb_utils.get_input_tensor_by_name(
                requests[0], self.inputs[0]
            ).as_numpy()

            bounding_boxes = self.non_max_suppression(detections=detections)

            outputs_tensors = [
                pb_utils.Tensor.from_dlpack(
                    self.outputs[0], to_dlpack(bounding_boxes)
                )
            ]

        return [pb_utils.InferenceResponse(output_tensors=outputs_tensors)]

    def finalize(self) -> None:
        """
        Cleanup method called when the model is unloaded.

        This method performs any necessary cleanup operations before the model
        is unloaded from memory. Currently just logs the cleanup operation.
        """

        self.logger.log_info(f"Cleaning up {self.model_name}...")

    def non_max_suppression(
        self,
        detections: Union[torch.Tensor, np.ndarray],
    ) -> np.ndarray:
        """
        Apply non-maximum suppression to detection results.

        This method processes raw detection results by filtering based on confidence scores
        and applying non-maximum suppression to remove overlapping boxes. It supports both
        CPU (NumPy) and GPU (PyTorch) implementations.

        Args:
            detections: Detection tensor/array of shape (batch_size, num_classes + 4 + num_masks, ...)
                containing the following components:
                - First 4 channels: bounding box coordinates in (x_center, y_center, width, height) format
                - Next num_classes channels: class confidence scores
                - Remaining channels: mask coefficients

        Returns:
            np.ndarray: Array of shape (num_detections, 6 + num_masks) containing the filtered
            and processed detections in the format:
            [x1, y1, x2, y2, confidence, class_id, mask_coefficients...]

        Note:
            The method uses several class attributes:
            - self.conf_thres: Confidence threshold for filtering detections
            - self.iou_thres: IoU threshold for NMS
            - self.classes: Optional list of classes to keep
            - self.santa_hat_plugin: Whether to normalize output coordinates
            - self.camera_width/height: Original image dimensions
            - self.model_dims: Model input dimensions
            - self.cuda: Whether to use GPU acceleration

        The method handles the following steps:
        1. Filters detections based on confidence threshold
        2. Converts boxes from (x,y,w,h) to (x1,y1,x2,y2) format
        3. Applies NMS to remove overlapping boxes
        4. Rescales boxes to original image dimensions
        5. Optionally normalizes coordinates
        """

        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        classes = self.classes
        normalize = self.santa_hat_plugin

        bs = detections.shape[0]
        nc = detections.shape[1] - 4
        nm = detections.shape[1] - nc - 4
        mi = 4 + nc

        if self.cuda:
            xc = detections[:, 4:mi].amax(1) > conf_thres

            detections = detections.transpose(-1, -2)
            detections[..., :4] = self._xywh2xyxy(detections[..., :4])

            output = [torch.zeros((0, 6 + nm), device=detections.device)] * bs
            for xi, x in enumerate(detections):
                x = x[xc[xi]]

                if not x.shape[0]:
                    continue

                box, cls, mask = x.split((4, nc, nm), 1)

                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[
                    conf.view(-1) > conf_thres
                ]

                if classes is not None:
                    x = x[
                        (
                            x[:, 5:6] == torch.tensor(classes, device=x.device)
                        ).any(1)
                    ]

                n = x.shape[0]
                if not n:
                    continue
                if n > 30000:
                    x = x[x[:, 4].argsort(descending=True)[:30000]]

                c = x[:, 5:6] * 7680
                scores = x[:, 4]

                boxes = x[:, :4] + c
                i = nms(boxes, scores, iou_thres)
                i = i[:300]

                output[xi] = x[i]

            output = output[0]

            img0_shape = (self.camera_width, self.camera_height)
            img1_shape = self.model_dims

            gain = min(
                img1_shape[0] / img0_shape[1], img1_shape[1] / img0_shape[0]
            )
            pad = (
                (img1_shape[1] - img0_shape[0] * gain) / 2,
                (img1_shape[0] - img0_shape[1] * gain) / 2,
            )

            output[:, [0, 2]] -= pad[0]
            output[:, [1, 3]] -= pad[1]
            output[:, :4] /= gain

            output[..., 0].clamp_(0, img0_shape[0])
            output[..., 1].clamp_(0, img0_shape[1])
            output[..., 2].clamp_(0, img0_shape[0])
            output[..., 3].clamp_(0, img0_shape[1])

            if normalize:
                output[..., :4] = torch.mm(
                    output[..., :4],
                    torch.diag(
                        torch.Tensor(
                            [
                                1 / img0_shape[0],
                                1 / img0_shape[1],
                                1 / img0_shape[0],
                                1 / img0_shape[1],
                            ]
                        )
                    ),
                )

            return output.numpy()

        else:
            xc = np.amax(detections[:, 4:mi], axis=1) > conf_thres

            detections = np.transpose(detections, (0, 2, 1))
            detections[..., :4] = self._xywh2xyxy(detections[..., :4])

            output = [np.zeros((0, 6 + nm), dtype=np.float32)] * bs

            for xi, x in enumerate(detections):
                x = x[xc[xi]]

                if x.shape[0] == 0:
                    continue

                box = x[:, :4]
                cls = x[:, 4 : 4 + nc]
                mask = x[:, 4 + nc :]

                conf = np.max(cls, axis=1, keepdims=True)
                j = np.argmax(cls, axis=1, keepdims=True).astype(np.float32)

                x = np.concatenate((box, conf, j, mask), axis=1)
                x = x[conf.reshape(-1) > conf_thres]

                if classes is not None:
                    x = x[
                        (
                            x[:, 5:6]
                            == np.array(classes, dtype=np.float32).reshape(
                                -1, 1
                            )
                        ).any(1)
                    ]

                n = x.shape[0]
                if n == 0:
                    continue
                if n > 30000:
                    idx = np.argsort(x[:, 4])[::-1][:30000]
                    x = x[idx]

                c = x[:, 5:6] * 7680
                scores = x[:, 4]
                boxes = x[:, :4] + c
                i = self._nms_numpy(boxes, scores)
                i = i[:300]

                output[xi] = x[i]

            output = output[0]

            img0_shape = (self.camera_width, self.camera_height)
            img1_shape = self.model_dims

            gain = min(
                img1_shape[0] / img0_shape[1], img1_shape[1] / img0_shape[0]
            )
            pad = (
                (img1_shape[1] - img0_shape[0] * gain) / 2,
                (img1_shape[0] - img0_shape[1] * gain) / 2,
            )

            output[:, [0, 2]] -= pad[0]
            output[:, [1, 3]] -= pad[1]
            output[:, :4] /= gain

            # Clip to image bounds
            output[..., 0] = np.clip(output[..., 0], 0, img0_shape[0])
            output[..., 1] = np.clip(output[..., 1], 0, img0_shape[1])
            output[..., 2] = np.clip(output[..., 2], 0, img0_shape[0])
            output[..., 3] = np.clip(output[..., 3], 0, img0_shape[1])

            if normalize:
                scale_matrix = np.array(
                    [
                        [1 / img0_shape[0], 0, 0, 0],
                        [0, 1 / img0_shape[1], 0, 0],
                        [0, 0, 1 / img0_shape[0], 0],
                        [0, 0, 0, 1 / img0_shape[1]],
                    ],
                    dtype=np.float32,
                )
                output[..., :4] = np.dot(output[..., :4], scale_matrix)

            return output

    def _xywh2xyxy(
        self, x: Union[torch.Tensor, NDArray[np.float32]]
    ) -> Union[torch.Tensor, NDArray[np.float32]]:
        """
        Convert bounding box coordinates from (x_center, y_center, width, height) to (x1, y1, x2, y2) format.

        This function transforms bounding box coordinates from center-width-height format (XYWH)
        to corner format (XYXY). The conversion can handle both CUDA-enabled PyTorch tensors
        and CPU-based NumPy arrays or PyTorch tensors.

        Args:
            x: Input array/tensor of shape (..., 4) where the last dimension contains
            coordinates in (x_center, y_center, width, height) format.
            Can be either a CUDA PyTorch tensor or a CPU-based array/tensor.

        Returns:
            Array/tensor of same type and shape as input, with coordinates converted
            to (x1, y1, x2, y2) format where (x1, y1) is the top-left corner and
            (x2, y2) is the bottom-right corner.

        Note:
            The function assumes self.cuda is set as a class attribute indicating
            whether CUDA computation should be used.

        Example:
            >>> # For CUDA tensor
            >>> boxes_xywh = torch.tensor([[10, 10, 20, 20]], device='cuda')
            >>> boxes_xyxy = self._xywh2xyxy(boxes_xywh)  # Returns [[0, 0, 20, 20]]
            >>>
            >>> # For CPU array
            >>> boxes_xywh = np.array([[10, 10, 20, 20]])
            >>> boxes_xyxy = self._xywh2xyxy(boxes_xywh)  # Returns [[0, 0, 20, 20]]
        """

        if self.cuda:
            y = torch.empty_like(x)
            dw = x[..., 2] / 2
            dh = x[..., 3] / 2
            y[..., 0] = x[..., 0] - dw
            y[..., 1] = x[..., 1] - dh
            y[..., 2] = x[..., 0] + dw
            y[..., 3] = x[..., 1] + dh
            return y

        else:
            y = x.copy()
            y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
            y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
            y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
            y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
            return y

    def _nms_numpy(
        self, boxes: NDArray[np.float32], scores: NDArray[np.float32]
    ) -> NDArray[np.int32]:
        """
        Perform non-maximum suppression on a set of bounding boxes and corresponding scores.

        This function implements the non-maximum suppression (NMS) algorithm to filter
        overlapping bounding boxes. It keeps the boxes with highest scores and removes
        boxes that have high intersection-over-union (IoU) with already selected boxes.

        Args:
            boxes: Array of shape (N, 4) where N is the number of boxes. Each box is
                represented as [x1, y1, x2, y2] in absolute coordinates.
            scores: Array of shape (N,) containing the confidence scores for each box.

        Returns:
            Array of indices of the selected boxes, sorted by score in descending order.
            The indices correspond to the original positions in the input arrays.

        Note:
            The function assumes self.iou_thres is set as a class attribute, defining
            the IoU threshold above which boxes are considered to overlap significantly.

        Example:
            >>> boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
            >>> scores = np.array([0.9, 0.8], dtype=np.float32)
            >>> nms_indices = self._nms_numpy(boxes, scores)
        """

        idx = scores.argsort()[::-1]
        keep = []

        while idx.size > 0:
            current = idx[0]
            keep.append(current)

            if idx.size == 1:
                break

            idx = idx[1:]
            current_box = boxes[current]
            remaining_boxes = boxes[idx]

            xx1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            yy1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            xx2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            yy2 = np.minimum(current_box[3], remaining_boxes[:, 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            intersection = w * h
            box1_area = (current_box[2] - current_box[0]) * (
                current_box[3] - current_box[1]
            )
            box2_area = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (
                remaining_boxes[:, 3] - remaining_boxes[:, 1]
            )
            union = box1_area + box2_area - intersection

            iou = intersection / union
            idx = idx[iou <= self.iou_thres]

        return np.array(keep, dtype=np.int32)
