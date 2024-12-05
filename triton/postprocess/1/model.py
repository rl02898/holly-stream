import json
import os
from ast import literal_eval
from typing import Any, Dict, List, Type, TypeVar, Union

import torch
import triton_python_backend_utils as pb_utils
from c_python_backend_utils import InferenceRequest, InferenceResponse
from dotenv import load_dotenv
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
    def initialize(self, args: Dict[str, Any]) -> None:
        self.logger = pb_utils.Logger
        self.model_name = args["model_name"]
        model_config = json.loads(args["model_config"])
        self.inputs: List[str] = [input["name"] for input in model_config["input"]]
        self.outputs: List[str] = [output["name"] for output in model_config["output"]]

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

        # Store configuration
        self.camera_width = args.CAMERA_WIDTH
        self.camera_height = args.CAMERA_HEIGHT
        self.model_dims = args.MODEL_DIMS
        self.conf_thres = args.CONFIDENCE_THRESHOLD
        self.iou_thres = args.IOU_THRESHOLD
        self.classes = args.CLASSES
        self.santa_hat_plugin = args.SANTA_HAT_PLUGIN

        # Precalculate constants
        self._initialize_constants()

    def _initialize_constants(self):
        """Precalculate constants used in processing"""
        self.img0_shape = (self.camera_width, self.camera_height)
        self.gain = min(
            self.model_dims[0] / self.img0_shape[1],
            self.model_dims[1] / self.img0_shape[0],
        )
        self.pad = torch.tensor(
            [
                (self.model_dims[1] - self.img0_shape[0] * self.gain) / 2,
                (self.model_dims[0] - self.img0_shape[1] * self.gain) / 2,
            ],
            device="cuda",
        )

        if self.santa_hat_plugin:
            self.normalize_matrix = torch.diag(
                torch.tensor(
                    [
                        1 / self.img0_shape[0],
                        1 / self.img0_shape[1],
                        1 / self.img0_shape[0],
                        1 / self.img0_shape[1],
                    ],
                    device="cuda",
                )
            )

    def execute(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            detections = from_dlpack(
                pb_utils.get_input_tensor_by_name(
                    requests[0], self.inputs[0]
                ).to_dlpack()
            )
            bounding_boxes = self.non_max_suppression(detections=detections)

        outputs_tensors = [
            pb_utils.Tensor.from_dlpack(self.outputs[0], to_dlpack(bounding_boxes))
        ]
        return [pb_utils.InferenceResponse(output_tensors=outputs_tensors)]

    def non_max_suppression(self, detections: torch.Tensor) -> torch.Tensor:
        bs = detections.shape[0]
        nc = detections.shape[1] - 4
        nm = detections.shape[1] - nc - 4
        mi = 4 + nc

        # Filter by confidence
        xc = detections[:, 4:mi].amax(1) > self.conf_thres

        # Convert boxes to xyxy format
        detections = detections.transpose(-1, -2)
        detections[..., :4] = self._xywh2xyxy(detections[..., :4])

        output = [torch.zeros((0, 6 + nm), device=detections.device)] * bs

        for xi, x in enumerate(detections):
            # Apply confidence threshold
            x = x[xc[xi]]
            if not x.shape[0]:
                continue

            # Split predictions
            box, cls, mask = x.split((4, nc, nm), 1)

            # Get maximum confidence and class
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[
                conf.view(-1) > self.conf_thres
            ]

            # Filter by class
            if self.classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(self.classes, device=x.device)).any(1)]

            n = x.shape[0]
            if not n:
                continue

            # Limit boxes if too many
            if n > 30000:
                x = x[x[:, 4].argsort(descending=True)[:30000]]

            # Apply NMS
            boxes = x[:, :4] + x[:, 5:6] * 7680
            keep = nms(boxes, x[:, 4], self.iou_thres)
            keep = keep[:300]
            output[xi] = x[keep]

        # Process output
        output = output[0]
        if output.shape[0] == 0:
            return output

        # Scale boxes back to original size
        output[:, [0, 2]] -= self.pad[0]
        output[:, [1, 3]] -= self.pad[1]
        output[:, :4] /= self.gain

        # Clip boxes to image bounds
        output[..., [0, 2]] = output[..., [0, 2]].clamp_(0, self.img0_shape[0])
        output[..., [1, 3]] = output[..., [1, 3]].clamp_(0, self.img0_shape[1])

        # Normalize if needed
        if self.santa_hat_plugin:
            output[..., :4] = torch.mm(output[..., :4], self.normalize_matrix)

        return output

    @staticmethod
    def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def finalize(self) -> None:
        torch.cuda.empty_cache()
        self.logger.log_info(f"Cleaning up {self.model_name}...")
