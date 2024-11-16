import json
import os
from ast import literal_eval
from typing import Any, Dict, List, Tuple, Type, TypeVar, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils
from c_python_backend_utils import InferenceRequest, InferenceResponse
from dotenv import load_dotenv
from torch.utils.dlpack import from_dlpack, to_dlpack

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
    Triton Python model for image preprocessing.

    This model handles image resizing operations in both CPU and GPU contexts,
    supporting both PyTorch (GPU) and NumPy (CPU) processing paths.

    Attributes:
        logger: Triton logger instance for model-specific logging
        model_name: Name of the model instance
        inputs: List of input tensor names defined in model config
        outputs: List of output tensor names defined in model config
        cuda: Boolean flag indicating if model should use GPU processing
        model_dims: Tuple of (height, width) for target image dimensions
    """

    def initialize(self, args: Dict[str, Any]) -> None:
        """
        Initialize the model with configuration parameters.

        This method is called once when the model loads. It sets up logging,
        processes the model configuration, and loads environment variables.

        Args:
            args: Dictionary containing initialization parameters:
                - model_name: Name of the model
                - model_config: JSON string containing model configuration
                    Must include:
                    - inputs: List of input tensor specifications
                    - outputs: List of output tensor specifications
                    - instance_group: List of instance configurations
                        including 'kind' field for GPU/CPU specification

        Note:
            Expects MODEL_DIMS environment variable to be set or uses
            default value of (640, 640).
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
        parser.add_arg("MODEL_DIMS", default=(640, 640), type=tuple)
        args = parser.parse_args()

        self.model_dims: Tuple[int, int] = args.MODEL_DIMS

    def execute(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """
        Execute preprocessing on a batch of requests.

        This method handles the main inference logic, processing images either
        through a CUDA-enabled PyTorch path or a CPU NumPy path based on the
        initialization configuration.

        Args:
            requests: List of InferenceRequest objects, each containing:
                - Input tensors accessible by name
                - Other request-specific metadata

        Returns:
            List of InferenceResponse objects containing:
                - Processed output tensors
                - Any additional response metadata

        Note:
            Currently processes only the first request in the batch.
            The processing path (CUDA/CPU) is determined by self.cuda flag.
        """

        if self.cuda:
            image = from_dlpack(
                pb_utils.get_input_tensor_by_name(
                    requests[0], self.inputs[0]
                ).to_dlpack()
            )

            resized_image = self._resize_image(image=image)

            outputs_tensors = [
                pb_utils.Tensor.from_dlpack(
                    self.outputs[0], to_dlpack(resized_image)
                )
            ]

        else:
            image = pb_utils.get_input_tensor_by_name(
                requests[0], self.inputs[0]
            ).as_numpy()

            resized_image = self._resize_image(image=image)

            outputs_tensors = [pb_utils.Tensor(self.outputs[0], resized_image)]

        return [pb_utils.InferenceResponse(output_tensors=outputs_tensors)]

    def finalize(self) -> None:
        """
        Cleanup method called when the model is unloaded.

        This method performs any necessary cleanup operations before the model
        is unloaded from memory. Currently just logs the cleanup operation.
        """

        self.logger.log_info(f"Cleaning up {self.model_name}...")

    def _resize_image(
        self,
        image: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Resize and pad an image to a target size while maintaining aspect ratio.

        This function handles two specific cases:
        1. PyTorch tensors in float16 with CUDA processing
        2. NumPy arrays in float32 with CPU processing

        Args:
            image: Input image in one of two formats:
                - PyTorch tensor (CHW) in torch.float16 when cuda_available=True
                - NumPy array (HWC) in np.float32 when cuda_available=False

        Returns:
            Resized and padded image with an extra dimension added at axis 0:
                - If self.cuda=True: torch.Tensor[float16] of shape (1, C, H, W)
                - If self.cuda=False: np.ndarray[float32] of shape (1, C, H, W)
            The output is normalized to [0, 1] range.

        Raises:
            TypeError: If input type doesn't match the processing mode:
                - self.cuda=True requires torch.float16 input
                - self.cuda=False requires np.float32 input

        Note:
            - When using CUDA, the image channels are flipped (RGB -> BGR or vice versa)
            - The padding value is set to 114 (normalized to [0, 1] range)
            - The aspect ratio is preserved during resizing
        """

        new_shape = self.model_dims

        if self.cuda:
            _, height, width = image.shape

            r = min(new_shape[0] / height, new_shape[1] / width)
            new_unpad = int(round(width * r)), int(round(height * r))

            if (height, width) != new_unpad[::-1]:
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=new_unpad[::-1],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
            dw /= 2
            dh /= 2

            top = int(round(dh - 0.1))
            bottom = int(round(dh + 0.1))
            left = int(round(dw - 0.1))
            right = int(round(dw + 0.1))

            padding = (left, right, top, bottom)

            image = F.pad(
                image.unsqueeze(0), padding, mode="constant", value=114 / 255
            ).squeeze(0)

            image = torch.flip(image, [0])
            image = image.unsqueeze(0)

            return image

        else:
            shape = image.shape[:2]
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = (
                new_shape[1] - new_unpad[0],
                new_shape[0] - new_unpad[1],
            )

            dw /= 2
            dh /= 2

            if shape[::-1] != new_unpad:
                image = cv2.resize(
                    image, new_unpad, interpolation=cv2.INTER_LINEAR
                )

            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

            pad_value = np.array([114, 114, 114], dtype=np.float32) / 255

            image = cv2.copyMakeBorder(
                image,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=tuple(pad_value),
            )

            image = image.transpose((2, 0, 1))[::-1]
            return np.ascontiguousarray(image[None])
