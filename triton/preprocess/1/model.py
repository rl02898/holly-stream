import json
import os
from ast import literal_eval
from typing import Any, Dict, List, Tuple, Type, TypeVar, Union

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
    def initialize(self, args: Dict[str, Any]) -> None:
        self.logger = pb_utils.Logger
        self.model_name = args["model_name"]
        model_config = json.loads(args["model_config"])
        self.inputs: List[str] = [input["name"] for input in model_config["input"]]
        self.outputs: List[str] = [output["name"] for output in model_config["output"]]

        # Parse model dimensions
        load_dotenv()
        parser = EnvArgumentParser()
        parser.add_arg("MODEL_DIMS", default=(640, 640), type=tuple)
        parser.add_arg("CAMERA_HEIGHT", default=720, type=int)
        parser.add_arg("CAMERA_WIDTH", default=1280, type=int)
        args = parser.parse_args()

        self.model_dims: Tuple[int, int] = args.MODEL_DIMS
        self.model_height = self.model_dims[0]
        self.model_width = self.model_dims[1]
        self.camera_height = args.CAMERA_HEIGHT
        self.camera_width = args.CAMERA_WIDTH

        self.device = torch.device("cuda")
        self.register_gpu_constants()

    def register_gpu_constants(self) -> None:
        """Pre-compute and store all constants on GPU"""
        h_ratio = self.model_height / self.camera_height
        w_ratio = self.model_width / self.camera_width
        self.scale = min(h_ratio, w_ratio)

        self.resize_height = int(round(self.camera_height * self.scale))
        self.resize_width = int(round(self.camera_width * self.scale))

        pad_w = (self.model_width - self.resize_width) // 2
        pad_h = (self.model_height - self.resize_height) // 2

        self.padding = (
            pad_w,
            self.model_width - self.resize_width - pad_w,
            pad_h,
            self.model_height - self.resize_height - pad_h,
        )

        self.pad_value = torch.tensor([114 / 255], dtype=torch.float16).to(self.device)
        self.flip_idx = torch.tensor([2, 1, 0]).to(self.device)
        self.scale_factor = torch.tensor([1.0 / 255.0], dtype=torch.float16).to(
            self.device
        )

    def execute(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        input_tensor = pb_utils.get_input_tensor_by_name(
            requests[0], self.inputs[0]
        ).as_numpy()
        torch_tensor = torch.from_numpy(input_tensor).cuda()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            resized_image = self._resize_image(torch_tensor)

        return [
            pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor.from_dlpack(
                        self.outputs[0], to_dlpack(resized_image)
                    )
                ]
            )
        ]

    def _resize_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(dtype=torch.float16).mul_(self.scale_factor)

        image = image.permute(2, 0, 1).unsqueeze(0)
        image = F.interpolate(
            image,
            size=(self.resize_height, self.resize_width),
            mode="bilinear",
            align_corners=False,
        )

        image = F.pad(image, self.padding, mode="constant", value=self.pad_value.item())

        return image[:, self.flip_idx].contiguous().half()

    def finalize(self) -> None:
        self.cached_tensors.clear()
        torch.cuda.empty_cache()
        self.logger.log_info(f"Cleaning up {self.model_name}...")
