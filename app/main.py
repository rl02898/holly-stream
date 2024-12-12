import os
import subprocess
from ast import literal_eval
from typing import Any, Dict, List, Optional, Tuple, Type
from urllib.parse import urlparse

import cv2
import imutils
import numpy as np
from dotenv import load_dotenv


class EnvArgumentParser:
    """
    A class for parsing environment variables as arguments with most Python types.
    """

    def __init__(self):
        self.dict: Dict[str, Any] = {}

    class _define_dict(dict):
        """
        A custom dictionary subclass for accessing arguments as attributes.
        """

        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def add_arg(self, variable: str, default: Any = None, d_type: Type = str) -> None:
        """
        Add an argument to be parsed from an environment variable.

        Args:
            variable (str): The name of the environment variable.
            default (Any): The default value if the environment variable is not set.
            d_type (Type): The expected data type of the argument. Defaults to str.
        """

        env = os.environ.get(variable)
        if env is None:
            try:
                if isinstance(default, d_type):
                    value = default
                else:
                    raise TypeError(
                        f"The default value for {variable} cannot be cast to the data type provided."
                    )
            except TypeError:
                raise TypeError(f"The type you provided for {variable} is not valid.")
        else:
            if callable(d_type):
                value = self._cast_type(env, d_type)
        self.dict[variable] = value

    @staticmethod
    def _cast_type(arg: str, d_type: Type) -> Any:
        """
        Cast the argument to the specified data type.

        Args:
            arg (str): The argument value as a string.
            d_type (Type): The desired data type.

        Returns:
            Any: The argument value casted to the specified data type.

        Raises:
            ValueError: If the argument does not match the given data type or is not supported.
        """

        if d_type in [list, tuple, bool, dict]:
            try:
                cast_value = literal_eval(arg)
                if not isinstance(cast_value, d_type):
                    raise TypeError(
                        f"The value cast type ({d_type}) does not match the value given for {arg}"
                    )
            except ValueError as e:
                raise ValueError(
                    f"Argument {arg} does not match given data type or is not supported:",
                    str(e),
                )
            except SyntaxError as e:
                raise SyntaxError(
                    f"Check the types entered for arugment {arg}:", str(e)
                )
        else:
            try:
                cast_value = d_type(arg)
            except ValueError as e:
                raise ValueError(
                    f"Argument {arg} does not match given data type or is not supported:",
                    str(e),
                )
            except SyntaxError as e:
                raise SyntaxError(
                    f"Check the types entered for arugment {arg}:", str(e)
                )

        return cast_value

    def parse_args(self) -> "_define_dict":
        """
        Parse the added arguments from the environment variables.

        Returns:
            _define_dict: A custom dictionary containing the parsed arguments.
        """

        return self._define_dict(self.dict)


class TritonClient:
    """
    A client class for interacting with Triton Inference Server.

    Args:
        url (str): The URL of the Triton Inference Server.
        model (str): The name of the model to be used for inference.

    Attributes:
        client (InferenceServerClient): The Triton Inference Server client instance.
        model_name (str): The name of the model being used.
        metadata (dict): The metadata of the model.
        config (dict): The configuration of the model.
        model_dims (Tuple[int, int]): The dimensions of the model input.
        classes (Optional[List[str]]): The list of class labels, if available.

    Raises:
        RuntimeError: If an unsupported protocol is used (other than HTTP or GRPC).
    """

    def __init__(self, url: str, model: str):
        """
        Initialize the TritonClient instance.

        Args:
            url (str): The URL of the Triton Inference Server.
            model (str): The name of the model to be used for inference.
        """

        self.model_name: str = model

        parsed_url = urlparse(url)
        if parsed_url.scheme == "grpc":
            from tritonclient.grpc import InferenceServerClient, InferInput

            self.client: InferenceServerClient = InferenceServerClient(
                parsed_url.netloc
            )
            self.metadata: dict = self.client.get_model_metadata(
                self.model_name, as_json=True
            )
            self.config: dict = self.client.get_model_config(
                self.model_name, as_json=True
            )["config"]

            def create_input_placeholders() -> List[InferInput]:
                return [
                    InferInput(i["name"], [int(s) for s in i["shape"]], i["datatype"])
                    for i in self.metadata["inputs"]
                ]

        elif parsed_url.scheme == "http":
            from tritonclient.http import InferenceServerClient, InferInput

            self.client: InferenceServerClient = InferenceServerClient(
                parsed_url.netloc
            )
            self.metadata: dict = self.client.get_model_metadata(self.model_name)
            self.config: dict = self.client.get_model_config(self.model_name)

            def create_input_placeholders() -> List[InferInput]:
                return [
                    InferInput(i["name"], [int(s) for s in i["shape"]], i["datatype"])
                    for i in self.metadata["inputs"]
                ]

        else:
            raise RuntimeError("Unsupported protocol. Use HTTP or GRPC.")

        self._create_input_placeholders_fn = create_input_placeholders
        self.output_name: str = self.metadata["outputs"][0]["name"]
        self.model_dims: Tuple[int, int] = self._get_dims()
        self.classes: Optional[List[str]] = self._get_classes()

    def __call__(self, *args: Any) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Perform inference on the provided inputs and return bounding boxes, confidence scores, and class indexes.

        Args:
            *args: Input arguments for the model. Expected format depends on the model configuration
                and should be compatible with self._create_inputs().

        Returns:
            Tuple containing:
                - List[List[float]]: Bounding boxes, where each box is [x1, y1, x2, y2]
                - List[float]: Confidence scores rounded to 2 decimal places
                - List[int]: Class indexes

        Raises:
            RuntimeError: If no inputs are provided or if the number of inputs does not match the expected number.
        """

        inputs = self._create_inputs(*args)
        response = self.client.infer(model_name=self.model_name, inputs=inputs)
        predictions = response.as_numpy(self.output_name).tolist()

        bboxes = [item[:4] for item in predictions]
        confs = [round(float(item[4]), 2) for item in predictions]
        indexes = [int(item[5]) for item in predictions]

        return bboxes, confs, indexes

    def _create_inputs(self, *args):
        """
        Create input placeholders for the model.

        Args:
            *args: The input arguments for the model.

        Returns:
            List[InferInput]: The list of input placeholders.

        Raises:
            RuntimeError: If no inputs are provided or if the number of inputs does not match the expected number.
        """

        args_len = len(args)
        if not args_len:
            raise RuntimeError("No inputs provided.")

        placeholders = self._create_input_placeholders_fn()

        if args_len:
            if args_len != len(placeholders):
                raise RuntimeError(
                    f"Expected {len(placeholders)} inputs, got {args_len}."
                )
            for input, value in zip(placeholders, args):
                input.set_data_from_numpy(value)

        return placeholders

    def _get_classes(self) -> Optional[List[str]]:
        """
        Get the class labels of the model, if available.

        Returns:
            Optional[List[str]]: The list of class labels, or None if not available.
        """

        label_filename = self.config["output"][0]["label_filename"]
        docker_file_path = f"/root/app/triton/{self.model_name}/{label_filename}"
        local_file_path = os.path.join(
            os.path.abspath(os.getcwd()),
            f"triton/{self.model_name}/{label_filename}",
        )

        if os.path.isfile(docker_file_path):
            with open(docker_file_path, "r") as file:
                classes = file.read().splitlines()
        elif os.path.isfile(local_file_path):
            with open(local_file_path, "r") as file:
                classes = file.read().splitlines()
        else:
            classes = None

        return classes

    def _get_dims(self) -> Tuple[int, int]:
        """
        Get the dimensions of the model input.

        Returns:
            Tuple[int, int]: The dimensions of the model input.
        """

        try:
            model_dims = tuple(self.config["input"][0]["dims"][2:4])
            return tuple(map(int, model_dims))
        except Exception:
            return (640, 640)


class Annotator:
    """
    A class for annotating frames with bounding boxes, class labels, and confidence scores.

    Args:
        classes (List[str]): A list of class labels.
        width (int): The width of the frame. Defaults to 1280.
        height (int): The height of the frame. Defaults to 720.
        santa_hat_plugin (bool): Indicates whether to use the Santa hat plugin. Defaults to False.

    Attributes:
        width (int): The width of the frame.
        height (int): The height of the frame.
        classes (List[str]): A list of class labels.
        colors (List[Tuple[float, float, float]]): A list of randomly generated colors for each class.
        santa_hat (np.ndarray): The Santa hat image.
        santa_hat_mask (np.ndarray): The Santa hat mask image.
        santa_hat_plugin (bool): Indicates whether to use the Santa hat plugin.
    """

    def __init__(
        self,
        classes: List[str],
        width: int = 1280,
        height: int = 720,
        santa_hat_plugin: bool = False,
    ):
        """
        Initialize the Annotator instance.

        Args:
            classes (List[str]): A list of class labels.
            width (int): The width of the frame. Defaults to 1280.
            height (int): The height of the frame. Defaults to 720.
            santa_hat_plugin (bool): Indicates whether to use the Santa hat plugin. Defaults to False.
        """

        self.width = width
        self.height = height
        self.classes = classes
        self.colors = list(np.random.rand(len(self.classes), 3) * 255)
        self.santa_hat = cv2.imread("./images/santa_hat.png")
        self.santa_hat_mask = cv2.imread("./images/santa_hat_mask.png")
        self.santa_hat_plugin = santa_hat_plugin

    def __call__(
        self,
        frame: np.ndarray,
        bboxes: List[List[float]],
        confs: List[float],
        indexes: List[int],
    ) -> np.ndarray:
        """
        Annotate the frame with bounding boxes, class labels, and confidence scores.

        Args:
            frame (np.ndarray): The input frame.
            bboxes (List[List[float]]): A list of bounding box coordinates.
            confs (List[float]): A list of confidence scores.
            indexes (List[int]): A list of class indexes.

        Returns:
            np.ndarray: The annotated frame.
        """

        if not self.santa_hat_plugin:
            for i in range(len(bboxes)):
                xmin, ymin, xmax, ymax = [int(j) for j in bboxes[i]]
                color = self.colors[indexes[i]]
                frame = cv2.rectangle(
                    img=frame,
                    pt1=(xmin, ymin),
                    pt2=(xmax, ymax),
                    color=color,
                    thickness=2,
                )

                frame = cv2.putText(
                    img=frame,
                    text=f"{self.classes[indexes[i]]} ({str(confs[i])})",
                    org=(xmin, ymin - 5),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=0.75,
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            return frame

        else:
            # For santa hat plugin, turn Normalize to True in nms function
            max_index = max(range(len(confs)), key=confs.__getitem__)
            return self._overlay_obj(frame, bboxes[max_index].copy())

    def _overlay_obj(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Overlay the Santa hat on the detected object.

        Args:
            frame (np.ndarray): The input frame.
            bbox (List[float]): The bounding box coordinates of the detected object.

        Returns:
            np.ndarray: The frame with the Santa hat overlaid on the detected object.
        """

        bbox = [
            int(i * scalar)
            for i, scalar in zip(
                bbox, [self.width, self.height, self.width, self.height]
            )
        ]
        x, y = bbox[0], bbox[1] + 20

        resize_width = bbox[2] - bbox[0]
        santa_hat = imutils.resize(self.santa_hat.copy(), width=resize_width)
        santa_hat_mask = imutils.resize(self.santa_hat_mask.copy(), width=resize_width)
        hat_height, hat_width = santa_hat.shape[0], santa_hat.shape[1]

        mask_boolean = santa_hat_mask[:, :, 0] == 0
        mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)

        if x >= 0 and y >= 0:
            h = hat_height - max(0, y + hat_height - self.height)
            w = hat_width - max(0, x + hat_width - self.width)
            frame[y - h : y, x : x + w, :] = (
                frame[y - h : y, x : x + w, :] * ~mask_rgb_boolean[0:h, 0:w, :]
                + (santa_hat * mask_rgb_boolean)[0:h, 0:w, :]
            )

        elif x < 0 and y < 0:
            h = hat_height + y
            w = hat_width + x
            frame[0 : 0 + h, 0 : 0 + w, :] = (
                frame[0 : 0 + h, 0 : 0 + w, :]
                * ~mask_rgb_boolean[
                    hat_height - h : hat_height, hat_width - w : hat_width, :
                ]
                + (santa_hat * mask_rgb_boolean)[
                    hat_height - h : hat_height, hat_width - w : hat_width, :
                ]
            )

        elif x < 0 and y >= 0:
            h = hat_height - max(0, y + hat_height - self.height)
            w = hat_width + x
            frame[y : y + h, 0 : 0 + w, :] = (
                frame[y : y + h, 0 : 0 + w, :]
                * ~mask_rgb_boolean[0:h, hat_width - w : hat_width, :]
                + (santa_hat * mask_rgb_boolean)[0:h, hat_width - w : hat_width, :]
            )

        elif x >= 0 and y < 0:
            h = hat_height + y
            w = hat_width - max(0, x + hat_width - self.width)
            frame[0 : 0 + h, x : x + w, :] = (
                frame[0 : 0 + h, x : x + w, :]
                * ~mask_rgb_boolean[hat_height - h : hat_height, 0:w, :]
                + (santa_hat * mask_rgb_boolean)[hat_height - h : hat_height, 0:w, :]
            )

        return frame


def main(
    triton_url: str,
    model_name: str,
    stream_ip: str,
    stream_port: int,
    stream_application: str,
    stream_key: str,
    camera_index: int,
    camera_width: int,
    camera_height: int,
    camera_fps: int,
    santa_hat_plugin: bool,
) -> None:
    """
    Main function to run the RTMP stream, object detection and annotation pipeline.

    Args:
        triton_url (str): The URL of the Triton server.
        model_name (str): The name of the model to use for object detection.
        stream_ip (str): The IP address of the RTMP stream server.
        stream_port (int): The port number of the RTMP stream server.
        stream_application (str): The application name for the RTMP stream.
        stream_key (str): The stream key for the RTMP stream.
        camera_index (int): The index of the camera to use for video capture.
        camera_width (int): The width of the camera frame.
        camera_height (int): The height of the camera frame.
        camera_fps (int): The frames-per-second to use on camera.
        santa_hat_plugin (bool): Indicates whether to use the Santa hat plugin.

    Returns:
        None
    """

    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    camera.set(cv2.CAP_PROP_FPS, camera_fps)

    rtmp_url = "rtmp://{}:{}/{}/{}".format(
        stream_ip, stream_port, stream_application, stream_key
    )

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "{}x{}".format(camera_width, camera_height),
        "-r",
        str(camera_fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-f",
        "flv",
        rtmp_url,
    ]

    model = TritonClient(triton_url, model_name)

    annotator = Annotator(model.classes, camera_width, camera_height, santa_hat_plugin)

    period = 2
    tracking_index = 0
    p = subprocess.Popen(command, stdin=subprocess.PIPE)

    try:
        while camera.isOpened():
            ret, frame = camera.read()

            if not ret:
                print("Frame failed to load...")
                break

            if tracking_index % period == 0:
                bboxes, confs, indexes = model(frame)
                tracking_index = 0

            if bboxes:
                frame = annotator(frame, bboxes, confs, indexes)
            tracking_index += 1

            p.stdin.write(frame.tobytes())

    finally:
        camera.release()

    return


if __name__ == "__main__":
    load_dotenv()
    parser = EnvArgumentParser()
    parser.add_arg("TRITON_URL", default="http://localhost:8000", d_type=str)
    parser.add_arg("MODEL_NAME", default="yolov8n", d_type=str)
    parser.add_arg("STREAM_IP", default="127.0.0.1", d_type=str)
    parser.add_arg("STREAM_PORT", default=1935, d_type=int)
    parser.add_arg("STREAM_APPLICATION", default="live", d_type=str)
    parser.add_arg("STREAM_KEY", default="stream", d_type=str)
    parser.add_arg("CAMERA_INDEX", default=0, d_type=int)
    parser.add_arg("CAMERA_WIDTH", default=640, d_type=int)
    parser.add_arg("CAMERA_HEIGHT", default=480, d_type=int)
    parser.add_arg("CAMERA_FPS", default=30, d_type=int)
    parser.add_arg("SANTA_HAT_PLUGIN", default=False, d_type=bool)
    args = parser.parse_args()

    main(
        args.TRITON_URL,
        args.MODEL_NAME,
        args.STREAM_IP,
        args.STREAM_PORT,
        args.STREAM_APPLICATION,
        args.STREAM_KEY,
        args.CAMERA_INDEX,
        args.CAMERA_WIDTH,
        args.CAMERA_HEIGHT,
        args.CAMERA_FPS,
        args.SANTA_HAT_PLUGIN,
    )
