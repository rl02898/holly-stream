import os
import subprocess
import time
from ast import literal_eval
from typing import Any, Dict, Type

import cv2
from dotenv import load_dotenv
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
from skimage.metrics import structural_similarity


class EnvArgumentParser():
    """
    A class for parsing environment variables as arguments with most
        Python types.
    """
    def __init__(self):
        self.dict: Dict[str, Any] = {}

    class _define_dict(dict):
        """
        A custom dictionary subclass for accessing arguments as
            attributes.
        """
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def add_arg(self, variable: str, default: Any = None, d_type: Type = str) -> None:
        """
        Add an argument to be parsed from an environment variable.

        Args:
            variable (str): The name of the environment variable.
            default (Any): The default value if the environment
                variable is not set.
            d_type (Type): The expected data type of the argument.
        """
        env = os.environ.get(variable)
        if env is None:
            try:
                if isinstance(default, d_type):
                    value = default
                else:
                    raise TypeError(f"The default value for {variable} cannot be cast to the data type provided.")
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
            ValueError: If the argument does not match the given data
                type or is not supported.
        """
        if d_type in [list, tuple, bool, dict]:
            try:
                cast_value = literal_eval(arg)
                if not isinstance(cast_value, d_type):
                    raise TypeError(f"The value cast type ({d_type}) does not match the value given for {arg}")
            except ValueError as e:
                raise ValueError(f"Argument {arg} does not match given data type or is not supported:", str(e))
            except SyntaxError as e:
                raise SyntaxError(f"Check the types entered for arugment {arg}:", str(e))
        else:
            try:
                cast_value = d_type(arg)
            except ValueError as e:
                raise ValueError(f"Argument {arg} does not match given data type or is not supported:", str(e))
            except SyntaxError as e:
                raise SyntaxError(f"Check the types entered for arugment {arg}:", str(e))
        
        return cast_value
    
    def parse_args(self) -> '_define_dict':
        """
        Parse the added arguments from the environment variables.

        Returns:
            _define_dict: A custom dictionary containing the parsed
                arguments.
        """
        return self._define_dict(self.dict)


def main(
    stream_user: str,
    stream_ip: str,
    camera_width: int,
    camera_height: int,
    motion_threshold: float,
    video_length: int,
    videos_file_path: str,
) -> None:
    """
    Main function detect motion every second, and record a video if
        motion is detected. If you need to flip the camera add the
        following to the camera.create_video_configuration:
            transform=Transform(hflip=1, vflip=1)

    Args:
        stream_user (str): The user to the server.
        stream_ip (str): The IP address of the stream server.
        camera_width (int): The width of the camera frame.
        camera_height (int): The height of the camera frame.
        motion_threshold (float): The threshold for the difference
            between frames [0.0-1.0].
        video_length (int): The length of video to record after motion
            is detected.
        videos_file_path (str): The path on the server where the saved
            video files will be uploaded to.

    Returns:
        None
    """

    camera = Picamera2()
    camera.configure(camera.create_video_configuration(
        main={
            "size": (camera_width, camera_height),
            "format": "BGR888"
        },
    ))
    camera.controls.Brightness = 0.1
    encoder = H264Encoder(bitrate=10000000)

    counter = 0

    while True:
        camera.start()
        old_frame = None
        tracking_index = 0

        while True:
            frame = camera.capture_array()[:, :, :3]

            if tracking_index > 2:
                score, _ = structural_similarity(
                    cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    full=True
                )

                if score < motion_threshold:
                    del frame, old_frame
                    break

            old_frame = frame
            counter += 1
            tracking_index += 1
            time.sleep(1)

        camera.stop()

        print("Motion detected, recording video...")

        camera.start_recording(
            encoder,
            FfmpegOutput("output-{}.mp4".format(counter))
        )

        time.sleep(video_length)

        camera.stop_recording()

        subprocess.run([
            "scp", "output-{}.mp4".format(counter),
            "{}@{}:{}".format(stream_user, stream_ip, videos_file_path)
        ])

        subprocess.run([
            "rm", "output-{}.mp4".format(counter)
        ])


if __name__ == "__main__":
    load_dotenv()
    parser = EnvArgumentParser()
    parser.add_arg("STREAM_USER", d_type=str)
    parser.add_arg("STREAM_IP", default="127.0.0.1", d_type=str)
    parser.add_arg("CAMERA_WIDTH", default=640, d_type=int)
    parser.add_arg("CAMERA_HEIGHT", default=480, d_type=int)
    parser.add_arg("MOTION_THRESHOLD", default=0.7, d_type=float)
    parser.add_arg("VIDEO_LENGTH", default=30, d_type=int)
    parser.add_arg("VIDEOS_FILE_PATH", d_type=str)
    args = parser.parse_args()

    main(
        stream_user=args.STREAM_USER,
        stream_ip=args.STREAM_IP,
        camera_width=args.CAMERA_WIDTH,
        camera_height=args.CAMERA_HEIGHT,
        motion_threshold=args.MOTION_THRESHOLD,
        video_length=args.VIDEO_LENGTH,
        videos_file_path=args.VIDEOS_FILE_PATH,
    )
