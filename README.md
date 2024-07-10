<img src="./app/images/logo.png" alt="Failed to load image." style="width: auto;">
This application will ingest your computers webcam feed (using ffmpeg), apply an object detection task on the feed with bounding boxes, and send that feed via RTMP to an address of your choice. You can also turn off object detection to create a simple live stream camera, good for a security system or monitoring system.

There is also a feature, added recently, that can detect motion (within a defined threshold), and record a x-second video and upload it to a remote web server. This feature can be activated the same way as the object detection task, a simple environment variable labeled either "True" or "False".

# Requirements

* FFmpeg
* Docker and the compose plugin
* At least 4GB of RAM
* Camera, at least 720p recommended
    * Your camera might require a driver installation before it can be used. If you have an ArduCam (I tested this on a IMX519 16MP ArduCam) the driver installation and camera packages can be installed [here](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/Quick-Start-Guide/).

# Prerequisites
There are a couple of recommended and required steps before you can run this application.

## (optional) A YOLOv8 model
This application utilizes a service called Nvidia Triton Inference Server, allowing for optimized inferencing. Our project uses an object detection model format called ONNX. A model is already supplied with this repository with the default 80 object for detection (see the [classes](#change-the-default-class-predictor)). You can also supply a custom YOLOv8 model. If you want to disable object detection see the [Deployment](#deployment) section.

## (optional) Train a custom YOLOv8 model
Follow these steps if you want to train a model from scratch ([official docs](https://docs.ultralytics.com/)). It is highly recommended to use a CUDA-enabled machine for training. If you already have a trained model in PyTorch format (`.pt`) skip to step 5.

1. Gather data. Collect images of the object(s) you want to detect. I created a script at `app/collect_data.py` to make this process easier.

2. Annotate your data in YOLO format. I highly recommend [Roboflow](https://roboflow.com/) and using a 70/30 split between your training and validation data.

3. Set up a python environment and install the YOLOv8 requirements:
    ```bash
    pip install ultralytics
    ```

4. Train your model. A CUDA-enabled machine is highly encouraged for this step. A training script will look something like this:
    ```python
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    model.train(
        data='data.yaml',
        epochs=1000,
        batch=16,
        device=0,
        optimizer='AdamW'
    )
    ```
    I had to hardcode the full path for train/val in my `data.yaml` for YOLOv8 to find the training data (e.g. `train: /home/user/holly-stream/train/images`).

5. After training, export your model as ONNX. 
    ```python
    from ultralytics import YOLO
    model = YOLO('runs/detect/train/weights/best.pt')
    model.export(
        format='onnx',   # export to ONNX
        half=True,       # FP16 quantization
        simplify=True    # ONNX model simplification
    )
    ```
    The model will be saved in the same directory your `best.pt` model is.

7. Move the model to the correct location, `holly-stream/triton/object_detection/1/model.onnx`. You may have to tinker with the config at `holly-stream/triton/object-detection/config.pbtxt` depending on the shape of your model inputs/outputs and corresponding names.

# Installation

Run the setup script provided. This will install the environment to run the main application.
```bash
./app/setup.sh
```

Pull the Triton image. This service is containerized and only needs Docker to run. If you only plan on live streaming without object detection, you will not need this image.
```bash
docker pull rcland12/detection-stream:raspbian-triton-latest
```

# Deployment
You have the following options for serving this feed:

1. To the same device (localhost).
2. To another local device (LAN).
3. To a remote web server (WAN).

You have two options for reading in this stream (client):

1. [Media player (VLC, Windows Media Player, etc.)](#watching-stream-through-streaming-software)
2. [Web page via Nginx/HLS.](#watching-stream-through-web-browser)

Pick an option from each list and follow the instructions below to deploy. If you are new to object detection I recommend you stick to the default model (section 1 from Prerequisites). Otherwise, you can supply your own YOLOv8 model (section 2 from Prerequisites).

---

You must have a camera attached to your Raspberry Pi. Create an `.env` file to define parameters that will be used by the streaming service and Triton. If you do not define any parameters it will default to a value given below, but YOU MUST DEFINE AN EMPTY `.env` FILE AT MINIMUM with the variable OBJECT_DETECTION (True/False).

```bash
touch .env
```

Here is a list of all possible arguments:

```bash
OBJECT_DETECTION=True
MODEL_NAME=yolov8n
MODEL_DIMS=(640, 640)
MODEL_REPOSITORY=/root/app/triton
CONFIDENCE_THRESHOLD=0.3
IOU_THRESHOLD=0.25
CLASSES=[0, 1]

# Used if your Triton model repo is hosted in s3
AWS_ACCESS_KEY_ID=<aws_access_key_id>
AWS_SECRET_ACCESS_KEY=<aws_secret_access_key>
AWS_DEFAULT_REGION=<aws_default_region>

STREAM_USER=username
STREAM_IP=127.0.0.1
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=stream

CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30
SANTA_HAT_PLUGIN=False

MOTION_DETECTION=False
MOTION_THRESHOLD=0.75
VIDEO_LENGTH=30
VIDEOS_FILE_PATH=/app/videos
```

A few comments about the parameters:
- The `OBJECT_DETECTION` variable is a boolean to turn that tasks on/off. If you turn it off you simply have a live stream feed. This is required.
- The `MODEL_NAME` variable is the name of the ensemble model in Triton.
- The `MODEL_DIMS` variable is the shape of your Yolov8 model inputs (e.g. (1280, 720)) in Python tuple format.
- The `MODEL_REPOSITORY` is the path in the Triton container to the model repository. You can also provide an s3 bucket path for this variable (e.g. `s3://example-s3-models-path/`). This model repository must have the same structure as the `triton/` directory in this repo.
- The `CONFIDENCE_THRESHOLD` and `IOU_THRESHOLD` variables are the hyperparameters used in the non-maximum supression algorithm in the postprocess Triton model.
- The `CLASSES` variable takes in a python list format. If you wish to include all possible classes, remove it from the `.env` file. The possible classes for the default model are [listed below](#change-the-default-class-predictor).
- If you provide an s3 bucket for `MODEL_REPOSITORY`, you must also provided the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION`.
---
- All arguments accept the data type present above. `STREAM_IP` takes a string, `STREAM_PORT` takes an integer, `STREAM_APPLICATION` takes a string, etc.
- If you are watching the stream on the same device set the `STREAM_IP` to `127.0.0.1`, or whatever localhost address you want.
- If you are streaming the feed to another device on your local network, set the `STREAM_IP` to the IPV4 address of that device. To find the IP address of a device on linux machine run `ip a` and look for it under `wl01` or something similar. On a windows machine open Windows Powershell and run `ipconfig` and look for `IP address`.
- If you are streaming the feed to another server set the `STREAM_IP` to the public IP address for that server. Also make sure you expose port 1935 on the firewall and router if necessary for the server your sending the stream to.
---
- The variable `SANTA_HAT_PLUGIN`, if set to True, will not add a bounding box, but a santa hat to the object with the highest probability score. I use this parameter when detecting my dog using a custom model, especially around Christmas!

Lastly, to receive the stream on the device you picked above you have two options:

1. [Watch the stream on streaming software such as VLC, Windows Media Player, OBS.](#watching-stream-through-streaming-software)
2. [Watch the stream through a browser.](#watching-stream-through-web-browser)

## Watching stream through streaming software
The client machine you are using must have Docker and the compose plugin. You will now launch a container that will pick up the feed and send it to your streaming software. If you already have Nginx running on port 1935 on your machine you will have to stop that service before you start this one. Also consider you may need to open port 1935 inbound on your client machine. Otherwise start this service:
```bash
docker compose up -d nginx-stream
```

Once the client software is running you can launch the streaming application from the server side (Raspberry Pi):
```bash
./run.sh
```

Lastly, on the client side you can open up your streaming software and find where you can watch a network stream or URL stream, then use the address you set up in the parameters:
```bash
rtmp://127.0.0.1:<STREAM_PORT>/<STREAM_APPLICATION>/<STREAM_KEY>

# example
rtmp://127.0.0.1:1935/live/stream
```

If `127.0.0.1` does not work, try `0.0.0.0`.

To stop the running services on the server (Raspberry Pi), run:
```bash
./stop.sh
```

To stop the running services on the client, run:
```bash
docker compose down
```

## Watching stream through web browser
The client machine you are using must have Docker and the compose plugin. You will now launch a container that will start a web server on localhost port 80. If you already have Nginx running on port 1935 on your machine you will have to stop that service before you start this one. Also consider you may need to open port 1935 inbound on your client machine. Otherwise start this service:
```bash
docker compose up -d nginx-web
```

Once the client software is running you can launch the streaming application from the server side (Raspberry Pi):
```bash
./run.sh
```

Lastly, on the client side navigate to the web address `http://localhost/index.html`. If you plan on serving this stream on a live web server, following the instructions in the next section.

To stop the running services on the server (Raspberry Pi), run:
```bash
./stop.sh
```

To stop the running services on the client, run:
```bash
docker compose down
```

## Streaming the feed to a web server
If you are streaming to a remote web server most of the steps will be the same. You will clone the repo on your web server. Before you launch the service you will have to make a few changes.

File `nginx/stream/index.html`:
- Everywhere there is a `http://localhost`, replace it with your domain name: `https://website.com`.
- On line 20, replace the last part of the src, `stream.m3u8`, with your stream key: `<stream_key>.m3u8`. This prevents people on your same network from streaming a video to your website.

File `nginx/nginx-web/nginx.conf`:
- Whitelist your IP address by first finding your home IP address (you can do so [here](https://whatismyipaddress.com/)).
- Under the rtmp block, aroud lines 40-43 add another line with `allow publish <your_ipv4>;`. Don't forget the semicolon.
- Add your domain name on line 27. Replace `server_name localhost;` with `server_name website.com;`. Don't forget the semicolon.
- If for some reason you want to change the application name edit line 46 by replacing `application live` with `application <application_name>`. Then remember to also make that change in your `.env` file for the variable STREAM_APPLICATION.

Now you can launch this service on your web server:
```bash
docker compose run -d nginx-web
```

Once the client software is running you can launch the streaming application from the server side (Raspberry Pi):
```bash
./run.sh
```

Lastly, you will be able to access the stream at `https://website.com/index.html`. You can of course make changes to this and create a different route for this stream, but this is the minimum requirements.

To stop the running services on the server (Raspberry Pi), run:
```bash
./stop.sh
```

To stop the running services on the client, run:
```bash
docker compose down
```

## Change the default class predictor
By default this application detects 80 different classes. To change or add classes for detection, add a CLASSES environmental variable to your `.env` file, if you don't already have it. Remove it to inference on all classes below. Otherwise, use a list to add classes you want to inference on. Such as `CLASSES=[0, 16, 17, 54, 67]`.

| class_index  | class_name     |
|--------------|----------------|
| 0            | person         |
| 1            | bicycle        |
| 2            | car            |
| 3            | motorcycle     |
| 4            | airplane       |
| 5            | bus            |
| 6            | train          |
| 7            | truck          |
| 8            | boat           |
| 9            | traffic light  |
| 10           | fire hydrant   |
| 11           | stop sign      |
| 12           | parking meter  |
| 13           | bench          |
| 14           | bird           |
| 15           | cat            |
| 16           | dog            |
| 17           | horse          |
| 18           | sheep          |
| 19           | cow            |
| 20           | elephant       |
| 21           | bear           |
| 22           | zebra          |
| 23           | giraffe        |
| 24           | backpack       |
| 25           | umbrella       |
| 26           | handbag        |
| 27           | tie            |
| 28           | suitcase       |
| 29           | frisbee        |
| 30           | skis           |
| 31           | snowboard      |
| 32           | sports ball    |
| 33           | kite           |
| 34           | baseball bat   |
| 35           | baseball glove |
| 36           | skateboard     |
| 37           | surfboard      |
| 38           | tennis racket  |
| 39           | bottle         |
| 40           | wine glass     |
| 41           | cup            |
| 42           | fork           |
| 43           | knife          |
| 44           | spoon          |
| 45           | bowl           |
| 46           | banana         |
| 47           | apple          |
| 48           | sandwich       |
| 49           | orange         |
| 50           | brocolli       |
| 51           | carrot         |
| 52           | hot dog        |
| 53           | pizza          |
| 54           | donut          |
| 55           | cake           |
| 56           | chair          |
| 57           | couch          |
| 58           | potted plant   |
| 59           | bed            |
| 60           | dining table   |
| 61           | toilet         |
| 62           | tv             |
| 63           | laptop         |
| 64           | mouse          |
| 65           | remote         |
| 66           | keyboard       |
| 67           | cell phone     |
| 68           | microwave      |
| 69           | oven           |
| 70           | toaster        |
| 71           | sink           |
| 72           | refrigerator   |
| 73           | book           |
| 74           | clock          |
| 75           | vase           |
| 76           | scissors       |
| 77           | teddy bear     |
| 78           | hair drier     |
| 79           | toothbrush     |
