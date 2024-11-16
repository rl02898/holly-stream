<img src="./app/images/logo.png" alt="Failed to load image." style="width: auto;">
This application will ingest your computers webcam feed (using ffmpeg), apply an object detection task on the feed with bounding boxes, and send that feed via RTMP to an address of your choice. You can also turn off object detection to create a simple live stream camera, good for a security system or monitoring system.

# Requirements

* FFmpeg
* Docker and the compose plugin
* Webcam, at least 720p recommended
* At least 4GB of RAM
* CUDA-enabled GPU
    * If your machine does not have a GPU, you can still run Triton but you will have to use an ONNX model and modify some of the `config.pbtxt` files throughout. If you are training a custom model it is highly encouraged to use a CUDA-enabled machine with a GPU. There are options such as Google Colab to train models with GPUs if need be.

# Prerequisites
There are a couple of recommended steps before you can run this application.

## (optional) A YOLOv8 or YOLOv5 model
This application uses a service called Nvidia Trition Inference Server. It uses an object detection model format called TensorRT. THIS MODEL FORMAT IS MACHINE SPECIFIC. If I trained a YOLOv8 model on a Nvidia GeForce GTX 1060 GPU and convert it, it will not work on another GPU architecture. Therefore, you must convert your own model. If you want to use the default YOLOv8 model, you can simply follow the steps below. Otherwise, the next section will explain how to train a custom model with labels of your choice. If you want to disable object detection see the [Deployment](#deployment) section.

### (optional) How to convert a YOLOv8n (nano) model to TensorRT format

1. Set up a python environment.

2. Install YOLOv8 requirements:
    ```bash
    pip install ultralytics
    ```

3. Convert the model to ONNX format in a python script:

    ```python
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    model.export(
        format='onnx',   # export to ONNX
        half=True,       # FP16 quantization
        simplify=True    # ONNX model simplification
    )
    ```
    This model should save to `weights/yolov8n.onnx`. If you do not a GPU that is CUDA-enabled, remove the `half` argument. FP16 format is only supported for GPU. You will use this model to run in Triton, so skip the next step.

4. Launch the Nvidia TensorRT container and convert the model:

    ```bash
    docker run -it --rm --gpus all -v ./weights:/models nvcr.io/nvidia/tensorrt:23.12-py3 \
    trtexec --onnx=/models/yolov8n.onnx --saveEngine=/models/model.plan --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw
    ```
    This process should take a couple minutes.

5. Move the TensorRT model (`weights/model.plan`) to the triton directory:

    ```bash
    mv weights/model.plan triton/object_detection/1/
    ```

Triton should now be ready.

## (optional) Train a custom YOLOv8 model
Follow these steps if you want to train a model from scratch ([official docs](https://docs.ultralytics.com/)). It is highly recommended to use a CUDA-enabled machine for training. If you already have a trained model in PyTorch format (`.pt`) skip to step 4.

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

6. Launch the Nvidia TensorRT container and convert the model. Move the ONNX model to a directory called `weights/` and run the command below. It will save the TensorRT model (`.plan`) to the same directory.
    ```bash
    docker run -it --rm --gpus all -v ./weights:/models nvcr.io/nvidia/tensorrt:23.12-py3 \
    trtexec --onnx=/models/yolov8n.onnx --saveEngine=/models/model.plan --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw
    ```
    This process should take a couple minutes.

7. Move the model to the correct location, `holly-stream/triton/object_detection/1/model.plan`.

# Installation

Pull the docker images:
```bash
docker pull rcland12/detection-stream:linux-latest
docker pull rcland12/detection-stream:linux-triton-latest
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

You must have a webcam attached to your Linux machine. Create an `.env` file to define parameters you wish to change. If you do not define a parameter it will default to a value given below, but YOU MUST DEFINE AN EMPTY `.env` FILE AT MINIMUM.

```bash
touch .env
```

Here is a list of all possible arguments:

```bash
OBJECT_DETECTION=True
CLASSES=None
MODEL_NAME=yolov8n
MODEL_DIMS=(640, 640)
MODEL_REPOSITORY=/root/app/triton
AWS_ACCESS_KEY_ID=<aws_access_key_id>
AWS_SECRET_ACCESS_KEY=<aws_secret_access_key>
AWS_DEFAULT_REGION=<aws_default_region>

STREAM_IP=127.0.0.1
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=stream

CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30
CAMERA_AUDIO=False
SANTA_HAT_PLUGIN=False
```

A few comments about the parameters:
- The `OBJECT_DETECTION` variable is a boolean to turn that tasks on/off. If you turn it off you simply have a live stream feed.
- The `CLASSES` variable takes in a list. If you wish to include all possible classes, remove it from the `.env` file. The possible classes for the default model are [listed below](#change-the-default-class-predictor).
- The `MODEL_NAME` variable is the name of the ensemble model in Triton.
- The `MODEL_REPOSITORY` is the path in the Triton container to the model repository. You can also provide an s3 bucket path for this variable (e.g. `s3://example-s3-models-path/`).
- If you provide an s3 bucket for `MODEL_REPOSITORY`, you must also provided the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION`.
---
- All arguments accept the data type present above. `STREAM_IP` takes a string, `STREAM_PORT` takes an integer, `STREAM_APPLICATION` takes a string, etc.
- If you are watching the stream on the same device set the `STREAM_IP` to `127.0.0.1`, or whatever localhost address you want.
- If you are streaming the feed to another device on your local network, set the `STREAM_IP` to the IPV4 address of that device. To find the IP address of a device on linux machine run `ip a` and look for it under `wl01` or something similar. On a windows machine open Windows Powershell and run `ipconfig` and look for `IP address`.
- If you are streaming the feed to another server set the `STREAM_IP` to the public IP address for that server. Also make sure you expose port 1935 on the firewall and router if necessary for the server your sending the stream to.
---
- The variable `CAMERA_AUDIO` is a boolean to set audio on/off if your camera also has a microphone. This variable is required if `OBJECT_DETECTION` is set to False, otherwise it will do nothing.
- The variable `SANTA_HAT_PLUGIN`, if set to True, will not add a bounding box, but a santa hat to the object with the highest probability score. I use this parameter when detecting my dog using a custom model, especially around Christmas!

Lastly, to receive the stream on the device you picked above you have two options:

1. [Watch the stream on streaming software such as VLC, Windows Media Player, OBS.](#watching-stream-through-streaming-software)
2. [Watch the stream through a browser.](#watching-stream-through-web-browser)

## Watching stream through streaming software
The client machine you are using must have Docker and the compose plugin. You will now launch a container that will pick up the feed and send it to your streaming software. If you already have Nginx running on port 1935 on your machine you will have to stop that service before you start this one.
```bash
docker compose up -d nginx-stream
```

Once the client software is running you can launch the streaming application from the server side (Linux machine):
```bash
./run.sh
```

Lastly, on the client side you can open up your streaming software and find where you can watch a network stream or URL stream, then use the address you set up in the parameters:
```bash
rtmp://0.0.0.0:<STREAM_PORT>/<STREAM_APPLICATION>/<STREAM_KEY>

# example
rtmp://0.0.0.0:1935/live/stream
```

To stop the running services on the server (Raspberry Pi), run:
```bash
./stop.sh
```

To stop the running services on the client, run:
```bash
docker compose down
```

## Watching stream through web browser
The client machine you are using must have Docker and the compose plugin. You will now launch a container that will start a web server on localhost port 80. If you already have Nginx running on port 1935 on your machine you will have to stop that service before you start this one.
```bash
docker compose up -d nginx-web
```

Once the client software is running you can launch the streaming application from the server side (Linux machine):
```bash
docker compose up -d linux-stream
```

Lastly, on the client side navigate to the web address `http://localhost/stream.html`. If you plan on serving this stream on a live web server, following the instructions in the next section.

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

File `nginx/stream/stream.html`:
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

Once the client software is running you can launch the streaming application from the server side (Linux machine):
```bash
./run.sh
```

Lastly, you will be able to access the stream at `https://website.com/stream.html`. You can of course make changes to this and create a different route for this stream, but this is the minimum requirements.

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
