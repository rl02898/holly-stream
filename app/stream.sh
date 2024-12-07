#!/bin/bash

if [ -z $CAMERA_AUDIO ]; then echo "The environment variable CAMERA_AUDIO is required. This is a boolean value True/False."; fi
if [ -z $CAMERA_WIDTH ]; then echo "You did not define CAMERA_WIDTH. It will default to 1280." && CAMERA_WIDTH=1280; fi
if [ -z $CAMERA_HEIGHT ]; then echo "You did not define CAMERA_HEIGHT. It will default to 720." && CAMERA_HEIGHT=720; fi
if [ -z $CAMERA_INDEX ]; then echo "You did not define CAMERA_INDEX. This value is usually 0 unless you have other devices connected. It will default to 0." && CAMERA_INDEX=0; fi
if [ -z $CAMERA_FPS ]; then echo "You did not define CAMERA_FPS. It will default to 30." && CAMERA_FPS=30; fi
if [ -z $STREAM_IP ]; then echo "You did not define STREAM_IP. It will default to 127.0.0.1. This is the IP address where the RTMP stream will be sent." && STREAM_IP="127.0.0.1"; fi
if [ -z $STREAM_PORT ]; then echo "You did not define STREAM_PORT. It will default to 1935." && STREAM_PORT=1935; fi
if [ -z $STREAM_APPLICATION ]; then echo "You did not define STREAM_APPLICATION. It will default to 'live'. If you are using Nginx as the RTMP server, this has to match the application name in the config." && STREAM_APPLICATION="live"; fi
if [ -z $STREAM_KEY ]; then echo "You did not define STREAM_KEY. It will default to 'stream'. If you are using HLS and a website to stream, this must match the name of your .m3u8 file." && STREAM_KEY="stream"; fi

DIMS="$CAMERA_WIDTH"x"$CAMERA_HEIGHT"

if [ "${CAMERA_AUDIO}" == "True" ]; then
    ffmpeg \
        -f v4l2 \
        -video_size $DIMS \
        -r $CAMERA_FPS \
        -i /dev/video$CAMERA_INDEX \
        -f alsa \
        -thread_queue_size 4096 \
        -ar 22050 \
        -ac 1 \
        -i default \
        -c:v libx264 \
        -preset fast \
        -pix_fmt yuv420p \
        -b:v 1500k \
        -maxrate 1500k \
        -bufsize 3000k \
        -g 60 \
        -c:a aac \
        -b:a 64k \
        -async 1 \
        -vsync 1 \
        -max_muxing_queue_size 1024 \
        -f flv rtmp://$STREAM_IP:$STREAM_PORT/$STREAM_APPLICATION/$STREAM_KEY
else
    ffmpeg \
        -f v4l2 \
        -video_size $DIMS \
        -r $CAMERA_FPS \
        -i /dev/video$CAMERA_INDEX \
        -c:v libx264 \
        -preset fast \
        -pix_fmt yuv420p \
        -b:v 1500k \
        -maxrate 1500k \
        -bufsize 3000k \
        -g 60 \
        -f flv rtmp://$STREAM_IP:$STREAM_PORT/$STREAM_APPLICATION/$STREAM_KEY
fi
