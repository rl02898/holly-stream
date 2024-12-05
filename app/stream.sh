#!/bin/bash

# Default configuration values
declare -A defaults=(
    [CAMERA_AUDIO]=""  # Required, no default
    [CAMERA_WIDTH]="1280"
    [CAMERA_HEIGHT]="720"
    [CAMERA_INDEX]="0"
    [CAMERA_FPS]="30"
    [STREAM_IP]="127.0.0.1"
    [STREAM_PORT]="1935"
    [STREAM_APPLICATION]="live"
    [STREAM_KEY]="stream"
)

check_env_var() {
    local var_name=$1
    local description=$2
    
    if [[ -z "${!var_name}" ]]; then
        if [[ -z "${defaults[$var_name]}" ]]; then
            echo "The environment variable $var_name is required. $description"
            exit 1
        else
            echo "You did not define $var_name. $description It will default to ${defaults[$var_name]}."
            eval "$var_name=${defaults[$var_name]}"
        fi
    fi
}

check_env_var "CAMERA_AUDIO" "This is a boolean value True/False."
check_env_var "CAMERA_WIDTH" "This sets the video width."
check_env_var "CAMERA_HEIGHT" "This sets the video height."
check_env_var "CAMERA_INDEX" "This value is usually 0 unless you have other devices connected."
check_env_var "CAMERA_FPS" "This sets the video framerate."
check_env_var "STREAM_IP" "This is the IP address where the RTMP stream will be sent."
check_env_var "STREAM_PORT" "This sets the RTMP port."
check_env_var "STREAM_APPLICATION" "If you are using Nginx as the RTMP server, this has to match the application name in the config."
check_env_var "STREAM_KEY" "If you are using HLS and a website to stream, this must match the name of your .m3u8 file."

# Validate CAMERA_AUDIO value
if [[ ! "${CAMERA_AUDIO}" =~ ^(True|False)$ ]]; then
    echo "Invalid input for CAMERA_AUDIO. Expecting True or False; received ${CAMERA_AUDIO}."
    exit 1
fi

# video only
if [ "${CAMERA_AUDIO}" == "True" ]; then
    ffmpeg \
        -f v4l2 \
        -i /dev/video$CAMERA_INDEX \
        -video_size $DIMS \
        -r $CAMERA_FPS \
        -f alsa \
        -i default \
        -c:v libx264 \
        -preset fast \
        -pix_fmt yuv420p \
        -b:v 1500k \
        -maxrate 1500k \
        -bufsize 3000k \
        -g 60 \
        -c:a aac \
        -b:a 128k \
        -ac 2 \
        -f flv rtmp://$STREAM_IP:$STREAM_PORT/$STREAM_APPLICATION/$STREAM_KEY

# audio/video
else
    ffmpeg \
        -f v4l2 \
        -input_format mjpeg \
        -i /dev/video$CAMERA_INDEX \
        -video_size $DIMS \
        -r $CAMERA_FPS \
        -c:v libx264 \
        -preset fast \
        -pix_fmt yuv420p \
        -b:v 1500k \
        -maxrate 1500k \
        -bufsize 3000k \
        -g 60 \
        -f flv rtmp://$STREAM_IP:$STREAM_PORT/$STREAM_APPLICATION/$STREAM_KEY

fi
