#!/bin/bash

source .env

if [ -z $OBJECT_DETECTION ]; then echo "The environment variable OBJECT_DETECTION is required. This is a boolean value True/False."; fi

if [ -z $MOTION_DETECTION ]; then echo "The environment variable MOTION_DETECTION is required. This is a boolean value True/False."; fi

if [ "${OBJECT_DETECTION}" == "True" ] && [ "${MOTION_DETECTION}" == "True" ]; then echo "Both OBJECT_DETECTION and MOTION_DETECTION cannot be True, choose only one"; fi

if [ "${OBJECT_DETECTION}" == "True" ]; then
    docker compose up -d triton

    ATTEMPT=1
    RETRIES=60
    INTERVAL=1
    TOTAL_TIME=$((RETRIES * INTERVAL))

    echo "Waiting to start Holly Stream until Triton is healthy."
    while [ $ATTEMPT -le $RETRIES ]; do
        url="http://localhost:8000/v2/health/ready"
        response=$(curl --write-out "%{http_code}" --silent --output /dev/null "$url")

        if [ $response -eq 200 ]; then
            echo "Triton STATUS: HEALTHY"
            break
        else
            ATTEMPT=$((ATTEMPT + 1))
            sleep $INTERVAL
        fi
    done

    if [ $ATTEMPT -gt $RETRIES ]; then
        echo "Triton failed all health checks after $TOTAL_TIME seconds. Stopping all services."
        exit 120
    fi

    nohup $PWD/.stream_env/bin/python3 app/main.py > .log.out &
    echo $! > .process.pid
    echo "Holly-stream started."

elif [ "${MOTION_DETECTION}" == "True" ]; then
    nohup $PWD/.stream_env/bin/python3 app/motion.py > .log.out &
    echo $! > .process.pid
    echo "Motion detection started."
    
elif [ "${OBJECT_DETECTION}" == "False" ]; then
    # Checking if required environment variables are defined. If not, then defining them.
    if [ -z $CAMERA_FPS ]; then echo "You did not define CAMERA_FPS. It will default to 30." && CAMERA_FPS=30; fi
    if [ -z $CAMERA_WIDTH ]; then echo "You did not define CAMERA_WIDTH. It will default to 1280." && CAMERA_WIDTH=1280; fi
    if [ -z $CAMERA_HEIGHT ]; then echo "You did not define CAMERA_HEIGHT. It will default to 720." && CAMERA_HEIGHT=720; fi
    if [ -z $STREAM_IP ]; then echo "You did not define STREAM_IP. It will default to 127.0.0.1. This is the IP address where the RTMP stream will be sent." && STREAM_IP="127.0.0.1"; fi
    if [ -z $STREAM_PORT ]; then echo "You did not define STREAM_PORT. It will default to 1935." && STREAM_PORT=1935; fi
    if [ -z $STREAM_APPLICATION ]; then echo "You did not define STREAM_APPLICATION. It will default to 'live'. If you are using Nginx as the RTMP server, this has to match the application name in the config." && STREAM_APPLICATION="live"; fi
    if [ -z $STREAM_KEY ]; then echo "You did not define STREAM_KEY. It will default to 'stream'. If you are using HLS and a website to stream, this must match the name of your .m3u8 file." && STREAM_KEY="stream"; fi
    
    MODEL_DIMS="$CAMERA_WIDTH"x"$CAMERA_HEIGHT"

    stream_command=$(cat <<-EOF
        libcamera-vid \
            --nopreview \
            --inline \
            --timeout 0 \
            --framerate $CAMERA_FPS \
            --width $CAMERA_WIDTH \
            --height $CAMERA_HEIGHT \
            --rotation 180 \
            --brightness 0.2 \
            --listen -o - | \
        ffmpeg \
            -i - \
            -nostdin \
            -profile:v high \
            -pix_fmt yuvj420p \
            -level:v 4.1 \
            -preset ultrafast \
            -tune zerolatency \
            -vcodec libx264 \
            -r $CAMERA_FPS \
            -s $MODEL_DIMS \
            -f flv rtmp://$STREAM_IP:$STREAM_PORT/$STREAM_APPLICATION/$STREAM_KEY
EOF
)

    echo "Holly-stream started."
    nohup sh -c "$stream_command" > .log.out &

else
    echo "Invalid input for OBJECT_DETECTION or MOTION_DETECTION. At least one value must be True."
    exit 120
fi
