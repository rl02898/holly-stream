#!/bin/bash

source .env

if [ -z $OBJECT_DETECTION ]; then echo "The environment variable OBJECT_DETECTION is required. This is a boolean value True/False."; fi

if [ "${OBJECT_DETECTION}" == "True" ]; then
    echo "Stopping holly-stream..."
    pid=$(cat .process.pid)
    kill "$pid"
    pkill raspberrypi_ipa
    rm .process.pid .log.out
    echo "Complete"

    echo "Stopping Triton..."
    docker compose down
    echo "Complete"


elif [ "${OBJECT_DETECTION}" == "False" ]; then
    echo "Stopping holly-stream..."
    pkill ffmpeg
    pkill raspberrypi_ipa
    rm .log.out
    echo "Complete"

else
    echo "Invalid input for OBJECT_DETECTION. Expecting True or False; received ${OBJECT_DETECTION}."
    exit 120
fi
