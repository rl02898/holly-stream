#!/bin/bash

source .env

[[ -z $OBJECT_DETECTION ]] && echo "The environment variable OBJECT_DETECTION is required. This is a boolean value True/False." && exit 1

if [[ ! "${OBJECT_DETECTION}" =~ ^(True|False)$ ]]; then
    echo "Invalid input for OBJECT_DETECTION. Expecting True or False; received ${OBJECT_DETECTION}."
    exit 120
fi

if [[ "${OBJECT_DETECTION}" == "True" ]]; then
    docker compose up -d triton
    
    echo "Waiting to start Holly Stream until Triton is healthy."
    for ((attempt=1; attempt<=60; attempt++)); do
        if curl -s -f "http://localhost:8000/v2/health/ready" > /dev/null; then
            break
        fi
        sleep 1
        [[ $attempt -eq 60 ]] && echo "Triton failed all health checks after 60 seconds. Stopping all services." && exit 120
    done
fi

docker compose up -d app
echo "Holly Stream has started. Performing health check..."
sleep 10

if docker container inspect -f '{{.State.Running}}' app &>/dev/null; then
    echo "Holly Stream STATUS: HEALTHY"
else
    echo "Holly Stream STATUS: UNHEALTHY"
    echo "Shutting down."
    [[ "${OBJECT_DETECTION}" == "True" ]] && docker compose down triton
    exit 1
fi
