#!/bin/bash

source .env

[[ -z $OBJECT_DETECTION ]] && echo "The environment variable OBJECT_DETECTION is required. This is a boolean value True/False." && exit 1

if [[ ! "${OBJECT_DETECTION}" =~ ^(True|False)$ ]]; then
    echo "Invalid input for OBJECT_DETECTION. Expecting True or False; received ${OBJECT_DETECTION}."
    exit 120
fi

SERVICES="app"
[[ "${OBJECT_DETECTION}" == "True" ]] && SERVICES="triton ${SERVICES}"

docker compose down ${SERVICES}
