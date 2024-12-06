#!/bin/bash

[[ -z $OBJECT_DETECTION ]] && echo "The environment variable OBJECT_DETECTION is required. This is a boolean value True/False." && exit 1

if [[ ! "${OBJECT_DETECTION}" =~ ^(True|False)$ ]]; then
    echo "Invalid input for OBJECT_DETECTION. Expecting True or False; received ${OBJECT_DETECTION}."
    exit 120
fi

if [[ "${OBJECT_DETECTION}" == "True" ]]; then
    exec python3 main.py
else
    exec /bin/bash stream.sh
fi
