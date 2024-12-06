#!/bin/bash

source .env

for var in DOCKER_USERNAME DOCKER_PASSWORD LATEST_VERSION; do
    if [[ -z "${!var}" ]]; then
        echo "Set ${var} in your .env file"
        exit 1
    fi
done

docker login -u ${DOCKER_USERNAME} -p ${DOCKER_PASSWORD}
docker images -q rcland12/detection-stream:linux-latest | xargs -I{} docker tag {} rcland12/detection-stream:linux-${LATEST_VERSION}
docker images -q rcland12/detection-stream:linux-triton-latest | xargs -I{} docker tag {} rcland12/detection-stream:linux-triton-${LATEST_VERSION}
docker images -q rcland12/detection-stream:nginx-latest | xargs -I{} docker tag {} rcland12/detection-stream:nginx-${LATEST_VERSION}
docker push rcland12/detection-stream:linux-${LATEST_VERSION}
docker push rcland12/detection-stream:linux-triton-${LATEST_VERSION}
docker push rcland12/detection-stream:nginx-${LATEST_VERSION}
docker rmi -f rcland12/detection-stream:linux-${LATEST_VERSION}
docker rmi -f rcland12/detection-stream:linux-triton-${LATEST_VERSION}
docker rmi -f rcland12/detection-stream:nginx-${LATEST_VERSION}
docker push rcland12/detection-stream:linux-latest
docker push rcland12/detection-stream:linux-triton-latest
docker push rcland12/detection-stream:nginx-latest
