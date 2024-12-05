#!/bin/bash

source .env

for var in DOCKER_USERNAME DOCKER_PASSWORD LATEST_VERSION; do
    if [[ -z "${!var}" ]]; then
        echo "Set ${var} in your .env file"
        exit 1
    fi
done


docker login -u ${DOCKER_USERNAME} -p ${DOCKER_PASSWORD}

REPO="rcland12/detection-stream"
VARIANTS=("linux" "linux-triton" "nginx")

process_image() {
    local variant=$1
    local latest_tag="${REPO}:${variant}-latest"
    local version_tag="${REPO}:${variant}-${LATEST_VERSION}"
    
    docker images -q ${latest_tag} | xargs -I{} docker tag {} ${version_tag}
    docker push ${version_tag}
    docker push ${latest_tag}
    docker rmi -f ${version_tag}
}

for variant in "${VARIANTS[@]}"; do
    process_image ${variant}
done
