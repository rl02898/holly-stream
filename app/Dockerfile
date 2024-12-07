FROM debian:bullseye-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /root/app

RUN apt-get update && \
    apt-get install -y \
        python3.10 \
        python3-pip \
        ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        python-dotenv \
        opencv-python \
        imutils \
        tritonclient[all]

COPY main.py stream.sh entrypoint.sh ./
COPY ./images/ ./images/

CMD ["/bin/bash", "entrypoint.sh"]
