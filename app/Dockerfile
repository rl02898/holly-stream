FROM debian:bullseye-slim
WORKDIR /root/app

COPY ./requirements.txt ./
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3-pip ffmpeg && \
    python3 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY main.py stream.sh entrypoint.sh ./
COPY ./images/santa*.png ./images/
CMD ["/bin/bash", "entrypoint.sh"]
