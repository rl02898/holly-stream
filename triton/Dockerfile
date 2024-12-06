ARG TRITON_VERSION=2.45.0
ARG TRITON_CONTAINER_VERSION=24.04

FROM nvcr.io/nvidia/tritonserver:${TRITON_CONTAINER_VERSION}-py3 AS full
FROM nvcr.io/nvidia/tritonserver:${TRITON_CONTAINER_VERSION}-py3-min

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION=${TRITON_VERSION} \
    NVIDIA_TRITON_SERVER_VERSION=${TRITON_CONTAINER_VERSION} \
    PATH=/opt/tritonserver/bin:${PATH} \
    UCX_MEM_EVENTS=no \
    TF_ADJUST_HUE_FUSED=1 \
    TF_ADJUST_SATURATION_FUSED=1 \
    TF_ENABLE_WINOGRAD_NONFUSED=1 \
    TF_AUTOTUNE_THRESHOLD=2 \
    TRITON_SERVER_GPU_ENABLED=1 \
    TRITON_SERVER_USER=triton-server \
    DEBIAN_FRONTEND=noninteractive \
    TCMALLOC_RELEASE_RATE=200 \
    DCGM_VERSION=3.2.6

RUN userdel tensorrt-server > /dev/null 2>&1 || true && \
    if ! id -u $TRITON_SERVER_USER > /dev/null 2>&1 ; then \
        useradd $TRITON_SERVER_USER; fi && \
    [ `id -u $TRITON_SERVER_USER` -eq 1000 ] && \
    [ `id -g $TRITON_SERVER_USER` -eq 1000 ] && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        clang \
        curl \
        dirmngr \
        git \
        gperf \
        libb64-0d \
        libcurl4-openssl-dev \
        libgoogle-perftools-dev \
        libjemalloc-dev \
        libnuma-dev \
        libre2-9 \
        libgl1 \
        software-properties-common \
        wget \
        libgomp1 \
        python3 \
        libarchive-dev \
        python3-pip \
        python3-wheel \
        python3-setuptools \
        libpython3-dev && \
    curl -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    apt install -y /tmp/cuda-keyring.deb && \
    rm /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y datacenter-gpu-manager=1:3.2.6 && \
    ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib && \
    echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf && \
    ldconfig && \
    rm -f ${_CUDA_COMPAT_PATH}/lib && \
    pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision python-dotenv numpy==1.26.4 && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /opt/tritonserver/*

WORKDIR /opt/tritonserver

COPY --chown=1000:1000 --from=full /opt/tritonserver/bin bin/
COPY --chown=1000:1000 --from=full /opt/tritonserver/lib lib/
COPY --chown=1000:1000 --from=full /opt/tritonserver/include include/
COPY --chown=1000:1000 --from=full /opt/tritonserver/backends/python /opt/tritonserver/backends/python/
COPY --chown=1000:1000 --from=full /opt/tritonserver/backends/tensorrt /opt/tritonserver/backends/tensorrt/
COPY --chown=1000:1000 --from=full /opt/tritonserver/repoagents/checksum /opt/tritonserver/repoagents/checksum/
COPY --chown=1000:1000 --from=full /usr/bin/serve /usr/bin/.

RUN chown triton-server:triton-server /opt/tritonserver/backends && \
    chown triton-server:triton-server /opt/tritonserver/repoagents

CMD ["/bin/sh", "-c", "tritonserver --model-repository=${MODEL_REPOSITORY} --allow-metrics=false"]
