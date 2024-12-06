#!/bin/bash

yolo export \
	model=yolo11n.pt \
	format=onnx \
	half=True \
	simplify=True \
	opset=20 \
	device=0

docker run --rm -it --gpus '"device=0"' -v ./:/models nvcr.io/nvidia/tensorrt:24.04-py3 \
	trtexec \
		--onnx=/models/yolo11n.onnx \
		--saveEngine=/models/model.plan \
		--fp16 \
		--inputIOFormats=fp16:chw \
		--outputIOFormats=fp16:chw \
		--device=0 \
		--workspace=4096 \
		--useCudaGraph \
		--timingCacheFile=/tmp/timing.cache
