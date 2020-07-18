ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

MAINTSTINER hehe "isliliangshan@163.com"

RUN apt-get update && apt-get install -y git libglib2.0-0 libsm6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ADD . ./app
WORKDIR /app
