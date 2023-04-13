FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get update && apt-get install -y git

RUN pip install \
    pytest \
    pylint \
    autopep8
