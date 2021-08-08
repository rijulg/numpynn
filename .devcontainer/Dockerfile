FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN apt-get update && apt-get install -y git

RUN pip install \
    pytest \
    pylint \
    autopep8