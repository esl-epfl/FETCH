# hash:sha256:8477892802699c815aacb5992722a8762e72ca23e596ee8420fdd5e034a71ca3
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir \
    einops==0.6.0 \
    scikit-learn==1.2.1 \
    torch==1.12.0 \
    torchvision==0.13.0
