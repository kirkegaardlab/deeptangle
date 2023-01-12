FROM python:3.8-slim-buster

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install wget unzip ffmpeg -y

COPY requirements.txt .

RUN ["pip3", "install", "--upgrade", "-r", "requirements.txt"]
RUN ["pip3", "install", "--upgrade", "https://github.com/alonfnt/dm-haiku/archive/refs/heads/avg_pool_perf.zip"]


WORKDIR /deeptangle 
RUN ["wget", "https://sid.erda.dk/share_redirect/cEjIpG1yQl", "-O", "weights.zip"]
RUN ["unzip", "weights.zip"]

COPY . /deeptangle

RUN ["pip3", "install", "-e", "."]

CMD ["python3", "examples/detect.py", "--model=weights/", "--input=/mnt/celegans_512.avi", "--output=/mnt/output.png", "--frame=100", "--correction_factor=1.3"]

