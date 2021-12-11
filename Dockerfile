FROM nvidia/cuda:10.2-base

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install -r /app/requirements.txt

WORKDIR /app

ENTRYPOINT ["python3"]

