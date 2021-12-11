#!/bin/bash

path = /home/ec2-user/SageMaker/docker-test/nftrip
docker run --gpus all -v $path:/app/ nftrip-docker neural_style_transfer.py