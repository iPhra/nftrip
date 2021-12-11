#!/bin/bash

SCRIPT=${1:-"neural_style_transfer.py"}
VOLUME="/home/ec2-user/SageMaker/docker-test/nftrip"
docker run --gpus all -v $VOLUME:/app/ nftrip-docker $SCRIPT