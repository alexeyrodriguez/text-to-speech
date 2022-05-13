#!/bin/bash

# Arguments: git repo url, branch, experiment configuration file, gcs bucket where to copy model

# Get required env
. /etc/profile

set -x

PROJECT=simple-tts
MODEL_DIR=model-path-$$

if [[ ! -d $PROJECT ]]
then
    git clone -b $2 $1 $PROJECT
    cd $PROJECT
else
    cd $PROJECT
    git pull
fi

# put the below stuff into requirements
pip install gin-config

mkdir -p $MODEL_DIR
python training.py --experiment $3 --model-dir $MODEL_DIR

gsutil -m cp -r $MODEL_DIR/* $4

# Clean up model dir
mkdir -p models
mv $MODEL_DIR/* models
rmdir $MODEL_DIR