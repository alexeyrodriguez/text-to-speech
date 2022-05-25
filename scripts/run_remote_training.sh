#!/bin/bash

# Arguments:
#  - git repo url
#  - branch
#  - gcs bucket where to copy model
#  - additional trainer args

GIT_REPO=$1
BRANCH=$2
GCS_PATH=$3
shift 3

# Get required env
. /etc/profile

set -x
set -e

PROJECT=simple-tts
MODEL_DIR=model-path-$$

if [[ ! -d $PROJECT ]]
then
    git clone -b $BRANCH $GIT_REPO $PROJECT
    cd $PROJECT
else
    cd $PROJECT
    git pull
fi

# put the below stuff into requirements
pip install gin-config wandb tensorflow-addons

mkdir -p $MODEL_DIR
python training.py --model-dir $MODEL_DIR "$@"

gsutil -m cp -r $MODEL_DIR/* "$GCS_PATH"

# Clean up model dir
mkdir -p models
mv $MODEL_DIR/* models
rmdir $MODEL_DIR