#!/bin/bash

# Arguments: git repo url, branch, experiment configuration file, gcs bucket where to copy model

# Get required env
. /etc/profile

PROJECT=simple-tts

if [[ ! -d $PROJECT ]]
then
    git clone -b $2 $1 $PROJECT
fi

cd $PROJECT

# put the below stuff into requirements
pip install gin-config

python training.py --experiment $3

# TODO copy model to gcs

