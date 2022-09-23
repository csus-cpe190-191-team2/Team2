#!/bin/sh
apt update -y
apt upgrade -y
apt install libatlas-base-dev -y
apt install ffmpeg libsm6 libxext6  -y
apt install python3-pip -y
pip3 install --upgrade setuptools pip
pip3 install -r requirements.txt 
echo Complete.
