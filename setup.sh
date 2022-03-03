#!/bin/sh
apt update -y
apt upgrade -y
apt install libatlas-base-dev -y
python3 -m ensurepip --upgrade
pip3 install -r requirements.txt 
echo Complete.
