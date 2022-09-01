#!/bin/bash
# apt dependencies
sudo apt update
sudo apt-get install ffmpeg libsm6 libxext6 mogrify -y

# pip dependencies
pip install -r requirements.txt

# Run NeRF
python run_nerf.py --config configs/$1.txt
