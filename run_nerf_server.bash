#!/bin/bash
# apt dependencies
sudo apt-get install ffmpeg libsm6 libxext6 -y

# pip dependencies
pip install -r requirements.py

# Run NeRF
python run_nerf.py --config configs/$1.txt
