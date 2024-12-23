#!/bin/bash

# Update package list
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libsndfile1 \
    sox \
    ffmpeg \
    libportaudio2

# Create directory structure
mkdir -p ~/audio-processor/{uploads,output}
chmod 755 ~/audio-processor/uploads
chmod 755 ~/audio-processor/output

# Install Python dependencies
pip3 install -r requirements.txt

# Set up swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Optional: Configure system for better performance
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
echo "vm.vfs_cache_pressure=50" | sudo tee -a /etc/sysctl.conf

# Apply sysctl changes
sudo sysctl -p

echo "Setup complete! 🚀"