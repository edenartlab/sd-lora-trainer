# Trainer

Code for finetuning and training LoRa modules on top of Stable Diffusion.

## Setup

1. Install Replicate 'cog':
``
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
``
2. Build the image with `cog build`
3. Run a training run with `sh test_train.sh`
