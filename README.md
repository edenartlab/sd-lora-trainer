# Trainer

Code for finetuning and training LoRa modules on top of Stable Diffusion.

## Setup

1. Install Replicate 'cog':

```
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

2. Build the image with `sudo cog build`
3. Run a training run with `sudo sh test_train.sh`


## TODO's

Code / Cleanup:
- ~~turn all/most of the args of the main() function in trainer_pti.py and the preprocess() function into a clean args_dict that makes it easy to add and distribute new parameters over the code and save these args to a .json file at the end.~~
- Modularize the logic in train.py as much as possible, trying to minimize dev work that needs to happen when SD3 drops (in progress)
- ~~make a clean train.py entrypoint that can be run as a normal python command (instead of having to use cog)~~
- ~~make it so the textual_inversion optimizer only optimizes the actual trained token embeddings instead of all of them + resetting later~~
- test if the trained concepts with peft are compatible with ComfyUI / AUTO1111

Algo:
- Add aspect_ratio bucketing into the dataloader so we can train on non-square images (take this from https://github.com/kohya-ss/sd-scripts)
- test if textual inversion training can also happen with prodigy_optimizer
- the random initialization of the token embeddings has a relatively large impact on the final outcome, there are prob ways to reduce
this random variance, eg CLIP_similarity pretraining.
- Improve the img captioning by swapping BLIP for cogVLM: https://github.com/THUDM/CogVLM

Bugfixing:
see msgs at: https://discord.com/channels/573691888050241543/1184175211998883950/1217550596878373037
- check how the pipe() objects work under the hood in HF diffusers library, is there a difference w how the unet is called in the training loop / which args it gets?
- Try to find out why the diffusers training script works for sd15 and ours doesnt:
See here: https://huggingface.co/blog/sdxl_lora_advanced_script
and here: https://github.com/huggingface/diffusers/tree/main/examples/advanced_diffusion_training
- figure out how to adaptively set lora_scale at inference time using peft + diffusers? (https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py#L240)


Bigger improvements:
- Add multi-token training
- implement perfusion: https://research.nvidia.com/labs/par/Perfusion/
- implement prompt-aligned: https://prompt-aligned.github.io/
- make compatible with ziplora: https://ziplora.github.io/



Tuning Experiments once code is fully ready:

- try-out conditioning noise injection during training to increase robustness
- re-test / tweak the adaptive learning rates instead of hard-pivot (also test Prodigy vs Adam)
- right now it looks like the diffusion model gets partially "destroyed" in the beginning of training (outputs from steps 100-200 look terrible), 
but it then recovers. Can we avoid this collapse? Is the learning rate too high?
- gradient_accumulation
- offset noise
- AB test Dora vs Lora
- sweep n_trainable_tokens to inject