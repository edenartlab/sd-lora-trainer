# Trainer

Code for finetuning and training LoRa modules on top of Stable Diffusion.

## Setup

Install all dependencies manually and run:
`python main.py -c training_args.json`

Adjust the arguments inside `training_args.json` accordingly.

--- 

You can also run this through cog as a docker image:
1. Install Replicate 'cog':

```
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

2. Build the image with `sudo cog build`
3. Run a training run with `sudo sh cog_test_train.sh`


## TODO's

Code / Cleanup:
- ~~turn all/most of the args of the main() function in trainer_pti.py and the preprocess() function into a clean args_dict that makes it easy to add and distribute new parameters over the code and save these args to a .json file at the end.~~
- Modularize the logic in train.py as much as possible, trying to minimize dev work that needs to happen when SD3 drops (in progress)
- ~~make a clean train.py entrypoint that can be run as a normal python command (instead of having to use cog)~~
- ~~make it so the textual_inversion optimizer only optimizes the actual trained token embeddings instead of all of them + resetting later~~
- Figure out how to swap out a lora_adapter module onto a base model without reloading the entire model pipe...
- Make sure the saved LoRa's are compatible with ComfyUI / AUTO1111
- properly measure regularization target_values for conditioning and add pooled_prompt_embeds into regularizer for SDXL

Algo:
- add txt-encoder LoRa + full finetuning options
- Test if timesteps = torch.randint() can be improved: look at sdxl training code! (see https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py#L1263, https://arxiv.org/pdf/2206.00364.pdf)
- Add aspect_ratio bucketing into the dataloader so we can train on non-square images (take this from https://github.com/kohya-ss/sd-scripts)
- test if textual inversion training can also happen with prodigy_optimizer
- add CLIP_similarity token warmup (txt = Done, img = TODO) or (aesthetic gradients: https://github.com/vicgalle/stable-diffusion-aesthetic-gradients/tree/main)
- improve data augmentation, eg by adding outpainted, smaller versions of faces / objects
- figure out why the initial onset of learning in the LoRa / Dora causes a temporary drop in img quality
- currently some sub-optimal stuff can happen in preprocess() when there's less than 3 or more than 45 imgs, try to improve this
- make a separate gpt4-v call to query the concept-description using a random sample of the training imgs, assembled into a single img grid

Bigger improvements:
- create good model evaluation script:
    - img/txt clip similarity (prompt following)
    - some kind of img-feature similarity (eg CLIP or FID or ...) between training imgs and generated imgs
- add stronger token regularization (eg CelebBasis spanning basis)
- Add multi-token training
- implement perfusion: https://research.nvidia.com/labs/par/Perfusion/
- implement prompt-aligned: https://prompt-aligned.github.io/
- make compatible with ziplora: https://ziplora.github.io/


Tuning Experiments once code is fully ready:
- gridsearch over LoRa target_modules=["to_k", "to_q", "to_v", "to_out.0"]
- try-out conditioning noise injection during training to increase robustness
- re-test / tweak the adaptive learning rates (also test Prodigy vs Adam)
- right now it looks like the diffusion model gets partially "destroyed" in the beginning of training (outputs from steps 100-200 look terrible), 
but it then recovers. Can we avoid this collapse? Is the learning rate too high?
- offset noise
- AB test Dora vs Lora
- sweep n_trainable_tokens to inject