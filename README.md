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

## Evaluation

Download the aesthetic predictor model checkpoint first from google drive. This should give you a file named: `aesthetic_score_best_model.pth` (99.2 MB)

```bash
gdown 1thEIlXVc8lkULVUBY9Ab45tsOERxkjxns
```

Once the model is downloaded, you can run the eval script with the following CLI args:

- `output_folder`: this is where the outputs of the model get saved as jpeg files
- `lora_path`: path to your LoRA checkpoint (make sure you edit `path_to_your_model_checkpoints` to point to the correct folder. It generally ends with something like `checkpoint-600` where `600` was the training step)
- `output_json`: save all scores in this json file
- `config_filename`: config file used for training

```bash
python3 evaluate.py \
--output_folder eval_images \
--lora_path path_to_your_model_checkpoint  \
--output_json eval_results.json \
--config_filename training_args.json
```


## TODO's

Code / Cleanup:
- cleanup optimizers / optimizeable params code into optimizer.py
- Make sure the saved LoRa's are compatible with ComfyUI / AUTO1111
- Figure out how to swap out a lora_adapter module onto a base model without reloading the entire model pipe...

Algo:
- Improve some of the chatgpt functionality:
    - separate the "gpt_description" / "gpt_segmentation" prompt calls and make them run on a subset of prompts in case there's a lot of imgs / prompts (possibly use img_grids for some gpt4-v calls)
    - currently some sub-optimal stuff can happen in preprocess() when there's less than 3 or more than 45 imgs, try to improve this
- Test if timesteps = torch.randint() can be improved: look at sdxl training code! (see https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py#L1263, https://arxiv.org/pdf/2206.00364.pdf)
- Add aspect_ratio bucketing into the dataloader so we can train on non-square images (take this from https://github.com/kohya-ss/sd-scripts)
- test if textual inversion training can also happen with prodigy_optimizer
- add CLIP_similarity token warmup (txt = Done, img = TODO) or (aesthetic gradients: https://github.com/vicgalle/stable-diffusion-aesthetic-gradients/tree/main)
- improve data augmentation, eg by adding outpainted, smaller versions of faces / objects
- figure out why the initial onset of learning in the LoRa / Dora causes a temporary drop in img quality

Bigger improvements:
- add stronger token regularization (eg CelebBasis spanning basis):
    - remove the fix_embedding_std() hack with gradient based std-matching penalty
    - grid-search the new CovarianceLoss() strength
- Add multi-token training
- implement perfusion ideas (key locking with superclass): https://research.nvidia.com/labs/par/Perfusion/
- implement prompt-aligned: https://prompt-aligned.github.io/

Tuning Experiments once code is fully ready:
- gridsearch over LoRa target_modules=["to_k", "to_q", "to_v", "to_out.0"] for both unet and txt-encoder
- try-out conditioning noise injection during training to increase robustness
- re-test / tweak the adaptive learning rates (also test Prodigy vs Adam)
- right now it looks like the diffusion model gets partially "destroyed" in the beginning of training (outputs from steps 100-200 look terrible), 
but it then recovers. Can we avoid this collapse? Is the learning rate too high?
- offset noise
- AB test Dora vs Lora
- sweep n_trainable_tokens to inject