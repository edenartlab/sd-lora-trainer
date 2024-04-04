
"""
Faces:
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip

Objects:
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/banny_all.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/banny_best.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/koji_color.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/plantoid_imgs.zip

Styles:
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/does.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip

"""
import random, os, ast, json, shutil
from itertools import product
import time

random.seed(int(1000*time.time()))

def hamming_distance(dict1, dict2):
    distance = 0
    for key in dict1.keys():
        if dict1[key] != dict2.get(key, None):
            distance += 1
    return distance

#######################################################################################

# Setup the base experiment config:
run_name             = "n_tokens"
caption_prefix       = ""
mask_target_prompts  = ""
n_exp                = 300  # how many random experiment settings to generate
min_hamming_distance = 1   # min_n_params that have to be different from any previous experiment to be scheduled
output_dir           = "gridsearch_configs"

# Define training hyperparameters and their possible values
# The params are sampled stochastically, so if you want to use a specific value more often, just put it in multiple times

hyperparameters = {
    "output_dir": ["lora_models/banny_ti"],
    "sd_model_version": ["sd15", "sdxl"],
    "lora_training_urls": [
        "/home/xander/Downloads/datasets/banny_best",
        "/home/xander/Downloads/datasets/plantoid"
    ],
    "concept_mode": ['object'],
    "mask_target_prompts": ["bananaman"],
    "seed": [5],
    "resolution": [512],
    "validation_img_size": [[768, 768]],
    "train_batch_size": [4],
    "n_sample_imgs": [4],
    "max_train_steps": [800],
    "checkpointing_steps": [200],
    "gradient_accumulation_steps": [1],
    "clip_grad_norm": [-1.0],
    "prodigy_d_coef": [0.0],
    "ti_lr": [0.001],
    "ti_weight_decay": [0.0005],
    "lora_weight_decay": [0.0001],
    "l1_penalty": [0.1],
    "off_ratio_power": [0.02],
    "snr_gamma": [5.0],
    "lora_rank": [12],
    "use_dora": ['true'],
    "aspect_ratio_bucketing": ['false'],
    "caption_model": ["blip"],
    "augment_imgs_up_to_n": [20],
    "verbose": ['true'],
    "debug": ['true'],
    "hard_pivot": ['false'],
    "weight_type": ["bf16"],
    "unet_learning_rate": [1.0],
    "dataloader_num_workers": [0],
    "left_right_flip_augmentation": ['true'],
    "name": ["unnamed"],
}

#######################################################################################

# Create a set to hold the combinations that have already been run
scheduled_experiments = set()

# if output_dir exists, remove it:
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)

# Open the shell script file
try_sampling_n_times = 500
for exp_index in range(n_exp):  # number of combinations you want to generate
    resamples, combination = 0, None

    while resamples < try_sampling_n_times:
        experiment_settings = {name: random.choice(values) for name, values in hyperparameters.items()}
        resamples += 1

        min_distance = float('inf')
        for str_experiment_settings in scheduled_experiments:
            existing_experiment_settings = dict(sorted(ast.literal_eval(str_experiment_settings)))
            distance = hamming_distance(experiment_settings, existing_experiment_settings)
            min_distance = min(min_distance, distance)

        if min_distance >= min_hamming_distance:
            str_experiment_settings = str(sorted(experiment_settings.items()))
            scheduled_experiments.add(str_experiment_settings)
            # Save the experiment to a JSON file
            config_filename = os.path.join(output_dir, f"config_{exp_index:03d}.json")
            with open(config_filename, "w") as f:
                json.dump(experiment_settings, f, indent=4)
            break

    if resamples >= try_sampling_n_times:
        print(f"\nCould not find a new experiment_setting after random sampling {try_sampling_n_times} times, dumping all experiment_settings to .json files")
        break

print(f"\n\n---> Saved {len(scheduled_experiments)} experiment configurations to {output_dir}")