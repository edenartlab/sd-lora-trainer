
"""
Faces:
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/gene.zip

Objects:
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/banny_all.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/banny_best.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/koji_color.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/plantoid_imgs.zip

Styles:
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/does.zip
https://edenartlab-lfs.s3.amazonaws.com/datasets/clipx.zip

/home/rednax/Documents/datasets/good_styles/eden_crystals
/home/rednax/Documents/datasets/good_styles/does2
/home/rednax/Documents/datasets/good_styles/beeple


"""

import random, os, ast, json, shutil
from itertools import product
import time
from tqdm import tqdm

random.seed(int(1000*time.time()))

def hamming_distance(dict1, dict2):
    distance = 0
    for key in dict1.keys():
        if dict1[key] != dict2.get(key, None):
            distance += 1
    return distance

#######################################################################################

# Setup the base experiment config:
exp_name             = "stitchly"
caption_prefix       = ""
mask_target_prompts  = ""
n_exp                = 100  # how many random experiment settings to generate
min_hamming_distance = 2   # min_n_params that have to be different from any previous experiment to be scheduled
nohup                = False
output_sh_path = f"gridsearch_configs/{exp_name}.sh"

# Define training hyperparameters and their possible values
# The params are sampled stochastically, so if you want to use a specific value more often, just put it in multiple times

hyperparameters = {
    "output_dir": [f"lora_models/{exp_name}"],
    "sd_model_version": ["sdxl"],
    "lora_training_urls": [
        "/home/rednax/Documents/datasets/good_styles/stitchly"

    ],
    "concept_mode": ['style'],
    "sample_imgs_lora_scale": [0.9],
    "disable_ti": ['false','true'],
    "caption_dropout": [0.2],
    "seed": [0],
    "resolution": [512,768],
    "train_batch_size": [4],
    "n_sample_imgs": [8],
    "max_train_steps": [2000],
    "checkpointing_steps": [500],
    "gradient_accumulation_steps": [1],

    "n_tokens": [3],
    "ti_lr": [0.001],
    "token_warmup_steps": [0],

    "unet_lr": [0.0003, 0.001],
    "lora_rank": [16],
    "use_dora": ['false'],

    "unet_optimizer_type": ['adamw'],
    "is_lora": ['true', 'false'],

    "text_encoder_lora_optimizer": [None],
    "text_encoder_lora_lr": [0.0e-4],

    "snr_gamma": [5.0],
    "caption_model": ["florence", "blip"],
    "augment_imgs_up_to_n": [40],
    "verbose": ['true'],
    "debug": ['true']
}

#######################################################################################

# Create a set to hold the combinations that have already been run
scheduled_experiments = set()

# if config_output_dir exists, remove it:
config_output_dir = f"gridsearch_configs/{exp_name}"
shutil.rmtree(config_output_dir, ignore_errors=True)
os.makedirs(config_output_dir, exist_ok=True)

# Open the shell script file
try_sampling_n_times = 120
for exp_index in tqdm(range(n_exp)):  # number of combinations you want to generate
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
            config_filename = f"{config_output_dir}/{exp_name}_{exp_index:03d}.json"
            dirname = os.path.dirname(config_filename)
            os.makedirs(dirname, exist_ok=True)

            with open(config_filename, "w") as f:
                json.dump(experiment_settings, f, indent=4)
            break

    if resamples >= try_sampling_n_times:
        print(f"\nCould not find a new experiment_setting after random sampling {try_sampling_n_times} times, dumping all experiment_settings to .json files")
        break

print(f"\n\n---> Saved {len(scheduled_experiments)} experiment configurations to {config_output_dir}")

def generate_sh_script(folder_path, output_sh_path):
    # Get a list of JSON files in the folder, sorted alphabetically
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
    
    # Open the output .sh file for writing
    with open(output_sh_path, 'w') as sh_file:
        # Write the shebang line for a bash script
        sh_file.write("#!/bin/bash\n\n")
        
        # Write a command for each JSON file
        for json_file in json_files:
            file_path = os.path.join("scripts/", folder_path, json_file)
            command = f"python main.py {file_path}\n"

            if nohup:
                command = f"nohup {command} > {file_path.replace('.json', '.log')} 2>&1 &\n"

            sh_file.write(command)

generate_sh_script(config_output_dir, output_sh_path)
print(f"\n---> Saved the executable shell script to {output_sh_path}")
