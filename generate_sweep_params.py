from trainer.utils.json_stuff import save_as_json
import itertools
import copy
import os
import random
random.seed(0)

GPU_IDS = [1,2,3]
wandb_log = True

def divide_list(lst, n):
    """
    Divide a list into N equal parts.

    Parameters:
    lst (list): The list to be divided.
    n (int): The number of parts to divide the list into.

    Returns:
    list of lists: A list containing N sublists, each of which is a part of the original list.
    """
    if n <= 0:
        raise ValueError("Number of parts must be greater than 0.")
    if n > len(lst):
        raise ValueError("Number of parts cannot be greater than the length of the list.")

    # Calculate the size of each part
    k, m = divmod(len(lst), n)
    
    # Create the divided parts
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def generate_sh_file(commands, filename="script.sh"):
    """
    Generates a .sh file with each command from the list written on a new line.

    :param commands: List of commands to be written to the .sh file.
    :param filename: Name of the .sh file to be created. Default is 'script.sh'.
    """
    with open(filename, 'w') as file:
        for command in commands:
            file.write(command + '\n')
    print(f"Saved: {filename}")

run_commands_dir = f"./sd3_sweep_commands"

os.system(
    f"rm -rf {run_commands_dir} && mkdir -p {run_commands_dir}"
)


config_folder = "./sd3_face_sweep_configs"
os.system(f"rm -rf {config_folder}")
os.system(f"mkdir -p {config_folder}")
sweep_params = {
    "unet_learning_rate": [
        5e-5,
        1e-4,
        3e-4,
        7e-4,
        1e-3,
        2e-3,
    ],
    "train_batch_size": [
        2,
        4,
        8,
        16
    ],
    "lora_rank": [
        2, 
        4, 
        6, 
        8,
    ],
    "ti_lr": [1e-3, None],
    "unet_optimizer_type": [
        "adamw",
        "adamw_8bit",
        "prodigy"
    ],
}

num_total_runs = 1
for key in sweep_params:
    num_total_runs *= len(sweep_params[key])

print(f"Num total runs: {num_total_runs}")

default_config = {
    "output_dir": "lora_models/sweep",
    "sd_model_version": "sd3",
    "lora_training_urls": "https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_big.zip",
    "concept_mode": "face",
    "seed": 0,
    "resolution": 512,
    "train_batch_size": 2,
    "n_sample_imgs": 6,
    "max_train_steps": 1000,
    "token_warmup_steps": 200,
    "checkpointing_steps": 1000, ## no need to save any checkpoints
    "gradient_accumulation_steps": 2,
    "sample_imgs_lora_scale": 0.8,
    "n_tokens": 2,
    "ti_lr": 0.001,
    "remove_ti_token_from_prompts": False,
    "text_encoder_lora_optimizer": None,
    "text_encoder_lora_lr": 0.5e-4,
    "text_encoder_lora_weight_decay": 1e-5,
    "text_encoder_lora_rank": 16,
    "lora_alpha_multiplier": 1.0,
    "lora_rank": 16,
    "use_dora": False,
    "caption_model": "blip",
    "debug": True,
}

keys, values = zip(*sweep_params.items())
combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

all_config_paths = []
for index, c in enumerate(combinations):
    config = copy.deepcopy(default_config)
    filename = f"{index}"

    # override default values with sweep params
    for key in c:
        """
        instead of editing the train batch size, we simply change the gradient
        accumulation value. Which has the same effect.
        We will also
        """
        if key == "train_batch_size":
            config["gradient_accumulation_steps"] = c[key] / config["train_batch_size"]
            config["max_train_steps"] = config["max_train_steps"] * config["gradient_accumulation_steps"]
            config["checkpointing_steps"] = config["checkpointing_steps"] * config["gradient_accumulation_steps"]
        else:
            config[key] = c[key]
        # print(f"{index} - Setting {key} to {c[key]}")
        filename += f"_{key}_{c[key]}"
    config_path = os.path.join(
            config_folder,
            f"{filename}.json"
        )
    save_as_json(
        dictionary_or_list=config,
        filename = config_path
    )
    all_config_paths.append(config_path)
    print(f"Saved: {config_path}")

print(f"Total: {index+1} configs")

all_commands = []

for c in all_config_paths:
    command = f"python3 main_sd3.py {c}"
    if wandb_log:
        command = command + " --wandb-log"
    all_commands.append(command)

random.shuffle(all_commands)

all_commands_split_by_gpu = divide_list(
    lst = all_commands,
    n = len(GPU_IDS)
)

for index, gpu_id in enumerate(GPU_IDS):
    commands_on_single_gpu = [
        f"CUDA_VISIBLE_DEVICES={gpu_id} {x}" for x in all_commands_split_by_gpu[index]
    ]
    generate_sh_file(
            commands =  commands_on_single_gpu,
            filename = os.path.join(
                run_commands_dir, 
                f"run_on_gpu_{gpu_id}.sh"
            )
        ) 