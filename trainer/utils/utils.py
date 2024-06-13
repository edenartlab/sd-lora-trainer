import os
from typing import Dict, List, Optional, Tuple

import random
import numpy as np
import pandas as pd
import gc
import PIL
import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PretrainedConfig
import torch.nn.functional as F
import matplotlib.pyplot as plt

dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32
}

import re
def replace_in_string(s, replacements):
    while True:
        replaced = False
        for target, replacement in replacements.items():
            new_s = re.sub(target, replacement, s, flags=re.IGNORECASE)
            if new_s != s:
                s = new_s
                replaced = True
        if not replaced:
            break
    return s

def fix_prompt(prompt: str):
    if not prompt:
        return prompt
    # Remove extra commas and spaces, and fix space before punctuation
    prompt = re.sub(r"\s+", " ", prompt)  # Replace multiple spaces with a single space
    prompt = re.sub(r",,", ",", prompt)  # Replace double commas with a single comma
    prompt = re.sub(r"\s?,\s?", ", ", prompt)  # Fix spaces around commas
    prompt = re.sub(r"\s?\.\s?", ". ", prompt)  # Fix spaces around periods
    return prompt.strip()  # Remove leading and trailing whitespace

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def zipdir(path, ziph, extension = '.py'):
    # Zip the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file), 
                                           os.path.join(path, '..')))

def pick_best_gpu_id():
    try:
        # pick the GPU with the most free memory:
        gpu_ids = [i for i in range(torch.cuda.device_count())]
        print(f"# of visible GPUs: {len(gpu_ids)}")
        gpu_mem = []
        for gpu_id in gpu_ids:
            free_memory, tot_mem = torch.cuda.mem_get_info(device=gpu_id)
            gpu_mem.append(free_memory)
            print("GPU %d: %d MB free" %(gpu_id, free_memory / 1024 / 1024))
        
        if len(gpu_ids) == 0:
            # no GPUs available, use CPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return None

        best_gpu_id = gpu_ids[np.argmax(gpu_mem)]
        # set this to be the active GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
        print("Using GPU %d" %best_gpu_id)
        return best_gpu_id
    except Exception as e:
        print(f'Error picking best gpu: {e}')
        print(f'Falling back to GPU 0')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return 0
    

import psutil
def print_system_info():
    try:
        # Print GPU memory information
        gpu_ids = [i for i in range(torch.cuda.device_count())]
        for gpu_id in gpu_ids:
            free_memory, total_memory = torch.cuda.mem_get_info(device=gpu_id)
            print(f"GPU {gpu_id}: {free_memory // 1024 // 1024} of {total_memory // 1024 // 1024} Mb free")

        # Print disk space information
        disk_usage = psutil.disk_usage('/')
        total_disk = disk_usage.total // (1024 * 1024)
        used_disk  = disk_usage.used // (1024 * 1024)
        percent_disk_used = disk_usage.percent
        print(f"Used disk space: {used_disk}/{total_disk} MB = {percent_disk_used}% used")

        # Print RAM information
        virtual_mem = psutil.virtual_memory()
        total_ram   = virtual_mem.total // (1024 * 1024)
        current_ram = virtual_mem.used // (1024 * 1024)
        percent_ram_used = virtual_mem.percent
        print(f"Current used RAM: {current_ram}/{total_ram} MB = {percent_ram_used}% used")
    
    except Exception as e:
        print(f'Error in gathering system info: {str(e)}')

    return


def plot_torch_hist(parameters, step, checkpoint_dir, name, bins=100, min_val=-1, max_val=1, ymax_f = 0.75, color = 'blue'):
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Flatten and concatenate all parameters into a single tensor
        all_params = torch.cat([p.data.view(-1) for p in parameters])
        norm = torch.norm(all_params)

        # Convert to CPU for plotting
        all_params_cpu = all_params.cpu().float().numpy()

        # Plot histogram
        plt.figure()
        plt.hist(all_params_cpu, bins=bins, density=False, color = color)
        plt.ylim(0, ymax_f * len(all_params_cpu.flatten()))
        plt.xlim(min_val, max_val)
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        plt.title(f'{name} (std: {np.std(all_params_cpu):.5f}, norm: {norm:.3f}, step {step:03d})')
        plt.savefig(f"{checkpoint_dir}/{name}_hist_{step:04d}.png")
        plt.close()
    except:
        print(f'Error plotting {name} histogram')

def plot_curve(value_dict, xlabel, ylabel, title, save_path, log_scale = False, y_lims = None):
    plt.figure()
    for key in value_dict.keys():
        values = value_dict[key]
        plt.plot(range(len(values)), values, label=key)

    if log_scale:
        plt.yscale('log')  # Set y-axis to log scale
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_lims is not None:
        plt.ylim(y_lims[0], y_lims[1])
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# plot the learning rates:
def plot_lrs(learning_rate_dict, save_path='learning_rates.png'):
    plt.figure()
    for key in learning_rate_dict.keys():
        lrs = learning_rate_dict[key]
        if len(lrs) == 0:
            continue
        plt.plot(range(len(lrs)), lrs, label=key)
    plt.yscale('log')  # Set y-axis to log scale
    plt.ylim(1e-6, 3e-3)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Curves')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# plot the learning rates:
def plot_grad_norms(grad_norms, save_path='grad_norms.png'):
    plt.figure()
    plt.plot(range(len(grad_norms['unet'])), grad_norms['unet'], label='unet')

    for i in range(2):
        try:
            plt.plot(range(len(grad_norms[f'text_encoder_{i}'])), grad_norms[f'text_encoder_{i}'], label=f'text_encoder_{i}')
        except:
            pass

    plt.yscale('log')  # Set y-axis to log scale
    plt.ylim(1e-6, 100.0)
    plt.xlabel('Step')
    plt.ylabel('Grad Norm')
    plt.title('Gradient Norms')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_token_stds(token_std_dict, save_path='token_stds.png', target_value_dict = {}):
    plt.figure()
    anchor_values = []
    for key in token_std_dict.keys():
        tokenizer_i_token_stds = token_std_dict[key]
        for i in range(len(tokenizer_i_token_stds)):
            stds = tokenizer_i_token_stds[i]
            if len(stds) == 0:
                continue
            anchor_values.append(stds[0])
            encoder_index = int(key.split('_')[-1])
            plt.plot(range(len(stds)), stds, label=f'{key}_tok_{i}', linestyle='dashed' if encoder_index > 0 else 'solid')

    plt.xlabel('Step')
    plt.ylabel('Token Embedding Std')
    centre_value = 0.013
    up_f, down_f = 1.4, 1.3
    try:
        plt.ylim(centre_value/down_f, centre_value*up_f)
    except:
        pass
    
    # Plotting target values as horizontal lines
    for label, value in target_value_dict.items():
        plt.axhline(y=value, color='r', linestyle='-' if '0' in label else '--', label=label)
        plt.text(0, value, label, ha='left', va='center')

    plt.title('Token Embedding Std')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

from scipy.signal import savgol_filter
def plot_loss(loss_dict, save_path='losses.png', window_length=31, polyorder=3, default_color='gray'):
    colormap = {'img_loss': 'blue', 'tot_loss': 'green', 'covariance_tok_reg_loss': 'orange', 'concept_description_loss': 'red'}
    values_to_add_to_title = ['concept_description_loss', 'covariance_tok_reg_loss']
    plot_smoothed = ['img_loss']

    plt.figure(figsize=(8, 5))
    
    for key, losses in loss_dict.items():
        if key == 'tot_loss':
            continue

        losses = np.array(losses)
        
        if len(losses) < window_length:
            continue

        if key in plot_smoothed:
            losses = savgol_filter(losses, window_length, polyorder)
            label = f'Smoothed {key}'
            linestyle = 'dashed'
        else:
            label = key
            linestyle = 'solid'

        plot_losses = losses / np.max(losses)
        color = colormap.get(key, default_color)  # Use the default color if the key is not in the colormap
        plt.plot(plot_losses, label=label, color=color, linestyle=linestyle)

    # Create the title:
    title = 'Loss values:'
    for key in values_to_add_to_title:
        if key in loss_dict:
            if loss_dict[key]:
                title += f' {key}: {loss_dict[key][-1]:.3f}'

    plt.title(title)
    plt.xlabel('Optimizer Step')
    plt.ylabel('Training Losses')
    plt.ylim(0, 1.1)  # Adjust the y-axis limits for normalized data
    plt.legend(loc='lower left')
    plt.savefig(save_path)
    plt.close()