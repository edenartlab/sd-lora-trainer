"""
Pre-trained checkpoint:
https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
"""
import os
import torch
from diffusers import StableDiffusion3Pipeline
import torch

# Load the pretrained model
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", 
    torch_dtype=torch.float16,
    seed = 0
)

# Load the LoRA weights from file
lora_weights_path = "sd3-xander/checkpoint-1000/pytorch_lora_weights.safetensors"

# Move model to GPU
pipe = pipe.to("cuda")

prompts = [
    "This is a picture of a man holding a glass of beer. He is wearing a casual plaid shirt and jeans. The man is holding a frosty glass of golden beer with a thick, foamy head in his right hand, lifting it slightly as if making a toast. The background features wooden tables and chairs, vintage beer signs, and warm ambient lighting",
    "A close up shot of a man as a dragon rider with a red sword named Za'roc. His face is clearly visible in the high cinematic shot.",
    "A man in 2075, looking for the last drop of water in mars. 4k HDR",
    # "A king in Skyrim"
]
for idx, prompt in enumerate(prompts):
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    image.save(
        os.path.join(
            "./outputs",
            f"{idx}_baseline.jpg"
        )
    )

pipe.load_lora_weights(lora_weights_path, alpha = 8)

for idx, prompt in enumerate(prompts):
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    image.save(
        os.path.join(
            "./outputs",
            f"{idx}.jpg"
        )
    )

print(f"Done!")
