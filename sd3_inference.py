"""
Pre-trained checkpoint:
https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
"""

import torch
from diffusers import StableDiffusion3Pipeline
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

# Load the pretrained model
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", 
    torch_dtype=torch.float16
)

# Load the LoRA weights from file
lora_weights_path = "lora_models/xander_sd15_final--03_03-40-49-sd15_face_lora_512_1.0_blip_5000/checkpoints/global_step_2600/transformer/pytorch_lora_weights.safetensors"
transformer_lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)
pipe.transformer  = get_peft_model(pipe.transformer, transformer_lora_config)

pipe.load_lora_into_transformer(
    load_file(lora_weights_path),
    transformer = pipe.transformer
)

# Move model to GPU
pipe = pipe.to("cuda")

from trainer.embedding_handler import TokenEmbeddingsHandler

from main_sd3 import compute_text_embeddings, load_sd3_tokenizers
prompts = [
    # 'in the style of <s0><s1>, airplane'
    # "<s0><s1>, there is a cartoon banana that is smoking weed on mars"
    "A man eating Waffles in Portugal"
]

tokenizers = load_sd3_tokenizers()

# embedding_handler = TokenEmbeddingsHandler(
#     text_encoders = text_encoders, 
#     tokenizers = tokenizers
# )

# embedding_handler.initialize_new_tokens(
#     inserting_toks=["<s0>","<s1>"],
#     starting_toks=None, 
#     seed=0
# )

# embedding_handler.load_embeddings(
#     file_path = "sd3_embeddings.safetensors",
#     txt_encoder_keys = ["1", "2", "3"]
# )

prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
    prompt = prompts, 
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3], 
    tokenizers = tokenizers,
    device="cuda:0"
)

image = pipe(
    prompt_embeds = prompt_embeds.half(),
    pooled_prompt_embeds = pooled_prompt_embeds.half(),
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
    generator = torch.Generator(device="cuda").manual_seed(0)
).images[0]
image.save("Sample.jpg")
print(f"Done!") 