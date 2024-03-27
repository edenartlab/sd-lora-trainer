import os, json
import torch
from safetensors.torch import load_file
from typing import Dict
from peft import PeftModel
from ..dataset_and_utils import TokenEmbeddingsHandler
from safetensors.torch import save_file

'''
from diffusers.utils import (
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft
)
'''

def patch_pipe_with_lora(pipe, lora_path):
    """
    update the pipe with the lora model and the token embeddings
    """

    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.unet.merge_adapter()
    
    # Load the textual_inversion token embeddings into the pipeline:
    try: #SDXL
        handler = TokenEmbeddingsHandler([pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2])
    except: #SD15
        handler = TokenEmbeddingsHandler([pipe.text_encoder, None], [pipe.tokenizer, None])

    embeddings_path = [f for f in os.listdir(lora_path) if f.endswith("embeddings.safetensors")][0]
    handler.load_embeddings(os.path.join(lora_path, embeddings_path))

    return pipe


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    """
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[
                f"{attn_processor_key}.{parameter_key}"
            ] = parameter

    return attn_processors_state_dict


def save_lora(
        output_dir, 
        global_step, 
        unet, 
        embedding_handler, 
        token_dict, 
        seed, 
        is_lora, 
        unet_lora_parameters, 
        unet_param_to_optimize_names,
        name: str = None
    ):
    """
    Save the LORA model to output_dir, optionally with some example images
    """
    print(f"Saving checkpoint at step.. {global_step}")

    if not is_lora:
        lora_tensors = {
            name: param
            for name, param in unet.named_parameters()
            if name in unet_param_to_optimize_names
        }
        save_file(lora_tensors, f"{output_dir}/unet.safetensors",)
    elif len(unet_lora_parameters) > 0:
        unet.save_pretrained(save_directory = output_dir)

    # Make sure all weird delimiter characters are removed from concept_name before using it as a filepath:
    name = name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")

    embedding_handler.save_embeddings(f"{output_dir}/{name}_embeddings.safetensors",)

    with open(f"{output_dir}/special_params.json", "w") as f:
        json.dump(token_dict, f)