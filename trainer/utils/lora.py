import os, json
import torch
from safetensors.torch import load_file
from typing import Dict
from peft import PeftModel
from ..dataset_and_utils import TokenEmbeddingsHandler
from safetensors.torch import save_file

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

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

    #pipe.load_lora_weights(lora_path, adapter_name = "my_adapter", weight_name="adapter_model.safetensors")
    #scales = {...}
    #pipe.set_adapters("my_adapter", scales)
    #pipe.set_adapters("my_adapter")
    #pipe.set_adapters(["my_adapter"], adapter_weights=[1.0])

    active_adapters = pipe.get_active_adapters()
    print(f"active_adapters: {active_adapters}")
    list_adapters_component_wise = pipe.get_list_adapters()
    print(f"list_adapters_component_wise: {list_adapters_component_wise}")
    
    # Load the textual_inversion token embeddings into the pipeline:
    try: #SDXL
        handler = TokenEmbeddingsHandler([pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2])
    except: #SD15
        handler = TokenEmbeddingsHandler([pipe.text_encoder, None], [pipe.tokenizer, None])

    embeddings_path = [f for f in os.listdir(lora_path) if f.endswith("embeddings.safetensors")][0]
    handler.load_embeddings(os.path.join(lora_path, embeddings_path))

    return pipe


from peft.utils import get_peft_model_state_dict
from diffusers.utils import (
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
)

def save_lora(
        output_dir, 
        global_step, 
        unet, 
        embedding_handler, 
        token_dict, 
        seed, 
        is_lora, 
        unet_lora_parameters, 
        unet_param_to_optimize,
        name: str = None
    ):
    """
    Save the LORA model to output_dir
    """
    print(f"Saving checkpoint at step.. {global_step}")

    if not is_lora:
        lora_tensors = {
            name: param
            for name, param in unet.named_parameters()
            if name in unet_param_to_optimize
        }
        save_file(lora_tensors, f"{output_dir}/unet.safetensors",)
    elif len(unet_lora_parameters) > 0:
        unet.save_pretrained(save_directory = output_dir)

    if 1:
        unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                #text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                #text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    # Make sure all weird delimiter characters are removed from concept_name before using it as a filepath:
    name = name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")

    embedding_handler.save_embeddings(f"{output_dir}/{name}_embeddings.safetensors")

    with open(f"{output_dir}/special_params.json", "w") as f:
        json.dump(token_dict, f)


######################################################


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