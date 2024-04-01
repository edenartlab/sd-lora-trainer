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


def blend_conditions(embeds1, embeds2, lora_scale, 
        token_scale_power = 0.75,  # adjusts the curve of the interpolation
        min_token_scale   = 0.33,  # minimum token scale (corresponds to lora_scale = 0)
        verbose = True,
        ):
        
    """
    using lora_scale, apply linear interpolation between two sets of embeddings
    """

    c1, uc1, pc1, puc1 = embeds1
    c2, uc2, pc2, puc2 = embeds2

    token_scale = lora_scale ** token_scale_power
    # rescale the [0,1] range to [min_token_scale, 1] range:
    token_scale = min_token_scale + (1 - min_token_scale) * token_scale

    if verbose:
        print(f"Setting token_scale to {token_scale:.2f} (lora_scale = {lora_scale}, power = {token_scale_power})")
        print('-------------------------')
    try:
        c   = (1 - token_scale) * c1   + token_scale * c2
        pc  = (1 - token_scale) * pc1  + token_scale * pc2
        uc  = (1 - token_scale) * uc1  + token_scale * uc2
        puc = (1 - token_scale) * puc1 + token_scale * puc2
    except:
        print(f"Error in blending conditions, reverting to c2, uc2, pc2, puc2")
        token_scale = 1.0
        c   = c2
        pc  = pc2
        uc  = uc2
        puc = puc2

    return (c, uc, pc, puc), token_scale

def patch_pipe_with_lora(pipe, lora_path, lora_scale = 1.0):
    """
    update the pipe with the lora model and the token embeddings
    """

    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.unet.merge_adapter()

    list_adapters_component_wise = pipe.get_list_adapters()
    print(f"list_adapters_component_wise: {list_adapters_component_wise}")

    #pipe.load_lora_weights(lora_path, adapter_name = "my_adapter", weight_name="pytorch_lora_weights.safetensors")
    #scales = {...}
    #pipe.set_adapters("my_adapter", scales)

    for key in list_adapters_component_wise:
        adapter_names = list_adapters_component_wise[key]
        for adapter_name in adapter_names:
            print(f"Set adapter '{adapter_name}' of '{key}' with scale = {lora_scale}")
            pipe.set_adapters(adapter_name, adapter_weights=[lora_scale])

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

    # Make sure all weird delimiter characters are removed from concept_name before using it as a filepath:
    name = name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")

    if not is_lora:
        lora_tensors = {
            name: param
            for name, param in unet.named_parameters()
            if name in unet_param_to_optimize
        }
        save_file(lora_tensors, f"{output_dir}/unet.safetensors",)
    elif len(unet_lora_parameters) > 0:
        unet.save_pretrained(save_directory = output_dir)

    if 0:
        unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                #text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                #text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    if 1:
        #lora_tensors = unet_attn_processors_state_dict(unet)
        lora_tensors = get_peft_model_state_dict(unet)
        save_file(lora_tensors, f"{output_dir}/{name}_lora_orig.safetensors")
        
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