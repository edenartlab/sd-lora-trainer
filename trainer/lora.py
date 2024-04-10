import os, json
import torch
from safetensors.torch import load_file, save_file
from typing import Dict
from peft import PeftModel
from trainer.embedding_handler import TokenEmbeddingsHandler
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

def patch_pipe_with_lora(pipe, lora_path, lora_scale = 1.0):
    """
    update the pipe with the lora model and the token embeddings
    """

    # this loads the lora model into the pipeline at full strength (1.0)
    
    pipe.unet.load_adapter(lora_path, "eden_lora")

    #peft_model.set_adapter(["adapter1", "adapter2"])  # activate both adapters

    # First lets see if any lora's are active and unload them:
    #pipe.unet.unmerge_adapter()

    list_adapters_component_wise = pipe.get_list_adapters()
    print(f"list_adapters_component_wise: {list_adapters_component_wise}")

    #state_dict, network_alphas = pipe.lora_state_dict(lora_path)
    #for key in state_dict.keys():
    #    print(f"{key}")

    #pipe.load_lora_weights(lora_path, adapter_name = "eden_lora")#, weight_name="pytorch_lora_weights.safetensors")
    #scales = {...}
    #pipe.set_adapters("eden_lora", scales)

    if 1:
        for key in list_adapters_component_wise:
            adapter_names = list_adapters_component_wise[key]
            for adapter_name in adapter_names:
                print(f"Set adapter '{adapter_name}' of '{key}' with scale = {lora_scale:.2f}")
                pipe.set_adapters(adapter_name, adapter_weights=[lora_scale])
    
        pipe.unet.merge_adapter()

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
        name: str = None,
        text_encoder_peft_models: list = None
    ):
    """
    Save the model + embeddings to output_dir
    """
    print(f"Saving checkpoint at step.. {global_step}")
    # Make sure all weird delimiter characters are removed from concept_name before using it as a filepath:
    name = name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")

    embedding_handler.save_embeddings(f"{output_dir}/{name}_embeddings.safetensors")

    print("A")
    with open(f"{output_dir}/special_params.json", "w") as f:
        json.dump(token_dict, f)
    print("A")

    if text_encoder_peft_models is not None:
        for idx, model in enumerate(text_encoder_peft_models):
            save_directory = os.path.join(output_dir, f"text_encoder_lora_{idx}")
            model.save_pretrained(
                save_directory = save_directory
            )
            print(f"Saved text encoder {idx} in: {save_directory}")

    if not is_lora:
        lora_tensors = {
            name: param
            for name, param in unet.named_parameters()
            if name in unet_param_to_optimize
        }
        save_file(lora_tensors, f"{output_dir}/unet.safetensors",)
    elif len(unet_lora_parameters) > 0:
        unet.save_pretrained(save_directory = output_dir)

        lora_tensors = get_peft_model_state_dict(unet)
        unet_lora_layers_to_save = convert_state_dict_to_diffusers(lora_tensors)
        StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                #text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                #text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

        # Convert to WebUI format
        lora_state_dict = load_file(f"{output_dir}/pytorch_lora_weights.safetensors")
        peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
        kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
        save_file(kohya_state_dict, f"{output_dir}/{name}.safetensors")
    else:
        print("No lora parameters to save.")


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