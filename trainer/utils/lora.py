import os, json
import torch
from safetensors.torch import load_file
from typing import Dict
from peft import PeftModel
from ..dataset_and_utils import TokenEmbeddingsHandler
from safetensors.torch import save_file
from .string import replace_in_string

'''
from diffusers.utils import (
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft
)
'''

def prepare_prompt_for_lora(prompt, lora_path, interpolation=False, verbose=True):
    if "_no_token" in lora_path:
        return prompt
        
    orig_prompt = prompt

    # Helper function to read JSON
    def read_json_from_path(path):
        with open(path, "r") as f:
            return json.load(f)

    # Check existence of "special_params.json"
    if not os.path.exists(os.path.join(lora_path, "special_params.json")):
        raise ValueError("This concept is from an old lora trainer that was deprecated. Please retrain your concept for better results!")

    token_map = read_json_from_path(os.path.join(lora_path, "special_params.json"))
    training_args = read_json_from_path(os.path.join(lora_path, "training_args.json"))
    
    try:
        lora_name = str(training_args["name"])
    except: # fallback for old loras that dont have the name field:
        return training_args["trigger_text"] + ", " + prompt

    lora_name_encapsulated = "<" + lora_name + ">"
    trigger_text = training_args["trigger_text"]

    try:
        mode = training_args["concept_mode"]
    except KeyError:
        try:
            mode = training_args["mode"]
        except KeyError:
            mode = "object"

    # Handle different modes
    if mode != "style":
        replacements = {
            "<concept>": trigger_text,
            "<concepts>": trigger_text + "'s",
            lora_name_encapsulated: trigger_text,
            lora_name_encapsulated.lower(): trigger_text,
            lora_name: trigger_text,
            lora_name.lower(): trigger_text,
        }
        prompt = replace_in_string(prompt, replacements)
        if trigger_text not in prompt:
            prompt = trigger_text + ", " + prompt
    else:
        style_replacements = {
            "in the style of <concept>": "in the style of TOK",
            f"in the style of {lora_name_encapsulated}": "in the style of TOK",
            f"in the style of {lora_name_encapsulated.lower()}": "in the style of TOK",
            f"in the style of {lora_name}": "in the style of TOK",
            f"in the style of {lora_name.lower()}": "in the style of TOK"
        }
        prompt = replace_in_string(prompt, style_replacements)
        if "in the style of TOK" not in prompt:
            prompt = "in the style of TOK, " + prompt
        
    # Final cleanup
    prompt = replace_in_string(prompt, {"<concept>": "TOK", lora_name_encapsulated: "TOK"})

    if interpolation and mode != "style":
        prompt = "TOK, " + prompt

    # Replace tokens based on token map
    prompt = replace_in_string(prompt, token_map)

    # Fix common mistakes
    fix_replacements = {
        r",,": ",",
        r"\s\s+": " ",  # Replaces one or more whitespace characters with a single space
        r"\s\.": ".",
        r"\s,": ","
    }
    prompt = replace_in_string(prompt, fix_replacements)

    if verbose:
        print('-------------------------')
        print("Adjusted prompt for LORA:")
        print(orig_prompt)
        print('-- to:')
        print(prompt)
        print('-------------------------')

    return prompt

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


def save_lora(output_dir, global_step, unet, embedding_handler, token_dict, args_dict, is_lora, unet_lora_parameters, unet_param_to_optimize_names):
    """
    Save the LORA model to output_dir, optionally with some example images

    """
    print(f"Saving checkpoint at step.. {global_step}")
    os.makedirs(output_dir, exist_ok=True)

    if not is_lora:
        lora_tensors = {
            name: param
            for name, param in unet.named_parameters()
            if name in unet_param_to_optimize_names
        }
        save_file(lora_tensors, f"{output_dir}/unet.safetensors",)
    elif len(unet_lora_parameters) > 0:
        unet.save_pretrained(save_directory = output_dir)

    try:
        concept_name = args_dict["name"].lower()
    except:
        concept_name = "eden_concept_lora"

    # Make sure all weird delimiter characters are removed from concept_name before using it as a filepath:
    concept_name = concept_name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")

    embedding_handler.save_embeddings(f"{output_dir}/{concept_name}_embeddings.safetensors",)

    with open(f"{output_dir}/special_params.json", "w") as f:
        json.dump(token_dict, f)
    with open(f"{output_dir}/training_args.json", "w") as f:
        json.dump(args_dict, f, indent=4)
