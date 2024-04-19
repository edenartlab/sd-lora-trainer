import os
from peft.utils import get_peft_model_state_dict
from diffusers.utils import (
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
)
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from safetensors.torch import load_file, save_file
import torch
from diffusers import EulerDiscreteScheduler
from peft import PeftModel
from .lora import patch_pipe_with_lora
from .utils.json_stuff import save_as_json

def remove_delimiter_characters(name: str):
    # Make sure all weird delimiter characters are removed from concept_name before using it as a filepath:
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")

# Convert to WebUI format

def convert_pytorch_lora_safetensors_to_webui(
    pytorch_lora_weights_filename: str,
    output_filename: str
):
    assert os.path.exists(pytorch_lora_weights_filename), f"Invalid path: {pytorch_lora_weights_filename}"
    lora_state_dict = load_file(pytorch_lora_weights_filename)
    peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
    kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
    save_file(kohya_state_dict, output_filename)

def save_checkpoint(
    output_dir: str, 
    global_step: int, 
    unet, 
    embedding_handler, 
    token_dict: dict, 
    is_lora: bool, 
    unet_lora_parameters, 
    name: str = None,
    text_encoder_peft_models: list = [None]
):
    """
    Save the model's embeddings and special parameters to the specified directory.

    Note: This function directly corresponds to the `load_checkpoint` method

    Args:
        `output_dir` (str): The directory path where the checkpoint will be saved.
        `global_step` (int): The current global step or epoch number.
        `unet`: The main model to save.
        `embedding_handler`: The handler for saving embeddings.
        `token_dict` (dict): Special parameters associated with the model.
        `is_lora` (bool): Whether the model includes LoRA components.
        `unet_lora_parameters`: Parameters associated with the LoRA components.
        `name` (str, optional): Name identifier for the checkpoint. Defaults to None.
        `text_encoder_peft_models` (list, optional): List of additional text encoder models to save. Defaults to None.

    Returns:
        None

    Saves:
        - {name}_embeddings.safetensors: Embeddings of the model.
        - special_params.json: Special parameters of the model.

    If `text_encoder_peft_models` is provided, saves each model in a separate directory with the
    following structure:
        - text_encoder_lora_{index}/
            - adapter_config.json
            - adapter_model.safetensors
            - README.md

    If `is_lora` is True, saves additional LoRA-related data:
        - LoRA weights
        - LoRA weights converted for web UI

    If `is_lora` is False then it assumes that it's a vanilla unet model and saves it in the usual huggingface way.

    """
    print(f"Saving checkpoint at step.. {global_step}")
    name = remove_delimiter_characters(name)

    embedding_handler.save_embeddings(
        os.path.join(
            output_dir,
            f"{name}_embeddings.safetensors"
        )
    )

    save_as_json(
        token_dict,
        filename = os.path.join(
            output_dir, "special_params.json"
        )
    )
    
    # Separate txt_lora saves:
    #for idx, model in enumerate(text_encoder_peft_models):
    #    if model is not None:
    #        save_directory = os.path.join(output_dir, f"text_encoder_lora_{idx}")
    #        model.save_pretrained(
    #            save_directory = save_directory
    #        )
    #        print(f"Saved text encoder {idx} in: {save_directory}")

    if is_lora:
        assert len(unet_lora_parameters) > 0, f"Expected len(unet_lora_parameters) to be greater than zero if is_lora is True"
        
        # This saves adapter_config.json:
        # TODO: adjust inference.py so it can load everything without needing this file
        # unet.save_pretrained(save_directory = output_dir)

        text_encoder_lora_layers = [None, None]
        for idx, model in enumerate(text_encoder_peft_models):
            if model is not None:
                lora_tensors = get_peft_model_state_dict(model)
                text_encoder_lora_layers[idx] = convert_state_dict_to_diffusers(lora_tensors)
                
        lora_tensors = get_peft_model_state_dict(unet)
        unet_lora_layers_to_save = convert_state_dict_to_diffusers(lora_tensors)

        # TODO have a separate save for non SDXL model:
        StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers[0],
                text_encoder_2_lora_layers=text_encoder_lora_layers[1],
            )

        convert_pytorch_lora_safetensors_to_webui(
            pytorch_lora_weights_filename=os.path.join(output_dir, "pytorch_lora_weights.safetensors"),
            output_filename=os.path.join(output_dir, f"{name}.safetensors")
        )
    else:
        unet.save_pretrained(save_directory = output_dir)

def load_checkpoint(
    pretrained_model_version: str,
    pretrained_model_path: str,
    checkpoint_folder: str,
    is_lora: bool,
    device: str
):
    """
    Load a pre-trained model checkpoint and prepare it for inference.

     Note: This function directly corresponds to the `save_checkpoint` method

    Args:
        `pretrained_model_version` (`str`): Version of the pre-trained model (`sd15` or `sdxl`).
        `pretrained_model_path` (`str`): Path to the pre-trained model file.
        `checkpoint_folder` (`str`): Path to the checkpoint folder.
        `is_lora` (`bool`): Whether LoRA model components are used.
        `device` (Union[`str`, `torch.device`]): Device for inference.

    Raises:
        NotImplementedError: If an unsupported `pretrained_model_version` is provided.
    """

    assert os.path.exists(pretrained_model_path), f"Invalid pretrained_model_path: {pretrained_model_path}"
    
    if pretrained_model_version == "sd15":
        pipe = StableDiffusionPipeline.from_single_file(
            pretrained_model_path, torch_dtype=torch.float16, use_safetensors=True)
    elif pretrained_model_version == "sdxl":
        pipe = StableDiffusionXLPipeline.from_single_file(
            pretrained_model_path, torch_dtype=torch.float16, use_safetensors=True)
    else:
        raise NotImplementedError(f"Invalid pretrained_model_version: {pretrained_model_version}")

    pipe = pipe.to('cuda', dtype=torch.float16)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config) #, timestep_spacing="trailing")

    assert os.path.exists(checkpoint_folder), f"Invalid checkpoint_folder: {checkpoint_folder}"    
    text_encoder_0_path =  os.path.join(
            checkpoint_folder, "text_encoder_lora_0"
        )
    text_encoder_1_path =  os.path.join(
            checkpoint_folder, "text_encoder_lora_1"
        )
    if os.path.exists(
        text_encoder_0_path
    ):
        
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_0_path)
        print(f"loaded text_encoder LoRA from: {text_encoder_0_path}")
    
    if os.path.exists(
        text_encoder_1_path
    ):
        
        pipe.text_encoder_2 = PeftModel.from_pretrained(pipe.text_encoder_2, text_encoder_1_path)
        print(f"loaded text_encoder LoRA from: {text_encoder_1_path}")

    if is_lora:
        pipe.unet = PeftModel.from_pretrained(model = pipe.unet, model_id = checkpoint_folder, adapter_name = 'eden_lora')

        pipe = pipe.to(device)
        pipe = patch_pipe_with_lora(pipe, checkpoint_folder)
    else:
        pipe.unet = pipe.unet.from_pretrained(checkpoint_folder)
        pipe = pipe.to(device, dtype=torch.float16)
        print(f"Successfully loaded full checkpoint for inference!")

    return pipe