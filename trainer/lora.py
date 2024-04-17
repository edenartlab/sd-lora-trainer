import os, json
import torch
from typing import Dict
from peft import PeftModel
from trainer.embedding_handler import TokenEmbeddingsHandler

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