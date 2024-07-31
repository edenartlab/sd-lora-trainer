from functools import reduce
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from typing import Optional
import torch
import torch.nn as nn
from diffusers.utils.deprecation_utils import deprecate
import torch.nn.functional as F
import math
from einops.layers.torch import Reduce
from torchtyping import TensorType
from einops import rearrange

# Find all instances of AttnProcessor2_0 in the UNet
def find_attnprocessor2_0(unet):

    module_names = []
    """
    this function assumes that there are fewer than 50 down blocks, attention modules and transformer blocks
    if you're not sure, feel free to set it to an arbitrarily large number
    don't worry, it won't slow anything down.
    """

    for block_type in ["down_blocks", "up_blocks"]:
        for down_block_index in range(50):
            for attentions_index in range(50):
                for transformer_blocks_index in range(50):
                    example_module_name = f"{block_type}.{down_block_index}.attentions.{attentions_index}.transformer_blocks.{transformer_blocks_index}.attn2.processor"

                    try:
                        module = get_module_by_name(module=unet, name = example_module_name)
                        assert isinstance(module, AttnProcessor2_0), f"Expected module to be an instance of AttnProcessor2_0 but found it to be: {type(module)}"
                        # print(f"Found: {example_module_name}")
                        module_names.append(example_module_name)
                    except AttributeError:
                        # print(f"Ignored name: {example_module_name}\nsince it does not exist")
                        pass
    print(f"Found: {len(module_names)} modules")
    return module_names

class DAAMLossAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name: str):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.name = name
        self.cross_attention_scores = None
        self.reduce_op = Reduce(
            "batch heads img text -> batch img text",
            reduction="sum"
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        """
        Mayukh's experiment
        """
        mayukh_experiment =  False
        if encoder_hidden_states is not None:
            """
            this triggers cross attn
            """
            mayukh_experiment = True
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if mayukh_experiment:
            # Calculate QK^T
            qk_t = torch.matmul(query, key.transpose(-2, -1))
            
            # Calculate attention scores (scaled QK^T)
            d_k = query.size(-1)  # Assuming the last dimension is the embedding dimension
            attention_scores = qk_t / math.sqrt(d_k)

            attention_scores = self.reduce_op(
                attention_scores,
            )
            self.cross_attention_scores = attention_scores

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class DAAMLoss:
    def __init__(self, attention_processors: list[DAAMLossAttnProcessor2_0]):
        self.attention_processors = attention_processors
        self.layer_names = [
            x.name for x in attention_processors
        ]

    def get_all_cross_attention_scores(self):
        cross_attention_scores = {}

        for p in self.attention_processors:
            cross_attention_scores[
                p.name
            ] = p.cross_attention_scores

        return cross_attention_scores

    def compute_single_token_loss(self, text_token_index: list[int], reduce = False):
        loss = {}
        cross_attention_scores = self.get_all_cross_attention_scores()

        for name, cross_attention_map in cross_attention_scores.items():
            """
            cross_attention_map.shape: (batch, image_patches, text_tokens)
            """
            assert cross_attention_map.ndim == 3
            loss[name] = cross_attention_map[:,:,text_token_index].norm() / cross_attention_map.shape[1]
        
        if reduce:
            all_losses = list(loss.values())
            return sum(all_losses)/len(all_losses)
        else:
            return loss

    def compute_loss(self, text_token_indices: list[int], reduce = False):
        losses = []
        for text_token_index in text_token_indices:
            losses.append(
                self.compute_single_token_loss(
                    text_token_index=text_token_index,
                    reduce = True
                )
            )

        if reduce:
            return sum(losses)/len(losses)
        else:
            return losses

    def get_image_heatmap(self, text_token_index: int, layer_name: str) -> TensorType["batch", "height", "width"]:
        cross_attention_scores = self.get_all_cross_attention_scores()
        assert layer_name in list(cross_attention_scores.keys())
        
        cross_attention_scores_single_token = cross_attention_scores[layer_name][:,:,text_token_index]
        assert cross_attention_scores_single_token.ndim == 2 ## batch, hw

        heatmap = rearrange(
            cross_attention_scores_single_token,
            "batch (height width) -> batch height width",
            height = int(math.sqrt(cross_attention_scores_single_token.shape[1])),
            width = int(math.sqrt(cross_attention_scores_single_token.shape[1]))
        )

        return heatmap

    def get_the_daam_heatmap(self, text_token_index: int) ->TensorType["batch", "height", "width"]:
        all_heatmaps = []
        for layer_name in self.layer_names:
            heatmap = self.get_image_heatmap(
                text_token_index=text_token_index,
                layer_name=layer_name
            )
            all_heatmaps.append(heatmap)

        ## each heatmap has a shape: batch, h, w where h=w
        ## now find the maximum possible height and width across all heatmaps
        max_height = max(heatmap.shape[1] for heatmap in all_heatmaps)
        max_width = max(heatmap.shape[2] for heatmap in all_heatmaps)

        ## now resize all_heatmaps to (batch, max_height, max_width) using F.interpolate
        resized_heatmaps = [
            F.interpolate(input = x.unsqueeze(1), size = (max_height, max_width)).squeeze(1)
            for x in all_heatmaps
        ]

        return sum(resized_heatmaps)


def get_module_by_name(module: nn.Module, name: str):
    """Retrieve a module nested in another by its access string."""
    if name == "":
        return module
    names = name.split(sep=".")
    return reduce(getattr, names, module)

def init_daam_loss(pipeline: StableDiffusionXLPipeline)-> tuple[StableDiffusionXLPipeline, DAAMLoss]:

    assert isinstance(pipeline, StableDiffusionXLPipeline)

    ## find out where the attention processor thingies are
    module_names = find_attnprocessor2_0(
        unet = pipeline.unet
    )

    all_daam_attention_processors = []
    # override the attention processor thingies
    for name in module_names:
        # print(f"Replacing: {name}")
        
        # Get parent module and attribute name
        parent_name = ".".join(name.split(".")[:-1])
        attr_name = name.split(".")[-1]

        # Get the parent module
        parent_module = get_module_by_name(module=pipeline.unet, name=parent_name)

        daam_attention_processor = DAAMLossAttnProcessor2_0(name=name)
        all_daam_attention_processors.append(daam_attention_processor)
        # Set the attribute
        setattr(parent_module, attr_name, daam_attention_processor)

        # Verify the replacement
        current_module = get_module_by_name(module=pipeline.unet, name=name)
        assert isinstance(current_module, DAAMLossAttnProcessor2_0)
        
    daam_loss = DAAMLoss(
        attention_processors=all_daam_attention_processors
    )
    return pipeline, daam_loss