from functools import reduce
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from typing import Optional
import os
import torch
import torch.nn as nn
from diffusers.utils.deprecation_utils import deprecate
import torch.nn.functional as F
import math
from einops.layers.torch import Reduce
from torchtyping import TensorType
from einops import rearrange

import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_token_attention_loss(folder, pipe, daam_loss, captions, timesteps, token_attention_loss, global_step, img_ratio):
    batch_index = 0
    timestep = timesteps[batch_index].item()
    if timestep > 700:
        batch_index += 1
        timestep = timesteps[batch_index].item()

    folder = os.path.join(folder, "attention_heatmaps")
    os.makedirs(folder, exist_ok=True)

    token_strings = [pipe.tokenizer.decode(x) for x in pipe.tokenizer.encode(captions[batch_index])]
    plot_token_indices = range(1, len(token_strings) - 1)  # Exclude start and end tokens

    # Calculate global min and max for consistent colormap
    all_heatmaps = [daam_loss.get_the_daam_heatmap(text_token_index=i, img_ratio=img_ratio)[batch_index].cpu().detach().float() 
                    for i in plot_token_indices]
    vmin, vmax = np.min([h.min() for h in all_heatmaps]), np.max([h.max() for h in all_heatmaps])

    # Heatmap plots
    fig, axes = plt.subplots(nrows=1, ncols=len(plot_token_indices), 
                             figsize=(3 * len(plot_token_indices), 10))
    title_str = f"Token Attention Heatmaps (Step: {global_step})\nDenoise Timestep: {timesteps[batch_index].item()}"
    fig.suptitle(title_str, fontsize=16)
    
    for idx, text_token_index in enumerate(plot_token_indices):
        heatmap = all_heatmaps[idx]
        im = axes[idx].imshow(heatmap, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[idx].set_title(f"{token_strings[text_token_index]}")
        axes[idx].axis("off")
        
        # Add colorbar
        divider = make_axes_locatable(axes[idx])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, orientation="horizontal")

    plt.tight_layout()
    fig.savefig(os.path.join(folder, f"heatmaps_{global_step}.jpg"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Histogram plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Token Attention Distribution (Step: {global_step})", fontsize=16)
    ax.set_xlabel("Attention Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    for idx, text_token_index in enumerate(plot_token_indices):
        heatmap = all_heatmaps[idx]
        ax.hist(heatmap.reshape(-1), bins=30, label=token_strings[text_token_index], 
                alpha=0.5, density=True)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # Add token attention loss as text
    plt.text(0.95, 0.95, f"Token Attention Loss: {token_attention_loss.item():.4f}", 
             transform=ax.transAxes, ha='right', va='top', 
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

    plt.ylim(0, 0.6)
    fig.savefig(os.path.join(folder, f"histogram_{global_step}.jpg"), dpi=300, bbox_inches='tight')
    plt.close(fig)



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

    def process_and_stack_attention_scores(self, img_ratio: float):
        reshaped_tensors = []
        min_heatmap_pixels = np.inf

        # Process each attention score
        for processor in self.attention_processors:
            score = processor.cross_attention_scores
            bs, seq_len, channels = score.shape

            # Calculate width and height based on img_ratio
            width = round(math.sqrt(seq_len * img_ratio))
            height = round(width / img_ratio)
            reshaped_score = rearrange(score, 'b (h w) c -> b h w c', h=height, w=width)
            reshaped_tensors.append(reshaped_score)

            if height*width < min_heatmap_pixels:
                min_heatmap_pixels = height*width
                min_heatmap_shape = height, width

        # Interpolate and standardize all tensors to the same size
        for i, heatmap in enumerate(reshaped_tensors):
            if heatmap.shape[1] * heatmap.shape[2] != min_heatmap_pixels:
                # Interpolating to match the smallest tensor size uniformly
                heatmap = F.interpolate(heatmap.permute(0, 3, 1, 2), size=(min_heatmap_shape[0], min_heatmap_shape[1]), mode='bicubic').permute(0, 2, 3, 1)
                reshaped_tensors[i] = heatmap


        # Stack all tensors along the first dimension
        stacked_tensor = torch.stack(reshaped_tensors, dim=0)
        return stacked_tensor

    def get_all_cross_attention_scores(self):
        cross_attention_scores = {}

        for p in self.attention_processors:
            cross_attention_scores[
                p.name
            ] = p.cross_attention_scores

        return cross_attention_scores

    def get_image_heatmap(self, text_token_index: int, layer_name: str, img_ratio: float) -> TensorType["batch", "height", "width"]:
        cross_attention_scores = self.get_all_cross_attention_scores()
        assert layer_name in list(cross_attention_scores.keys())
        
        cross_attention_scores_single_token = cross_attention_scores[layer_name][:,:,text_token_index]

        width = round(math.sqrt(cross_attention_scores_single_token.shape[-1] * img_ratio))
        height = round(width / img_ratio)

        heatmap = rearrange(
            cross_attention_scores_single_token,
            "batch (height width) -> batch height width",
            height = height,
            width = width
        )

        return heatmap

    def get_the_daam_heatmap(self, text_token_index: int, img_ratio: float, resize = 'min') -> TensorType["batch", "height", "width"]:
        all_heatmaps = []
        for layer_name in self.layer_names:
            heatmap = self.get_image_heatmap(
                text_token_index=text_token_index,
                layer_name=layer_name,
                img_ratio = img_ratio
            )
            all_heatmaps.append(heatmap)

        if resize == 'max':
            heatmap_height = max(heatmap.shape[1] for heatmap in all_heatmaps)
            heatmap_width = max(heatmap.shape[2] for heatmap in all_heatmaps)
        elif resize == 'min':
            heatmap_height = min(heatmap.shape[1] for heatmap in all_heatmaps)
            heatmap_width = min(heatmap.shape[2] for heatmap in all_heatmaps)

        ## now resize all_heatmaps to (batch, heatmap_height, heatmap_width) using F.interpolate
        resized_heatmaps = [
            F.interpolate(input = x.unsqueeze(1), size = (heatmap_height, heatmap_width)).squeeze(1)
            for x in all_heatmaps
        ]

        resized_heatmaps = torch.stack(resized_heatmaps)

        ## Average the heatmaps:
        avg_heatmap = resized_heatmaps.mean(dim=0)

        return avg_heatmap

def get_module_by_name(module: nn.Module, name: str):
    """Retrieve a module nested in another by its access string."""
    if name == "":
        return module
    names = name.split(sep=".")
    return reduce(getattr, names, module)


def init_daam_loss(pipeline):
    ## find out where the attention processor thingies are
    module_names = find_attnprocessor2_0(
        unet = pipeline.unet
    )

    all_daam_attention_processors = []
    # override the attention processor thingies
    for name in module_names:
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