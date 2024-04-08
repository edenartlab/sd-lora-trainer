import torch
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support

def compute_grad_norm(parameters, norm_type = 2.0, foreach = None, error_if_nonfinite = False):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    first_device = grads[0].device
    grouped_grads = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])
    norms = []
    for ((device, _), ([grads], _)) in grouped_grads.items():
        if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
            norms.extend(torch._foreach_norm(grads, norm_type))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grads])

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    return total_norm

def conditioning_norm_regularization_loss(loss, config, prompt_embeds_norms, prompt_embeds):
    # Adds a regularization loss to norms of the prompt_conditioning vectors.

    # Create a loss to fix the regularization norm to a target value:
    if config.sd_model_version == 'sdxl':
        target_norm = 34.8
    if config.sd_model_version == 'sd15':
        target_norm = 27.8

    if config.cond_reg_w > 0.0:
        # Compute the norms of the conditioning signals:
        conditioning_norms = prompt_embeds.norm(dim=-1).mean(dim=0)
        regularization_norm_value = conditioning_norms[2:].mean()
        regularization_loss = (regularization_norm_value - target_norm).pow(2)
        prompt_embeds_norms['main'].append(regularization_norm_value.item())
        loss += config.cond_reg_w * regularization_loss

    return loss, prompt_embeds_norms

from trainer.utils.inference import get_conditioning_signals

def tok_conditioning_norm_regularization_loss(loss, config, prompt_embeds_norms, pipe, embedding_handler):
    # Create a loss to fix the regularization norm to a target value:
    if config.sd_model_version == 'sdxl':
        target_norm = 34.8
    if config.sd_model_version == 'sd15':
        target_norm = 27.8

    # These are some hardcoded, quite arbitrary prompts that contain the new tokens:
    reg_captions = ["a photo of TOK", "TOK", "a photo of TOK next to TOK", "TOK and TOK"]
    token_replacement = config.token_dict["TOK"]
    
    if config.tok_cond_reg_w > 0.0: # regularize the txt-conditioning of several regularization prompts containing the new tokens:

        reg_captions = [caption.replace("TOK", token_replacement) for caption in reg_captions]
        reg_token_indices = [[],[]]

        for i, tokenizer in enumerate(embedding_handler.tokenizers):
            if tokenizer is not None:
                for caption in reg_captions:
                    token_indices = tokenizer(
                        caption,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt",
                    ).input_ids.squeeze()
                    reg_token_indices[i].append(token_indices)
                reg_token_indices[i] = torch.stack(reg_token_indices[i])
            else:
                reg_token_indices[i] = None

        reg_prompt_embeds, reg_pooled_prompt_embeds, reg_add_time_ids = get_conditioning_signals(
        config, pipe, reg_token_indices, embedding_handler.text_encoders
        )

        reg_conditioning_norms = reg_prompt_embeds.norm(dim=-1).mean(dim=0)
        regularization_norm_value = reg_conditioning_norms[2:].mean()
        regularization_loss = (regularization_norm_value - target_norm).pow(2)
        loss += config.tok_cond_reg_w * regularization_loss
        prompt_embeds_norms['reg'].append(regularization_norm_value.item())

    return loss, prompt_embeds_norms