import torch

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