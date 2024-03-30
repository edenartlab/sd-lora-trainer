def get_avg_lr(optimizer):
    try:
        # Calculate the weighted average effective learning rate
        total_lr = 0
        total_params = 0
        for group in optimizer.param_groups:
            d = group['d']
            lr = group['lr']
            bias_correction = 1  # Default value
            if group['use_bias_correction']:
                beta1, beta2 = group['betas']
                k = group['k']
                bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))

            effective_lr = d * lr * bias_correction

            # Count the number of parameters in this group
            num_params = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_lr += effective_lr * num_params
            total_params += num_params

        if total_params == 0:
            return 0.0
        else: return total_lr / total_params
    except:
        return optimizer.param_groups[0]['lr']