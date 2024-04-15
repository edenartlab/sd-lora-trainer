class OptimizerCollection:
    def __init__(
        self,
        optimizer_unet = None,
        optimizer_textual_inversion = None,
        optimizer_text_encoder_lora = None
    ):
        """
        run operations on all the relevant optimizers with a single function call
        """
        self.optimizer_unet=optimizer_unet
        self.optimizer_textual_inversion=optimizer_textual_inversion
        self.optimizer_text_encoder_lora=optimizer_text_encoder_lora
        
        self.optimizers = [
            self.optimizer_unet,
            self.optimizer_textual_inversion,
            self.optimizer_text_encoder_lora
        ]

    def zero_grad(self):
        for optimizer in self.optimizers:
            if optimizer is not None:
                optimizer.zero_grad()
    
    def step(self):
        for optimizer in self.optimizers:
            if optimizer is not None:
                optimizer.step()