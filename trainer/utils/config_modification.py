from ..config import TrainingConfig

def modify_config_based_on_concept_mode(config: TrainingConfig):
    if config.concept_mode == "face":
        config.left_right_flip_augmentation = False  # always disable lr flips for face mode!
        config.mask_target_prompts = "face"
        config.clipseg_temperature = 0.4

    if config.concept_mode == "concept": # gracefully catch any old versions of concept_mode
        config.concept_mode = "object"

    if config.concept_mode == "style": # for styles you usually want the LoRA matrices to absorb a lot (instead of just the token embedding)
        config.l1_penalty = 0.05
    return config