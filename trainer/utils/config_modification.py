from ..config import TrainingConfig

def modify_args_based_on_concept_mode(
    concept_mode: str,
    left_right_flip_augmentation: bool,
    mask_target_prompts: str,
    clipseg_temperature: float,
    l1_penalty: float
):
    if concept_mode == "face":
        left_right_flip_augmentation = False  # always disable lr flips for face mode!
        mask_target_prompts = "face"
        clipseg_temperature = 0.4

    if concept_mode == "concept": # gracefully catch any old versions of concept_mode
        concept_mode = "object"

    if concept_mode == "style": # for styles you usually want the LoRA matrices to absorb a lot (instead of just the token embedding)
        l1_penalty = 0.05

    return (
        concept_mode,
        left_right_flip_augmentation,
        mask_target_prompts,
        clipseg_temperature,
        l1_penalty
    )