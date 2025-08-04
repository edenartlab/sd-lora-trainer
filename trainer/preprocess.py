import gc
import fnmatch
import mimetypes
import os
import time
import re
import shutil
import tarfile
import base64
import requests
import concurrent.futures
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from zipfile import ZipFile

from PIL import ImageEnhance, ImageFilter, Image
import random
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import (
    BlipForConditionalGeneration,
    Blip2ForConditionalGeneration,
    BlipProcessor,
    Blip2Processor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
    Swin2SRForImageSuperResolution,
    Swin2SRImageProcessor,
)

from trainer.utils.io import download_and_prep_training_data
from trainer.utils.utils import fix_prompt
from trainer.config import model_paths

import re
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI API key loaded")
except:
    OPENAI_API_KEY = None
    client = None
    print("WARNING: Could not find OPENAI_API_KEY in .env, disabling gpt prompt generation.")

# Put some boundaries to make the gpt pass work well: (very long text often confuses the model and also costs more money...)
MIN_GPT_PROMPTS = 3
MAX_GPT_PROMPTS = 80

def _find_files(pattern, dir="."):
    """Return list of files matching pattern in a given directory, in absolute format.
    Unlike glob, this is case-insensitive.
    """
    rule = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    return [os.path.join(dir, f) for f in os.listdir(dir) if rule.match(f)]


def preprocess(
    config,
    working_directory,
    concept_mode, 
    input_zip_path: Path,
    caption_text: str,
    mask_target_prompts: str,
    target_size: int,
    crop_based_on_salience: bool,
    use_face_detection_instead: bool,
    left_right_flip_augmentation: bool = False,
    augment_imgs_up_to_n: int = 0,
    caption_model: str = "blip",
    seed: int = 0,
) -> Path:

    if os.path.exists(working_directory):
        shutil.rmtree(working_directory)
    os.makedirs(working_directory)

    # Setup directories for the training data:
    TEMP_IN_DIR  = os.path.join(working_directory, "images_in")
    TEMP_OUT_DIR = os.path.join(working_directory, "images_out")

    for path in [TEMP_OUT_DIR, TEMP_IN_DIR]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    download_and_prep_training_data(input_zip_path, TEMP_IN_DIR)

    config = load_and_save_masks_and_captions(
        config,
        concept_mode, 
        files=TEMP_IN_DIR,
        output_dir=TEMP_OUT_DIR,
        seed=seed,
        caption_text=caption_text,
        mask_target_prompts=mask_target_prompts,
        target_size=target_size,
        crop_based_on_salience=crop_based_on_salience,
        use_face_detection_instead=use_face_detection_instead,
        add_lr_flips = left_right_flip_augmentation,
        augment_imgs_up_to_n = augment_imgs_up_to_n,
        caption_model = caption_model
    )

    return config, Path(TEMP_OUT_DIR)


@torch.no_grad()
@torch.cuda.amp.autocast()
def swin_ir_sr(
    images: List[Image.Image],
    model_id: Literal[
        "caidas/swin2SR-classical-sr-x2-64",
        "caidas/swin2SR-classical-sr-x4-48",
        "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    ] = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    target_size: Optional[Tuple[int, int]] = None,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    **kwargs,
) -> List[Image.Image]:
    """
    Upscales images using SwinIR. Returns a list of PIL images.
    If the image is already larger than the target size, it will not be upscaled
    and will be returned as is.

    """

    model = Swin2SRForImageSuperResolution.from_pretrained(
        model_id, cache_dir = model_paths.get_path("SR")
    ).to(device)
    processor = Swin2SRImageProcessor()

    out_images = []

    for image in tqdm(images):
        ori_w, ori_h = image.size
        if target_size is not None:
            if ori_w >= target_size[0] and ori_h >= target_size[1]:
                out_images.append(image)
                continue

        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        output = (
            outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        )
        output = np.moveaxis(output, source=0, destination=-1)
        output = (output * 255.0).round().astype(np.uint8)
        output = Image.fromarray(output)

        out_images.append(output)

    return out_images


@torch.no_grad()
@torch.cuda.amp.autocast()
def clipseg_mask_generator(
    images: List[Image.Image],
    target_prompts: Union[List[str], str],
    model_id: Literal[
        "CIDAS/clipseg-rd64-refined", "CIDAS/clipseg-rd16"
    ] = "CIDAS/clipseg-rd64-refined",
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    bias: float = 0.01,
    temp: float = 1.0,
    **kwargs,
) -> List[Image.Image]:
    # Handle single prompt case
    if isinstance(target_prompts, str):
        print(f'Using "{target_prompts}" as CLIP-segmentation prompt for all images.')
        target_prompts = [target_prompts] * len(images)

    # Initialize model and processor
    model = None
    if any(target_prompts):
        processor = CLIPSegProcessor.from_pretrained(
            model_id, cache_dir = model_paths.get_path("CLIP")
        )
        model = CLIPSegForImageSegmentation.from_pretrained(
            model_id, cache_dir = model_paths.get_path("CLIP")
        ).to(device)

    masks = []

    for image, prompt in tqdm(zip(images, target_prompts)):
        original_size = image.size
        print(f"Original image size: {original_size}")

        if prompt != "":
            # Manually resize image to 224x224
            input_image = image.resize((224, 224), Image.Resampling.LANCZOS)
            print(f"Resized image size: {input_image.size}")

            # Process text with tokenizer (no image-specific args)
            text_inputs = processor.tokenizer(
                [prompt, ""], return_tensors = "pt", padding = True, truncation = True
            )

            # Process images with image_processor
            image_inputs = processor.image_processor(
                images = [input_image] * 2,
                return_tensors = "pt",
                do_resize = False,  # We've already resized
                do_normalize = True,
            )

            # Combine inputs
            inputs = { **text_inputs, **image_inputs }
            inputs = { k: v.to(device) for k, v in inputs.items() }

            # Debug: Check input shapes
            print(f"Input pixel_values shape: {inputs['pixel_values'].shape}")

            # Run model
            outputs = model(**inputs)

            # Process logits
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits / temp, dim=0)[0]
            probs = (probs + bias).clamp_(0, 1)
            probs = 255 * probs / probs.max()

            # Create and resize mask
            mask = Image.fromarray(probs.cpu().numpy()).convert("L")
            mask = mask.resize(original_size, Image.Resampling.BILINEAR)
        else:
            mask = Image.new("L", original_size, 255)

        masks.append(mask)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return masks

import textwrap
def cleanup_prompts_with_chatgpt(
    prompts, 
    concept_mode,    # face / object / style
    seed,  # seed for chatgpt reproducibility
    verbose = True):

    if concept_mode == "object":
        chat_gpt_prompt_1 = textwrap.dedent("""
            Analyze a set of (poor) image descriptions each featuring the same concept, figure or thing.
            Tasks:
            1. Deduce a concise (max 10 words) visual description of just the concept TOK (Concept Description), try to be as visually descriptive of TOK as possible!
            2. Substitute the concept in each description with the placeholder "TOK", rearranging or adjusting the text where needed. Hallucinate TOK into the description if necessary (but dont mention when doing so, simply provide the final description)!
            3. Streamline each description to its core elements, ensuring clarity and mandatory inclusion of the placeholder string "TOK".
            The descriptions are:""")
        
        chat_gpt_prompt_2 = textwrap.dedent("""
            Respond with "Concept Description: ..." followed by a list (using "-") of all the revised descriptions, each mentioning "TOK".
            """)

    elif concept_mode == "face":
        chat_gpt_prompt_1 = textwrap.dedent("""
            Analyze a set of (poor) image descriptions, each featuring a person named TOK.
            Tasks:
            1. Deduce a concise (max 10 words) visual description of TOK (TOK Description), try to be as visually descriptive of TOK as possible, always mention their skin color, hallucinate a basic description if necessary (eg black man with long beard).
            2. Rewrite each description, injecting "TOK" naturally into each description, adjusting where needed.
            3. Streamline each description to focus on the context and surroundings of TOK instead of the visual appearance of TOK's face. Ensure mandatory inclusion of "TOK".
            The descriptions are:""")
        
        chat_gpt_prompt_2 = textwrap.dedent("""
            Respond with "TOK Description: ..." followed by a list (using "-") of all the revised descriptions, each mentioning "TOK".
            """)

    elif concept_mode == "style":
        chat_gpt_prompt_1 = textwrap.dedent("""
            Analyze a set of (poor) image descriptions, each featuring an example of a common aesthetic style named TOK.
            Tasks:
            1. Deduce a concise (max 7 words) visual description of the aesthetic style (Style Description).
            2. Rewrite each description to focus solely on the non-stylistic contents of the image like characters, objects, colors, scene, context etc but not the stylistic elements captured by TOK.
            3. Integrate "in the style of TOK" naturally into each description, typically at the beginning while summarizing each description to its core elements, ensuring clarity and mandatory inclusion of "TOK".
            The descriptions are:""")
        
        chat_gpt_prompt_2 = textwrap.dedent("""
            Respond with "Style Description: ..." followed by a list (using "-") of all the revised descriptions, each mentioning "in the style of TOK".
            """)

    final_chatgpt_prompt = chat_gpt_prompt_1 + "\n- " + "\n- ".join(prompts) + "\n" + chat_gpt_prompt_2
    print("Final chatgpt prompt:")
    print(final_chatgpt_prompt)
    print("--------------------------")
    print(f"Calling chatgpt with seed {seed}...")

    response = client.chat.completions.create(
            model="gpt-4o",
            seed=seed,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_chatgpt_prompt},
            ])

    gpt_completion = response.choices[0].message.content

    if verbose: # pretty print the full response json:
        print("----- GPT response: -----")
        print(gpt_completion)
        print("--------------------------")
    
    # extract the final rephrased prompts from the response:
    prompts = []
    for line in gpt_completion.split("\n"):
        if line.startswith("-") or re.match(r'^\d+\.', line):
            prompts.append(line[2:])

    gpt_concept_description = extract_gpt_concept_description(gpt_completion, concept_mode)
    trigger_text = "TOK"
    if concept_mode == 'style':
        trigger_text = "in the style of TOK, "

    return prompts, gpt_concept_description, trigger_text

def extract_gpt_concept_description(gpt_completion, concept_mode):
    """
    Extracts the concept name from the GPT completion based on the concept mode.
    """

    if concept_mode == 'face':
        prefix = "TOK Description:"
    elif concept_mode == 'style':
        prefix = "Style Description:"
    elif concept_mode == 'object':
        prefix = "Concept Description:"

    for line in gpt_completion.split("\n"):
        if line.startswith(prefix):
            concept_name = line[len(prefix):].strip()
            break

    return concept_name


def post_process_captions(captions, text, concept_mode, job_seed, skip_gpt_cleanup=False):
    text = text.strip()
    gpt_cleanup_worked = False
    gpt_concept_description = None

    if (len(captions) >= MIN_GPT_PROMPTS and len(captions) <= MAX_GPT_PROMPTS and not text and client) and not skip_gpt_cleanup:
        retry_count = 0
        while retry_count < 5:
            try:
                gpt_captions, gpt_concept_description, trigger_text = cleanup_prompts_with_chatgpt(captions, concept_mode, job_seed + retry_count)
                n_toks = sum("TOK" in caption for caption in gpt_captions)
                
                if n_toks > int(0.8 * len(captions)) and (len(gpt_captions) == len(captions)):
                    # gpt-cleanup (mostly) worked, lets just ensure every caption contains "TOK" and finish
                    print("Making sure TOK is added to every training prompt...")
                    gpt_captions = ["TOK, " + caption if "TOK" not in caption else caption for caption in gpt_captions]
                    captions = gpt_captions
                    gpt_cleanup_worked = True
                    break
                else:
                    if len(gpt_captions) == len(captions):
                        print(f'GPT-4 did not return enough {n_toks}/{len(captions)} prompts containing "TOK", retrying...')
                    else:
                        print(f'GPT-4 returned the wrong number of prompts {len(gpt_captions)} instead of {len(captions)}, retrying...')
                    retry_count += 1
                    gpt_cleanup_worked = False

            except Exception as e:
                retry_count += 1
                gpt_cleanup_worked = False
                print(f"An error occurred after try {retry_count}: {e}")
                time.sleep(0.5)

    if not gpt_cleanup_worked:
        # simple concat of trigger text with rest of prompt:
        if len(text) == 0:
            print("WARNING: no captioning text was given and we're not doing chatgpt cleanup...")
            print("Concept mode: ", concept_mode)
            if concept_mode == "style":
                trigger_text = "in the style of TOK, "
                captions = [trigger_text + caption for caption in captions]
            else:
                trigger_text = "TOK, "
                captions = [trigger_text + caption for caption in captions]
        else:
            trigger_text = text
            captions = [trigger_text + ", " + caption for caption in captions]

    captions = [fix_prompt(caption) for caption in captions]
    return captions, trigger_text, gpt_concept_description


def blip_caption_dataset(
        images: List[Image.Image],
        captions: List[str],
        model_id: Literal[
            "Salesforce/blip-image-captioning-large",
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip2-opt-2.7b",
        ] = "Salesforce/blip-image-captioning-large"
        ):
    
    # If non of the captions are None, we dont need to do anything:
    if all(captions):
        print(f"All captions are already generated, skipping captioning...")
        return captions

    print(f"Using model {model_id} for image captioning...")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "blip2" in model_id:
        processor = Blip2Processor.from_pretrained(model_id, cache_dir = model_paths.get_path("BLIP"))
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, cache_dir = model_paths.get_path("BLIP"), torch_dtype=torch.float16
        ).to(device)
    else:
        processor = BlipProcessor.from_pretrained(model_id, cache_dir = model_paths.get_path("BLIP"))
        model = BlipForConditionalGeneration.from_pretrained(
            model_id, cache_dir = model_paths.get_path("BLIP"), torch_dtype=torch.float16
        ).to(device)

    for i, image in enumerate(tqdm(images)):
        if captions[i] is None:
            inputs = processor(image, return_tensors="pt").to(device, torch.float16)
            out = model.generate(**inputs, max_length=100, do_sample=True, top_k=40, temperature=0.65)
            captions[i] = processor.decode(out[0], skip_special_tokens=True)

    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return captions

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

import uuid
def prep_img_for_gpt_api(pil_img, max_size=(512, 512)):
    # create a temporary file to save the resized image:
    resized_img = pil_img.copy()
    resized_img.thumbnail(max_size, Image.Resampling.LANCZOS)
    output_path = f"temp_{uuid.uuid4()}.jpg"
    resized_img.save(output_path, quality=95)
    base64_image = encode_image(output_path)
    os.remove(output_path)
    return base64_image

def gpt4_v_caption_dataset(
    images, captions, 
    batch_size=4,
    ):

    if not OPENAI_API_KEY:
        print(f"Skipping GPT-4 Vision captioning because OPENAI_API_KEY is not set.")
        return captions

    prompt = "Concisely describe this image without assumptions with at most 20 words. Dont start with statements like 'The image features...', just describe what you see."
    #prompt = "Concisely describe the main subject(s) in the image without assumptions with at most 20 words. Ignore the background and only describe the people / animals / objects in the foreground. Dont start with statements like 'The image features...', just describe what you see."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    def fetch_caption(index, img):
        base64_image = prep_img_for_gpt_api(img,  max_size=(512, 512))

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}}
                    ]
                }
            ],
            "max_tokens": 60
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
        try:
            result = response.json()["choices"][0]["message"]["content"]
        except:
            print(response.json())
            result = ""

        return index, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_index = {executor.submit(fetch_caption, i, img): i for i, img in enumerate(images) if captions[i] is None}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                captions[index] = future.result()[1]
                print(f"Caption {index + 1}/{len(images)}: {captions[index]}")
            except Exception as exc:
                captions[index] = None
                print(f"Caption generation for image {index + 1} failed with exception: {exc}")

    return captions



device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
@torch.no_grad()
def florence_caption_dataset(images, captions):

    #workaround for unnecessary flash_attn requirement
    from unittest.mock import patch
    from transformers.dynamic_module_utils import get_imports
    from transformers import AutoProcessor, AutoModelForCausalLM 

    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        try:
            imports.remove("flash_attn")
        except:
            pass
        return imports
    
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", attn_implementation="sdpa", device_map=device, torch_dtype=torch_dtype,trust_remote_code=True)
            
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True, cache_dir = model_paths.get_path("FLORENCE"))

    for i, image in enumerate(tqdm(images)):
        if captions[i] is None:
            #prompt = random.choice(["<CAPTION>", "<DETAILED_CAPTION>"])
            prompt = "<CAPTION>"
            prompt = "<DETAILED_CAPTION>"
            prompt = "<MORE_DETAILED_CAPTION>"

            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=random.choice([2,3,4])
                )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
            caption = parsed_answer[prompt]
            captions[i] = caption.replace("The image shows a ", "A ")

    model.to('cpu')
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    return captions


@torch.no_grad()
def caption_dataset(
        images: List[Image.Image],
        captions: List[str],
        caption_model: Literal["blip", "gpt4-v", "florence"] = "blip"
    ) -> List[str]:

    # if all captions are already generated, we dont need to do anything:
    if all(captions):
        print(f"All captions loaded from disk, skipping captioning...")
        return captions

    if "blip" in caption_model:
        captions = blip_caption_dataset(images, captions)
    elif "gpt4-v" in caption_model:
        captions = gpt4_v_caption_dataset(images, captions)
    elif "florence" in caption_model:
        captions = florence_caption_dataset(images, captions)
    else:
        print("WARNING: not using any captions!")
        captions = [""] * len(images)

    gc.collect()
    torch.cuda.empty_cache()

    return captions

def load_image_with_orientation(path, mode = "RGB"):
    image = Image.open(path)

    # Try to get the Exif orientation tag (0x0112), if it exists
    try:
        exif_data = image._getexif()
        orientation = exif_data.get(0x0112)
    except (AttributeError, KeyError, IndexError):
        orientation = None

    # Apply the orientation, if it's present
    if orientation:
        if orientation == 2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180)
        elif orientation == 4:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            image = image.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            image = image.rotate(-90)
        elif orientation == 7:
            image = image.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            image = image.rotate(90)

    return image.convert(mode)

def hue_augmentation(image, hue_change_max = 4):
    """
    Apply hue augmentation to the input image.

    :param image: PIL Image object
    :param hue_change: Amount to change the hue (0-360)
    :return: Augmented PIL Image object
    """
    hue_change = random.uniform(1, hue_change_max)
    # Convert the image to HSV color space
    hsv_image = image.convert('HSV')

    # Split into individual channels
    h, s, v = hsv_image.split()

    # Apply the hue change
    h = h.point(lambda i: (i + hue_change) % 256)
    hsv_image = Image.merge('HSV', (h, s, v))
    return hsv_image.convert('RGB')

def color_jitter(image):
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]
    factor_ranges = [[0.9, 1.1], [0.9, 1.25], [0.9, 1.2]]
    for i, enhancer in enumerate(enhancers):
        low, high = factor_ranges[i]
        factor = random.uniform(low, high)
        image = enhancer(image).enhance(factor)
    return image

def random_crop(image, scale=(0.85, 0.95)):
    width, height = image.size
    new_width, new_height = width * random.uniform(*scale), height * random.uniform(*scale)

    left = random.uniform(0, width - new_width)
    top = random.uniform(0, height - new_height)

    return image.crop((left, top, left + new_width, top + new_height))

def gaussian_blur(image, radius = 1.0):
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def augment_image(image):
    image = hue_augmentation(image)
    image = color_jitter(image)
    image = random_crop(image)
    if random.random() < 0.5:
        image = gaussian_blur(image, radius = random.uniform(0.0, 1.0))
    return image

def round_to_nearest_multiple(x, multiple):
    return int(float(multiple) * round(float(x) / float(multiple)))

'''
For Stable Diffusion 1.5, outputs are optimised around 512x512 pixels. Many common fine-tuned versions of SD1.5 are optimised around 768x768. The best resolutions for common aspect ratios are typically:
1:1 (square): 512x512, 768x768
3:2 (landscape): 768x512
2:3 (portrait): 512x768
4:3 (landscape): 768x576
3:4 (portrait): 576x768
16:9 (widescreen): 912x512
9:16 (tall): 512x912

For SDXL, outputs are optimised around 1024x1024 pixels. The best resolutions for common aspect ratios are typically:
stable-diffusion-xl-1024-v0-9 supports generating images at the following dimensions:
1024 x 1024
1152 x 896
896 x 1152
1216 x 832
832 x 1216
1344 x 768
768 x 1344
1536 x 640
640 x 1536

'''

def calculate_new_dimensions(target_size, target_aspect_ratio):
    """
    Calculate the new width and height given a target size and aspect ratio.
    """
    # Calculate the total number of pixels
    n_pixels = target_size ** 2

    # Calculate the new width and height based on the target aspect ratio
    new_width  = (n_pixels * target_aspect_ratio) ** 0.5
    new_height = (n_pixels / new_width)

    # round up/down to the nearest multiple of 64:
    new_width  = round_to_nearest_multiple(new_width, 64)
    new_height = round_to_nearest_multiple(new_height, 64)

    return [new_width, new_height]


def load_and_save_masks_and_captions(
    config,
    concept_mode: str,
    files: Union[str, List[str]],
    output_dir: str = "tmp_out",
    seed: int = 0,
    caption_text: Optional[str] = None,
    mask_target_prompts: Optional[Union[List[str], str]] = None,
    target_size: int = 1024,
    crop_based_on_salience: bool = True,
    use_face_detection_instead: bool = False,
    n_length: int = -1,
    add_lr_flips: bool = False,
    augment_imgs_up_to_n: int = 0,
    use_dataset_captions: bool = True, # load captions from the dataset if they exist
    caption_model: str = "blip"
):
    """
    Loads images from the given files, generates masks for them, and saves the masks and captions and upscale images
    to output dir. If mask_target_prompts is given, it will generate kinda-segmentation-masks for the prompts and save them as well.
    """
    os.makedirs(output_dir, exist_ok=True)

    # load images
    if isinstance(files, str):
        if os.path.isdir(files):
            files = (
                _find_files("*.png", files)
                + _find_files("*.jpg", files)
                + _find_files("*.jpeg", files)
            )

        if len(files) == 0:
            raise Exception(
                f"No images were found... Are you sure you provided a valid dataset?"
            )
        if n_length == -1:
            n_length = len(files)
        files = sorted(files)[:n_length]
        
    images, captions, img_paths = [], [], []
    for file in files:
        images.append(load_image_with_orientation(file))
        img_paths.append(file)
        caption_file = os.path.splitext(file)[0] + ".txt"
        if os.path.exists(caption_file) and use_dataset_captions:
            with open(caption_file, "r") as f:
                captions.append(f.read())
        else:
            captions.append(None)

    # Compute average aspect ratio of images:
    aspect_ratios = [image.size[0] / image.size[1] for image in images]
    avg_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    print(f"Average aspect ratio of images (width / height): {avg_aspect_ratio:.3f}")
    config.train_img_size = calculate_new_dimensions(target_size, avg_aspect_ratio)
    config.train_aspect_ratio = config.train_img_size[0] / config.train_img_size[1]
    target_size = max(config.train_img_size)
    print(f"New train_img_size: {config.train_img_size}")

    if config.validation_img_size is None:
        config.validation_img_size = [0, 0]
        multiplier = 2.0 if config.sd_model_version == "sdxl" else 1.0
        config.validation_img_size[0] = config.train_img_size[0] * multiplier
        config.validation_img_size[1] = config.train_img_size[1] * multiplier
    elif isinstance(config.validation_img_size, int):
        n_pixels = config.validation_img_size ** 2
        config.validation_img_size = [0, 0]
        config.validation_img_size[0] = (n_pixels * config.train_aspect_ratio) ** 0.5
        config.validation_img_size[1] = (n_pixels / config.validation_img_size[0])
    
    config.validation_img_size[0]  = round_to_nearest_multiple(config.validation_img_size[0], 64)
    config.validation_img_size[1] = round_to_nearest_multiple(config.validation_img_size[1], 64)
    print(f"Validation_img_size was set to: {config.validation_img_size}")

    n_training_imgs = len(images)
    n_captions      = len([c for c in captions if c is not None])
    print(f"Loaded {n_training_imgs} images, {n_captions} of which have captions.")

    if len(images) < 50: # upscale images that are smaller than target_size:
        print("upscaling imgs..")
        upscale_margin = 0.75
        images = swin_ir_sr(images, target_size=(int(config.train_img_size[0]*upscale_margin), int(config.train_img_size[0]*upscale_margin)))

    if add_lr_flips and len(images) < MAX_GPT_PROMPTS:
        print(f"Adding LR flips... (doubling the number of images from {n_training_imgs} to {n_training_imgs*2})")
        images   = images + [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]
        captions = captions + captions


    print(f"Generating {len(images)} captions using {caption_model} in {concept_mode} mode...")
    captions = caption_dataset(images, captions, caption_model = caption_model)

    # Save captions back to disk:
    for i, img_path in enumerate(img_paths):
        caption_path = os.path.splitext(img_path)[0] + ".txt"
        with open(caption_path, "w") as f:
            f.write(captions[i])

    # It's nice if we can achieve the gpt pass, so if we're not losing too much, cut-off the n_images to just match what we're allowed to give to gpt:
    if (len(images) > MAX_GPT_PROMPTS) and (len(images) < MAX_GPT_PROMPTS*1.33):
        images = images[:MAX_GPT_PROMPTS-1]
        captions = captions[:MAX_GPT_PROMPTS-1]

    # Cleanup prompts using chatgpt:
    captions = [fix_prompt(caption) for caption in captions]
    trigger_text = ""
    gpt_concept_description = None
    if not config.disable_ti:
        captions, trigger_text, gpt_concept_description = post_process_captions(captions, caption_text, concept_mode, seed, skip_gpt_cleanup=config.skip_gpt_cleanup)
    
    if config.prompt_modifier:
        print(config.prompt_modifier)
        captions = [config.prompt_modifier.format(caption) for caption in captions]
    
    aug_imgs, aug_caps = [],[]
    # if we still have a small amount of imgs, do some basic augmentation:
    while len(images) + len(aug_imgs) < augment_imgs_up_to_n: 
        print(f"Adding augmented version of each training img...")
        aug_imgs.extend([augment_image(image) for image in images])
        aug_caps.extend(captions)

    images.extend(aug_imgs)
    captions.extend(aug_caps)
    

    if (gpt_concept_description is not None) and ((mask_target_prompts is None) or (mask_target_prompts == "")):
        print(f"Using GPT concept name as CLIP-segmentation prompt: {gpt_concept_description}")
        mask_target_prompts = gpt_concept_description

    if mask_target_prompts is None or config.concept_mode == "style":
        mask_target_prompts = ""
        temp = 999
    else:
        temp = config.clipseg_temperature

    print(f"Generating {len(images)} masks...")
    # Make sure we have a bias for the background pixels to never 100% ignore them
    background_bias = 0.05
    if not use_face_detection_instead:
        seg_masks = clipseg_mask_generator(
            images=images, target_prompts=mask_target_prompts, temp=temp, bias=background_bias
        )
    else:
        mask_target_prompts = "FACE detection was used"
        if add_lr_flips:
            print("WARNING you are applying face detection while also doing left-right flips, this might not be what you intended?")
        seg_masks = face_mask_google_mediapipe(images=images, bias=background_bias*255)

    print("Masks generated! Cropping images to center of mass...")
    # find the center of mass of the mask
    if crop_based_on_salience:
        coms = [_center_of_mass(mask) for mask in seg_masks]
    else:
        coms = [(image.size[0] / 2, image.size[1] / 2) for image in images]
        
    # based on the center of mass, crop the image to a square
    print("Cropping and resizing images...")
    images = [
        _crop_to_aspect_ratio(image, com, target_aspect_ratio = config.train_aspect_ratio,  # width / height
            resize_to = target_size)
        for image, com in zip(images, coms)
    ]

    seg_masks = [
        _crop_to_aspect_ratio(mask, com, target_aspect_ratio = config.train_aspect_ratio,  # width / height
            resize_to = target_size)
        for mask, com in zip(seg_masks, coms)
    ]

    print("Expanding masks...")
    if use_face_detection_instead:
        dilation_radius = -0.02 * (config.train_img_size[0] + config.train_img_size[0]) / 2
        blur_radius     = 0.02 * (config.train_img_size[0] + config.train_img_size[0]) / 2
    else:
        dilation_radius = 0.0
        blur_radius     = 0.005 * (config.train_img_size[0] + config.train_img_size[0]) / 2

    for i in range(len(seg_masks)):
        seg_masks[i] = grow_mask(seg_masks[i], dilation_radius=dilation_radius, blur_radius=blur_radius)
    print("Done!")
    
    data = []
    # clean TEMP_OUT_DIR first
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    os.makedirs(output_dir, exist_ok=True)
    
    if config.disable_ti:
        print('------------------ WARNING -------------------')
        print("Removing 'TOK, ' from captions...")
        print("This will completely disable textual_inversion!!")
        print('------------------ WARNING -------------------')
        if gpt_concept_description:
            replace_str = gpt_concept_description
        else:
            replace_str = ""
        captions = [caption.replace("TOK, ", replace_str + ", ") for caption in captions]
        captions = [caption.replace("TOK", replace_str) for caption in captions]
    else:
        captions = ["TOK, " + caption if "TOK" not in caption else caption for caption in captions]

    # iterate through the images, masks, and captions and add a row to the dataframe for each
    print("Saving final training dataset...")
    for idx, (image, mask, caption) in enumerate(zip(images, seg_masks, captions)):

        image_name = f"{idx}.src.jpg"
        mask_file = f"{idx}.mask.jpg"
        # save the image and mask files
        image.save(os.path.join(output_dir, image_name), quality=95)
        mask.save(os.path.join(output_dir, mask_file), quality=95)

        # add a new row to the dataframe with the file names and caption
        data.append(
            {"image_path": image_name, "mask_path": mask_file, "caption": caption},
        )

    df = pd.DataFrame(columns=["image_path", "mask_path", "caption"], data=data)
    # save the dataframe to a CSV file
    df.to_csv(os.path.join(output_dir, "captions.csv"), index=False)
    print("---> Training data 100% ready to go!")

    # do a final prompt cleaning pass to fix weird commas and spaces:
    captions = [fix_prompt(caption) for caption in captions]

    # Update the training attributes with some info from the pre-processing:
    config.training_attributes["n_training_imgs"] = n_training_imgs
    config.training_attributes["trigger_text"] = trigger_text
    config.training_attributes["segmentation_prompt"] = mask_target_prompts
    config.training_attributes["gpt_description"] = gpt_concept_description
    config.training_attributes["captions"] = captions

    return config


from PIL import Image, ImageFilter, ImageChops

def grow_mask(mask, dilation_radius=5, blur_radius=3):
    dilation_radius = int(dilation_radius)
    blur_radius     = int(blur_radius)

    # Load the image
    mask = mask.convert('L')  # Ensure it's in grayscale

    # Get the minimum pixel value in the mask:
    min_mask_value = int(np.min(np.array(mask)))

    # Dilate the mask
    if dilation_radius > 0:
        mask = mask.filter(ImageFilter.MinFilter(dilation_radius * 2 + 1))

    # Apply Gaussian blur to the dilated mask
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    # Clip the mask pixel values to make sure they dont go below the minimum value
    mask = ImageChops.lighter(mask, Image.new('L', mask.size, min_mask_value))

    return mask


def _center_of_mass(mask: Image.Image):
    """
    Returns the center of mass of the mask
    """
    x, y = np.meshgrid(np.arange(mask.size[0]), np.arange(mask.size[1]))
    mask_np = np.array(mask) + 0.01
    x_ = x * mask_np
    y_ = y * mask_np

    x = np.sum(x_) / np.sum(mask_np)
    y = np.sum(y_) / np.sum(mask_np)

    return x, y

def _crop_to_aspect_ratio(
    image: Image.Image,
    com: List[Tuple[int, int]],
    target_aspect_ratio: float = 1.0,  # width / height
    resize_to: Optional[int] = None
):
    """
    Crops the image to the specified aspect ratio around the center of mass of the mask.
    """
    cx, cy = com
    width, height = image.size

    if target_aspect_ratio > 1:  # Wider than tall
        new_width = int(min(width, height * target_aspect_ratio))
        new_height = int(new_width / target_aspect_ratio)
    else:  # Taller than wide or square
        new_height = int(min(height, width / target_aspect_ratio))
        new_width = int(new_height * target_aspect_ratio)

    left = int(max(cx - new_width / 2, 0))
    right = int(min(left + new_width, width))
    top = int(max(cy - new_height / 2, 0))
    bottom = int(min(top + new_height, height))

    # Adjust if the crop goes beyond the image boundaries
    if right > width:
        overshoot = right - width
        right = width
        left = max(0, left - overshoot)  # Adjust left as well symmetrically

    if bottom > height:
        overshoot = bottom - height
        bottom = height
        top = max(0, top - overshoot)  # Adjust top as well symmetrically

    image = image.crop((left, top, right, bottom))

    if resize_to:
        if target_aspect_ratio > 1:
            resize_height = int(resize_to / target_aspect_ratio)
            image = image.resize((resize_to, resize_height), Image.Resampling.LANCZOS)
        else:
            resize_width = int(resize_to * target_aspect_ratio)
            image = image.resize((resize_width, resize_to), Image.Resampling.LANCZOS)

    return image




def face_mask_google_mediapipe(
    images: List[Image.Image], blur_amount: float = 0.0, bias: float = 10.0
) -> List[Image.Image]:
    """
    Returns a list of images with masks on the face parts.
    """
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.1
    )
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.1
    )

    masks = []
    for image in tqdm(images):
        image_np = np.array(image)

        # Perform face detection
        results_detection = face_detection.process(image_np)
        ih, iw, _ = image_np.shape
        if results_detection.detections:
            for detection in results_detection.detections:
                bboxC = detection.location_data.relative_bounding_box

                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                # make sure bbox is within image
                bbox = (
                    max(0, bbox[0]),
                    max(0, bbox[1]),
                    min(iw - bbox[0], bbox[2]),
                    min(ih - bbox[1], bbox[3]),
                )

                # Extract face landmarks
                face_landmarks = face_mesh.process(
                    image_np[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
                ).multi_face_landmarks

                # https://github.com/google/mediapipe/issues/1615
                # This was def helpful
                indexes = [
                    10,
                    338,
                    297,
                    332,
                    284,
                    251,
                    389,
                    356,
                    454,
                    323,
                    361,
                    288,
                    397,
                    365,
                    379,
                    378,
                    400,
                    377,
                    152,
                    148,
                    176,
                    149,
                    150,
                    136,
                    172,
                    58,
                    132,
                    93,
                    234,
                    127,
                    162,
                    21,
                    54,
                    103,
                    67,
                    109,
                ]

                if face_landmarks:
                    mask = Image.new("L", (iw, ih), 0)
                    mask_np = np.array(mask)

                    for face_landmark in face_landmarks:
                        face_landmark = [face_landmark.landmark[idx] for idx in indexes]
                        landmark_points = [
                            (int(l.x * bbox[2]) + bbox[0], int(l.y * bbox[3]) + bbox[1])
                            for l in face_landmark
                        ]
                        mask_np = cv2.fillPoly(
                            mask_np, [np.array(landmark_points)], 255
                        )

                    mask = Image.fromarray(mask_np)

                    # Apply blur to the mask
                    if blur_amount > 0:
                        mask = mask.filter(ImageFilter.GaussianBlur(blur_amount))

                    # Apply bias to the mask
                    if bias > 0:
                        mask = np.array(mask)
                        mask = mask + bias * np.ones(mask.shape, dtype=mask.dtype)
                        mask = np.clip(mask, 0, 255)
                        mask = Image.fromarray(mask)

                    # Convert mask to 'L' mode (grayscale) before saving
                    mask = mask.convert("L")
                    masks.append(mask)
                else:
                    # If face landmarks are not available, add a black mask of the same size as the image
                    masks.append(Image.new("L", (iw, ih), 0))

        else:
            print("No face detected, adding full mask")
            # If no face is detected, add a black mask of the same size as the image
            masks.append(Image.new("L", (iw, ih), 0))

    return masks
