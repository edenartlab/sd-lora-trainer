# Have SwinIR upsample
# Have BLIP auto caption
# Have CLIPSeg auto mask concept

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

MODEL_PATH = "./cache"
MAX_GPT_PROMPTS = 40

import re
def fix_prompt(prompt: str):
    # Remove extra commas and spaces, and fix space before punctuation
    prompt = re.sub(r"\s+", " ", prompt)  # Replace multiple spaces with a single space
    prompt = re.sub(r",,", ",", prompt)  # Replace double commas with a single comma
    prompt = re.sub(r"\s?,\s?", ", ", prompt)  # Fix spaces around commas
    prompt = re.sub(r"\s?\.\s?", ". ", prompt)  # Fix spaces around periods
    return prompt.strip()  # Remove leading and trailing whitespace


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
    temp: float,
    left_right_flip_augmentation: bool = False,
    augment_imgs_up_to_n: int = 0,
    caption_model: str = "blip",
    seed: int = 0,
) -> Path:

    if os.path.exists(working_directory):
        print(f"working_directory {working_directory} already existed.. deleting and recreating!")
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

    n_training_imgs, trigger_text, segmentation_prompt, captions = load_and_save_masks_and_captions(
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
        temp=temp,
        add_lr_flips = left_right_flip_augmentation,
        augment_imgs_up_to_n = augment_imgs_up_to_n,
        caption_model = caption_model
    )

    return Path(TEMP_OUT_DIR), n_training_imgs, trigger_text, segmentation_prompt, captions


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
        model_id, cache_dir=MODEL_PATH
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
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    bias: float = 0.01,
    temp: float = 1.0,
    **kwargs,
) -> List[Image.Image]:
    """
    Returns a greyscale mask for each image, where the mask is the probability of the target prompt being present in the image
    """

    if isinstance(target_prompts, str):
        print(
            f'Using "{target_prompts}" as CLIP-segmentation prompt for all images.'
        )
        target_prompts = [target_prompts] * len(images)

    if any(target_prompts):
        processor = CLIPSegProcessor.from_pretrained(model_id, cache_dir=MODEL_PATH)
        model = CLIPSegForImageSegmentation.from_pretrained(
            model_id, cache_dir=MODEL_PATH
        ).to(device)

    masks = []

    for image, prompt in tqdm(zip(images, target_prompts)):
        original_size = image.size

        if prompt != "":
            inputs = processor(
                text=[prompt, ""],
                images=[image] * 2,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)

            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits / temp, dim=0)[0]
            probs = (probs + bias).clamp_(0, 1)
            probs = 255 * probs / probs.max()

            # make mask greyscale
            mask = Image.fromarray(probs.cpu().numpy()).convert("L")

            # resize mask to original size
            mask = mask.resize(original_size)
        else:
            mask = Image.new("L", original_size, 255)

        masks.append(mask)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return masks



def cleanup_prompts_with_chatgpt(
    prompts, 
    concept_mode,    # face / object / style
    seed,  # seed for chatgpt reproducibility
    verbose = True):

    if concept_mode == "object_injection":
        chat_gpt_prompt_1 = """
            I have a set of images, each containing the same concept / figure. I have the following (poor) descriptions for each image:
            """
        
        chat_gpt_prompt_2 = """
            I want you to:
            1. Find a good, short name/description of the single central concept that's in all the images. This [Concept Name]  might eg already be present in the descriptions above, pick the most obvious name or words that would fit in all descriptions.
            2. Insert the text "TOK, [Concept Name]" into all the descriptions above by rephrasing them where needed to naturally contain the text TOK, [Concept Name] while keeping as much of the description as possible.

            Reply by first stating the "Concept Name:", followed by an enumerated list (using "-") of all the revised "Descriptions:".
            """

    if concept_mode == "object":
        chat_gpt_prompt_1 = """
        Analyze a set of (poor) image descriptions each featuring the same concept, figure or thing.
        Tasks:
        1. Deduce a concise, fitting name for the concept that is visually descriptive (Concept Name).
        2. Substitute the concept in each description with the placeholder "TOK", rearranging or adjusting the text where needed. Hallucinate TOK into the description if necessary (but dont mention when doing so, simply provide the final description)!
        3. Streamline each description to its core elements, ensuring clarity and mandatory inclusion of the placeholder string "TOK".
        The descriptions are:
        """
        
        chat_gpt_prompt_2 = """
        Respond with the chosen "Concept Name:" followed by a list (using "-") of all the revised descriptions, each mentioning "TOK".
        """

    elif concept_mode == "face":
        chat_gpt_prompt_1 = """
        Analyze a set of (poor) image descriptions, each featuring a person named TOK.
        Tasks:
        1. Rewrite each description, ensuring it refers only to a single person or character.
        2. Integrate "a photo of TOK" naturally into each description, rearranging or adjusting where needed.
        3. Streamline each description to its core elements, ensuring clarity and mandatory inclusion of "TOK".
        The descriptions are:
        """
        
        chat_gpt_prompt_2 = """
        Respond with "Concept Name: TOK" followed by a list (using "-") of all the revised descriptions, each mentioning "a photo of TOK".
        """

    elif concept_mode == "style":
        chat_gpt_prompt_1 = """
        Analyze a set of (poor) image descriptions, each featuring the same style named TOK.
        Tasks:
        1. Rewrite each description to focus solely on the TOK style.
        2. Integrate "in the style of TOK" naturally into each description, typically at the beginning.
        3. Summarize each description to its core elements, ensuring clarity and mandatory inclusion of "TOK".
        The descriptions are:
        """
        
        chat_gpt_prompt_2 = """
        Respond with "Style Name: TOK" followed by a list (using "-") of all the revised descriptions, each mentioning "in the style of TOK".
        """

    final_chatgpt_prompt = chat_gpt_prompt_1 + "\n- " + "\n- ".join(prompts) + "\n\n" + chat_gpt_prompt_2
    print("Final chatgpt prompt:")
    print(final_chatgpt_prompt)
    print("--------------------------")
    print(f"Calling chatgpt with seed {seed}...")

    response = client.chat.completions.create(
            model="gpt-4-1106-preview",
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
        if line.startswith("-"):
            prompts.append(line[2:])

    gpt_concept_name = extract_gpt_concept_name(gpt_completion, concept_mode)

    trigger_text = "TOK, " + gpt_concept_name if concept_mode == 'object_injection' else "TOK"

    if concept_mode == 'style':
        trigger_text = ", in the style of TOK"
        gpt_concept_name = ""  # Disables segmentation for style (use full img)

    return prompts, gpt_concept_name, trigger_text

def extract_gpt_concept_name(gpt_completion, concept_mode):
    """
    Extracts the concept name from the GPT completion based on the concept mode.
    """
    concept_name = ""
    prefix = ""
    if concept_mode in ['face', 'style']:
        concept_name = concept_mode
        prefix = "Style Name:" if concept_mode == 'style' else ""
    elif concept_mode in ['object_injection', 'object']:
        prefix = "Concept Name:"
        concept_mode = 'object_injection'

    if prefix:
        for line in gpt_completion.split("\n"):
            if line.startswith(prefix):
                concept_name = line[len(prefix):].strip()
                break

    return concept_name


def post_process_captions(captions, text, concept_mode, job_seed):

    text = text.strip()
    print(f"Input captioning text: {text}")

    if len(captions) > 3 and len(captions) < MAX_GPT_PROMPTS and not text and client:
        retry_count = 0
        while retry_count < 10:
            try:
                gpt_captions, gpt_concept_name, trigger_text = cleanup_prompts_with_chatgpt(captions, concept_mode, job_seed + retry_count)
                n_toks = sum("TOK" in caption for caption in gpt_captions)
                
                if n_toks > int(0.8 * len(captions)) and (len(gpt_captions) == len(captions)):
                    # Ensure every caption contains "TOK"
                    gpt_captions = ["TOK, " + caption if "TOK" not in caption else caption for caption in gpt_captions]
                    captions = gpt_captions
                    break
                else:
                    retry_count += 1
            except Exception as e:
                retry_count += 1
                print(f"An error occurred after try {retry_count}: {e}")
                time.sleep(1)
        else:
            gpt_concept_name, trigger_text = None, "TOK"
    else:
        # simple concat of trigger text with rest of prompt:
        if len(text) == 0:
            print("WARNING: no captioning text was given and we're not doing chatgpt cleanup...")
            print("Concept mode: ", concept_mode)
            if concept_mode == "style":
                trigger_text = "in the style of TOK, "
                captions = [trigger_text + caption for caption in captions]
            else:
                trigger_text = "a photo of TOK, "
                captions = [trigger_text + caption for caption in captions]
        else:
            trigger_text = text
            captions = [trigger_text + ", " + caption for caption in captions]

        gpt_concept_name = None

    captions = [fix_prompt(caption) for caption in captions]
    return captions, trigger_text, gpt_concept_name


def blip_caption_dataset(
        images: List[Image.Image],
        captions: List[str],
        model_id: Literal[
            "Salesforce/blip-image-captioning-large",
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip2-opt-2.7b",
        ] = "Salesforce/blip-image-captioning-large"
        ):

    print(f"Using model {model_id} for image captioning...")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "blip2" in model_id:
        processor = Blip2Processor.from_pretrained(model_id, cache_dir=MODEL_PATH)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, cache_dir=MODEL_PATH, torch_dtype=torch.float16
        ).to(device)
    else:
        processor = BlipProcessor.from_pretrained(model_id, cache_dir=MODEL_PATH)
        model = BlipForConditionalGeneration.from_pretrained(
            model_id, cache_dir=MODEL_PATH, torch_dtype=torch.float16
        ).to(device)

    for i, image in enumerate(tqdm(images)):
        if captions[i] is None:
            inputs = processor(image, return_tensors="pt").to(device, torch.float16)
            out = model.generate(**inputs, max_length=100, do_sample=True, top_k=40, temperature=0.65)
            captions[i] = processor.decode(out[0], skip_special_tokens=True)

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

    prompt = "Concisely describe this image without assumptions with at most 30 words. Dont start with statements like 'The image features...', just describe what you see."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    def fetch_caption(index, img):
        base64_image = prep_img_for_gpt_api(img,  max_size=(512, 512))

        payload = {
            "model": "gpt-4-vision-preview",
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
        return index, response.json()["choices"][0]["message"]["content"]

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



@torch.no_grad()
def caption_dataset(
        images: List[Image.Image],
        captions: List[str],
        caption_model: Literal[str] = "blip"
    ) -> List[str]:

    if "blip" in caption_model:
        captions = blip_caption_dataset(images, captions)
    elif "gpt4-v" in caption_model:
        captions = gpt4_v_caption_dataset(images, captions)

    return captions



def _crop_to_square(
    image: Image.Image, com: List[Tuple[int, int]], resize_to: Optional[int] = None
):
    cx, cy = com
    width, height = image.size
    if width > height:
        left_possible = max(cx - height / 2, 0)
        left = min(left_possible, width - height)
        right = left + height
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top_possible = max(cy - width / 2, 0)
        top = min(top_possible, height - width)
        bottom = top + width

    image = image.crop((left, top, right, bottom))

    if resize_to:
        image = image.resize((resize_to, resize_to), Image.Resampling.LANCZOS)

    return image


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

def gaussian_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=1))

def augment_image(image):
    image = hue_augmentation(image)
    image = color_jitter(image)
    image = random_crop(image)
    if random.random() < 0.5:
        image = gaussian_blur(image)
    return image



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
    temp: float = 1.0,
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
            print("Scanning directory for images...")
            files = (
                _find_files("*.png", files)
                + _find_files("*.jpg", files)
                + _find_files("*.jpeg", files)
            )

        if len(files) == 0:
            raise Exception(
                f"No files found in {files}. Either {files} is not a directory or it does not contain any .png or .jpg/jpeg files."
            )
        if n_length == -1:
            n_length = len(files)
        files = sorted(files)[:n_length]
        
    images, captions = [], []
    for file in files:
        images.append(load_image_with_orientation(file))
        caption_file = os.path.splitext(file)[0] + ".txt"
        if os.path.exists(caption_file) and use_dataset_captions:
            with open(caption_file, "r") as f:
                captions.append(f.read())
        else:
            captions.append(None)

    n_training_imgs = len(images)
    n_captions      = len([c for c in captions if c is not None])
    print(f"Loaded {n_training_imgs} images, {n_captions} of which have captions.")

    if len(images) < 50: # upscale images that are smaller than target_size:
        print("upscaling imgs..")
        upscale_margin = 0.75
        images = swin_ir_sr(images, target_size=(int(target_size*upscale_margin), int(target_size*upscale_margin)))

    if add_lr_flips and len(images) < 40:
        print(f"Adding LR flips... (doubling the number of images from {n_training_imgs} to {n_training_imgs*2})")
        images   = images + [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]
        captions = captions + captions

    # It's nice if we can achieve the gpt pass, so if we're not losing too much, cut-off the n_images to just match what we're allowed to give to gpt:
    if (len(images) > MAX_GPT_PROMPTS) and (len(images) < MAX_GPT_PROMPTS*1.33):
        images = images[:MAX_GPT_PROMPTS-1]
        captions = captions[:MAX_GPT_PROMPTS-1]

    # Use BLIP for autocaptioning:
    print(f"Generating {len(images)} captions using mode: {concept_mode}...")
    captions = caption_dataset(images, captions, caption_model = caption_model)

    # Cleanup prompts using chatgpt:
    captions, trigger_text, gpt_concept_name = post_process_captions(captions, caption_text, concept_mode, seed)

    aug_imgs, aug_caps = [],[]
    while len(images) + len(aug_imgs) < augment_imgs_up_to_n: # if we still have a very small amount of imgs, do some basic augmentation:
        print(f"Adding augmented version of each training img...")
        aug_imgs.extend([augment_image(image) for image in images])
        aug_caps.extend(captions)

    images.extend(aug_imgs)
    captions.extend(aug_caps)
    
    if (gpt_concept_name is not None) and ((mask_target_prompts is None) or (mask_target_prompts == "")):
        print(f"Using GPT concept name as CLIP-segmentation prompt: {gpt_concept_name}")
        mask_target_prompts = gpt_concept_name

    if mask_target_prompts is None:
        print("Disabling CLIP-segmentation")
        mask_target_prompts = ""
        temp = 999

    print(f"Generating {len(images)} masks...")
    if not use_face_detection_instead:
        seg_masks = clipseg_mask_generator(
            images=images, target_prompts=mask_target_prompts, temp=temp
        )
    else:
        mask_target_prompts = "FACE detection was used"
        if add_lr_flips:
            print("WARNING you are applying face detection while also doing left-right flips, this might not be what you intended?")
        seg_masks = face_mask_google_mediapipe(images=images)

    print("Masks generated! Cropping images to center of mass...")
    # find the center of mass of the mask
    if crop_based_on_salience:
        coms = [_center_of_mass(mask) for mask in seg_masks]
    else:
        coms = [(image.size[0] / 2, image.size[1] / 2) for image in images]
        
    # based on the center of mass, crop the image to a square
    print("Cropping squares...")
    images = [
        _crop_to_square(image, com, resize_to=None) 
        for image, com in zip(images, coms)
    ]

    seg_masks = [
        _crop_to_square(mask, com, resize_to=target_size) 
        for mask, com in zip(seg_masks, coms)
    ]

    print("Resizing images to training size...")
    images = [
        image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        for image in images
    ]

    data = []
    # clean TEMP_OUT_DIR first
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    os.makedirs(output_dir, exist_ok=True)

    # Make sure we've correctly inserted the TOK into every caption:
    captions = ["TOK, " + caption if "TOK" not in caption else caption for caption in captions]
    for caption in captions:
        print(caption)

    if config.remove_ti_token_from_prompts:
        print('------------ WARNING ------------')
        print("Removing 'TOK, ' from captions...")
        print("This will completely break textual_inversion!!")
        print('------------ WARNING ------------')
        captions = [caption.replace("TOK, ", "") for caption in captions]

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

    return n_training_imgs, trigger_text, mask_target_prompts, captions



def face_mask_google_mediapipe(
    images: List[Image.Image], blur_amount: float = 0.0, bias: float = 50.0
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

                print(bbox)

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
                    masks.append(Image.new("L", (iw, ih), 255))

        else:
            print("No face detected, adding full mask")
            # If no face is detected, add a white mask of the same size as the image
            masks.append(Image.new("L", (iw, ih), 255))

    return masks
