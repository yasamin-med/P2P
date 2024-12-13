#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import matplotlib.pyplot as plt
import argparse
import logging
import math
import os
import time
import random
import shutil
import warnings
from pathlib import Path
import torch.nn as nn
import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
import torchvision.models as models
import json
# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
import diffusers
import csv
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from torchvision.utils import save_image
import copy
from transformers.tokenization_utils import AddedToken
if is_wandb_available():
    import wandb
from diffusers.utils.torch_utils import randn_tensor
if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__)


def save_model_card(repo_id: str, images=None, base_model=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- textual_inversion
inference: true
---
    """
    model_card = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.safety_checker = None
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()
    return images


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    # mine 
    #default=100 for repeat
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    parser.add_argument(
        "--saving_image_dir",
        type=str,
        default="generated_example",
        help="directory for images",
    )    
    parser.add_argument(
        "--class_name",
        type=str,
        default="benign",
        help="name of class like benign, normal , malignant",
    )    
    parser.add_argument("--classifier_name", type=str , default= 'resnet34' , help ='name of classifier' ) 
    parser.add_argument("--classifier_path", type=str , default="/scratch/st-ilker-1/yasamin/Breast/result_split_1/result_original"  , help ='path to baselines checkpoints')
    parser.add_argument("--num_class", type=int , default=3  , help ='number of classes')
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        required=True,
        help="path to saved csv path",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]
Breast_ultrasound_template = ["ultrasound image of {} tumor in breast"]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.templates = Breast_ultrasound_template

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.image_names = [file_path for file_path in os.listdir(self.data_root)]
 
        
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats
            # mine
            #self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        #self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)
        # mine
        # text = self.prompts[i % self.num_images]
        example["name"] = self.image_names[i % self.num_images]

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        
        # mine
        #example["label"] = self.labels[i % self.num_images]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        #image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.placeholder_token]

    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    # mine
    #token_ids = tokenizer.encode(args.initializer_token, add_special_tokens= False)
    tokenizer.add_tokens(AddedToken('benign', single_word=True))
    tokenizer.add_tokens(AddedToken('malignant', single_word=True))
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens= False)
    print(token_ids)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    
    text_encoder.text_model.encoder.requires_grad_(True)
    text_encoder.text_model.final_layer_norm.requires_grad_(True)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(True)
    text_encoder.get_input_embeddings().requires_grad_(True)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        list(text_encoder.parameters()),
        #text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # optimizer = torch.optim.AdamW(
    #     text_encoder.text_model.encoder().parameters(),  # only optimize the embeddings
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )    
    # mine
    # optimizer = torch.optim.LBFGS(
    #     text_encoder.get_input_embeddings().parameters(),
    #     lr=args.learning_rate,
    #     max_iter=20, 
        
    # )

    # Dataset and DataLoaders creation:
    
    # train_dataset = TextualInversionDataset(
    #     data_root=args.train_data_dir,
    #     tokenizer=tokenizer,
    #     size=args.resolution,
    #     placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
    #     repeats=args.repeats,
    #     learnable_property=args.learnable_property,
    #     center_crop=args.center_crop,
    #     set="train",
    # )
    # mine
    train_dataset = TextualInversionDataset(
    data_root=args.train_data_dir,
    tokenizer=tokenizer,
    size=args.resolution,
    placeholder_token=((tokenizer.convert_ids_to_tokens(initializer_token_id))),
    repeats=1, #args.repeats
    learnable_property=args.learnable_property,
    center_crop=args.center_crop,
    set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    text_encoder.train()
    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    # mine
    # Get the vocabulary
    embeding_clip = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    vocabulary = tokenizer.get_vocab()
    

    tokenizer_orig = copy.deepcopy(tokenizer)
    if args.classifier_name == 'squeezenet1.1':
        net = models.squeezenet1_1(pretrained=False)
        net.classifier[1] = nn.Conv2d(512, args.num_class, kernel_size=(1, 1), stride=(1, 1))
        net.load_state_dict(torch.load(args.classifier_path + "/squeezenet1.1/squeezenet1.1.pth") )
    elif args.classifier_name == 'densenet121':
        net = models.densenet121(pretrained=False)
        net.classifier = nn.Linear(net.classifier.in_features,  args.num_class)
        net.load_state_dict(torch.load(args.classifier_path + "/densenet121/densenet121.pth") )
    elif args.classifier_name == 'resnet34':
        net = models.resnet34(pretrained=False)
        net.fc = nn.Linear(net.fc.in_features,  args.num_class)
        net.load_state_dict(torch.load(args.classifier_path  + "/resnet34/resnet34.pth") )
    net.eval()
    net.to(accelerator.device)
    # resnet34 = models.resnet34(pretrained=False)
    # resnet34.fc = nn.Linear(resnet34.fc.in_features, 3)
    # resnet34.load_state_dict(torch.load("/scratch/st-ilker-1/yasamin/Breast/result_split_2/result_original/resnet34/resnet34.pth") )
    # resnet34.eval()
    # resnet34.to(accelerator.device)
    

    
    
    # all_embeddings = []
    # # embeding
    # for token_id in range(len(vocabulary)):
    #     token = tokenizer.convert_ids_to_tokens(token_id)
    #     token_input_ids = tokenizer(token, return_tensors="pt").input_ids[0]
    #     with torch.no_grad():
    #         token_features = embeding_clip(token_input_ids[1])
    #     token_embedding = token_features # Take the embedding of the [CLS] token
    #     all_embeddings.append(token_embedding.numpy()) 

    # all_embeddings = np.vstack(all_embeddings)
    alpha_bar = noise_scheduler.alphas_cumprod.to(accelerator.device)
    alpha = noise_scheduler.alphas.to(accelerator.device)
    print("thresholding config" , noise_scheduler.config.thresholding)
    print("clip sample config" , noise_scheduler.config.clip_sample)
    
   
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    count = 0
    row_list = []
    name_save_list = []
    image_dict = {}
  
    criterion = nn.CrossEntropyLoss()
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Normalize(mean=mean, std=std),  # Example normalization values for ResNet
    ])
    coeff_loss_list = torch.linspace(0 , 1 , noise_scheduler.config.num_train_timesteps).to(accelerator.device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    pil_to_tensor = transforms.PILToTensor()
    
    if args.class_name == "benign":
        label = torch.tensor([0], device=accelerator.device)
    elif args.class_name == "malignant":
        label = torch.tensor([1], device=accelerator.device)
    else:
        label = torch.tensor([2], device=accelerator.device)
    all_images_saved = False
    # while not all_images_saved:
    #     if len(name_save_list) == len(train_dataset):
    #         all_images_saved = True
    

# Define the CSV file path
    csv_file_path = args.csv_path
    csv_file_dir = os.path.dirname(csv_file_path)
    if not os.path.exists(csv_file_dir):
        os.makedirs(csv_file_dir)
    

# If it's the first iteration, create the file and write the header
    with open(csv_file_path, 'a', newline='') as csvfile:
        # Create a csv writer object
        csvwriter = csv.writer(csvfile)
        
        # Check if file is empty (i.e., first time writing to it)
        if csvfile.tell() == 0:
            # Write the header once
            csvwriter.writerow(["name", "time_step", "mse" , "class attack"])
            
    for step, batch in enumerate(train_dataloader):
        text_encoder.train()
        print("name" , batch["name"])

        attack_flag = 0
        T_reverse = 50
        time_start = time.time()
        for epoch in range(first_epoch, T_reverse * 10): #args.num_train_epochs
            with accelerator.accumulate(text_encoder):
                
                if not os.path.exists(args.saving_image_dir):
                    os.makedirs(args.saving_image_dir)
                
                landa = 1
               # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor
                #label = batch["label"].long()
                name = batch["name"]
                


                # Sam ple noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                #timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                
                #mine
                # before 100
                timesteps = torch.randint(0, T_reverse, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) #x_hat_t

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                embeding_clip_new =  accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

                timesteps_cpu = timesteps
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                alpha_bar_t = alpha_bar[timesteps_cpu]
                alpha_t =  alpha[timesteps_cpu]
                
                #print("alpha_bar_t" ,alpha_bar_t)
                #print("alpha_t" , alpha_t)
                #print("model_pred" , model_pred)
                
                # noisy_latents = x_t_ , model_pred = predicted noise
                #print("noisy latent ",noisy_latents)
                latent_0 =  torch.sqrt(1. / alpha_bar_t) * (noisy_latents - torch.sqrt( (1. - alpha_bar_t )) * model_pred) 
                # torch.save(noisy_latents, 'noisy_latents.pt')
                # torch.save(model_pred, 'model_pred.pt')
                #print("latent 0:", latent_0)
                #print("latent shape: ", latent_0.shape)
                
                #image_adver = vae.decode(latent_0).sample
                image_adver = vae.decode(latent_0 / vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                
                #print("image shape:", image_adver.shape)
                
                # save_image(image_adver.cpu(), f"/scratch/st-ilker-1/yasamin/Textual_attack/breast/img_{timesteps.cpu()}.png")
                # detach


                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                #loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # mine

                #print(image_adver)
                #save_image(image_adver, 'img1.png')
                do_denormalize = [True] * image_adver.shape[0]
                #image_adver = image_processor.postprocess(image_adver.detach(), output_type="pil", do_denormalize=do_denormalize)
                image_adver = image_processor.postprocess(image_adver, output_type="pt", do_denormalize=do_denormalize)
                #image_adver_transformed = (image_adver + 1.0) / 2
                
                image_adver_numpy = image_adver.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                if image_adver_numpy.ndim == 3:
                    image_adver_numpy = image_adver_numpy[None, ...]
                image_adver_numpy = (image_adver_numpy * 255).round().astype("uint8")
                if image_adver_numpy.shape[-1] == 1:
                    # special case for grayscale (single channel) images
                    image_adver_pil = [Image.fromarray(image.squeeze(), mode="L") for image in image_adver_numpy]
                else:
                    image_adver_pil = [Image.fromarray(image) for image in image_adver_numpy]

                
                
                #image_adver_pil[0].save(args.saving_image_dir + f"/img_{name[0]}_{epoch}_{timesteps.cpu()}_reconstrcuted.png")
                
                # image_adver_tensor = pil_to_tensor(image_adver[0])
                # image_adver_tensor_cuda = image_adver_tensor.to(accelerator.device).to(dtype=weight_dtype)
                image_adver_transformed = transform(image_adver[0]).unsqueeze(0)
                
                print(image_adver_transformed.shape)


                predicted_class = net(image_adver_transformed)
                
                print("predicted class for attack: ", predicted_class.cpu())
                predicted_class_softmax = F.softmax(predicted_class, dim=1)
                print("predicted classfor attack with softmax: " ,predicted_class_softmax.cpu())
                _, class_max = torch.max(predicted_class , 1 )
                # finding second best class of adversary example
                modified_tensor = predicted_class.clone()

                # Set the maximum values to a very small number (effectively removing them)
                modified_tensor[range(modified_tensor.size(0)), class_max] = float('-inf')

                # Find the second maximum value
                _, class_second_max = torch.max(modified_tensor, 1)
                
                print("time step" , timesteps.cpu())
                print("epoch ", epoch)
                print("class second max" , class_second_max)
                print("class_max" , class_max)
                img_orig = (batch["pixel_values"].to(dtype=weight_dtype) + 1) / 2
                print("shape original: " , img_orig.shape)
                

                
                img_orig = transform(img_orig)
                
                predicted_class_orig = net(img_orig)
                print("predicted class for original: ", predicted_class_orig.cpu())
                predicted_class_orig_softmax = F.softmax(predicted_class_orig , dim=1)
                print("predicted class for original with softmax:" ,predicted_class_orig_softmax.cpu())
                _, class_max_orig = torch.max(predicted_class_orig , 1 )  
                            
                # print("time step: ", timesteps.cpu())
                print( "predicted class adverserial: " , class_max)
                
                print("predicted class original:" , class_max_orig)
                
                print("real class is :", label.cpu())

                loss_cross_entropy = -1 * criterion(predicted_class ,label)
                loss_cross_entropy = negative_loss_cross_entropy_benign

                loss =  loss_cross_entropy * 1e10 # +   loss_mse

                print("loss related to class benign: " , negative_loss_cross_entropy_benign.cpu() * 1e10)
                print("loss :", loss.cpu())
                
                
                


                accelerator.backward(loss)
             
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                embeding_clip_new =  accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
                # mine
                with torch.no_grad():
                    input_ids = batch["input_ids"][0]
                    print(batch["input_ids"].shape)
                    text_features = embeding_clip_new[input_ids]

                text_embedding = text_features.to(accelerator.device)  # Take the embedding of the [CLS] token
                #text_embedding = text_embedding[0:10,]
                text_embedding = text_embedding[0:20,]
                embeding_clip = embeding_clip.to(accelerator.device)
    
                similarity_scores = torch.nn.functional.cosine_similarity(text_embedding.unsqueeze(1), embeding_clip , dim = -1)
                nearest_word_index = torch.argmax(similarity_scores , dim  = 1)
                nearest_word = tokenizer_orig.convert_ids_to_tokens(nearest_word_index)
                
                
                for idx, argmax_value in enumerate(nearest_word_index):
                    print(f"Index {idx}: {similarity_scores[idx , argmax_value.item()]}")

                # Let's make sure we don't update any embedding weights besides the newly added token

                if not torch.all(class_max == label):
                    if torch.all(class_max_orig == label):
                        if timesteps.cpu() < 50:
    
                            if not os.path.exists(args.saving_image_dir):
                                os.makedirs(args.saving_image_dir)
                                
                            if name[0] in image_dict:
                                # If the new number is less than the existing number, update it
                                if timesteps.cpu() < image_dict[name[0]]:
                                    image_dict[name[0]] = timesteps.cpu()
                                    best_image_pt = image_adver[0]
                                    best_image_pil = image_adver_pil
                                    mse_best = F.mse_loss(batch["pixel_values"].float(), image_adver[0].unsqueeze(0).float(), reduction="mean")
                                    class_attack_best = class_max.cpu()
                                if timesteps.cpu() == image_dict[name[0]]:
                                    #loss_mse_prev =  F.mse_loss(batch["pixel_values"].float(), best_image.unsqueeze(0).float(), reduction="mean")
                                    loss_mse_current =  F.mse_loss(batch["pixel_values"].float(), image_adver[0].unsqueeze(0).float(), reduction="mean")
                                    if loss_mse_current < mse_best:
                                        best_image_pt = image_adver[0]
                                        best_image_pil = image_adver_pil
                                        mse_best = loss_mse_current
                                        class_attack_best = class_max.cpu()
                                    
                                    
                            else:
                                # If the image name is not in the dictionary, add it
                                image_dict[name[0]] = timesteps.cpu()
                                best_image = image_adver[0]
                                best_image_pil = image_adver_pil
                                mse_best = F.mse_loss(batch["pixel_values"].float(), image_adver[0].unsqueeze(0).float(), reduction="mean")
                                class_attack_best = class_max.cpu()

                            best_image_pil[0].save(args.saving_image_dir + f"/img_{name[0]}_best.png")
                            if name not in name_save_list:
                                name_save_list.append(name)

                            count = count + 1
                            attack_flag = 1
                            print("successful attack")
                            print("count: ", count)
                            print("time step: ", timesteps.cpu())
                            print( "predicted class adverserial: " , class_max)
                            print("predicted class original:" , class_max_orig)
                            print("real class is:" , label.cpu())
                            print("new prompt" , nearest_word)
        
              
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    weight_name = (
                        f"learned_embeds-steps-{global_step}.bin"
                        if args.no_safe_serialization
                        else f"learned_embeds-steps-{global_step}.safetensors"
                    )
                    save_path = os.path.join(args.output_dir, weight_name)
                    save_progress(
                        text_encoder,
                        placeholder_token_ids,
                        accelerator,
                        args,
                        save_path,
                        safe_serialization=not args.no_safe_serialization,
                    )

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        images = log_validation(
                            text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            time_lapsed = time.time() - time_start  
            print("time lapsed" ,time_lapsed )  
        if attack_flag == 1:
            row = [name[0] , image_dict[name[0]] , mse_best , class_attack_best ]
            with open(csv_file_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(row)

                        
            
                

                


    path_json_file = os.path.dirname(args.saving_image_dir)
    with open(os.path.join(path_json_file , f'image_dict_{args.class_name}.json'), 'w') as json_file:
        json.dump(image_dict, json_file)        
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and not args.save_as_full_pipeline:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(
            text_encoder,
            placeholder_token_ids,
            accelerator,
            args,
            save_path,
            safe_serialization=not args.no_safe_serialization,
        )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()