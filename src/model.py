import os, sys, pdb

import diffusers
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler


def make_1step_sched():
    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step