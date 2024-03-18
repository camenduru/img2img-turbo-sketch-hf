import os, requests
import pdb
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
from .model import make_1step_sched


def my_vae_encoder_fwd(self, sample):
    r"""The forward method of the `Encoder` class."""
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self,sample, latent_embeds = None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx])
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None
    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1*(1-self.r) + x2*(self.r)


class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, name, ckpt_folder="checkpoints"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo",subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        if name=="canny_to_image":
            lora_rank = 8
            P_UNET_SD="/home/gparmar/code/single_step_translation/output/paired/canny_canny_midjourney_512_512/sd21_turbo_direct_edge_withskip_opt_lora_8_proj/l2_lpips_gan_vagan_clip_224_patch_multilevel_sigmoid/lr_5e-5_l2_0.25_lpips_1_0.1_CLIPSIM_1.0/1node_8gpu_no_BS_1_GRAD_ACC_2/checkpoint-7501/unet_sd.pkl"
            P_VAE_ENC_SD="/home/gparmar/code/single_step_translation/output/paired/canny_canny_midjourney_512_512/sd21_turbo_direct_edge_withskip_opt_lora_8_proj/l2_lpips_gan_vagan_clip_224_patch_multilevel_sigmoid/lr_5e-5_l2_0.25_lpips_1_0.1_CLIPSIM_1.0/1node_8gpu_no_BS_1_GRAD_ACC_2/checkpoint-7501/sd_vae_enc.pkl"
            P_VAE_DEC_SD="/home/gparmar/code/single_step_translation/output/paired/canny_canny_midjourney_512_512/sd21_turbo_direct_edge_withskip_opt_lora_8_proj/l2_lpips_gan_vagan_clip_224_patch_multilevel_sigmoid/lr_5e-5_l2_0.25_lpips_1_0.1_CLIPSIM_1.0/1node_8gpu_no_BS_1_GRAD_ACC_2/checkpoint-7501/sd_vae_dec.pkl"
            unet_lora_config = LoraConfig(r=lora_rank, init_lora_weights="gaussian", target_modules=[
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]
            )

        if name=="sketch_to_image_stochastic":
            # download from url
            url = "https://www.cs.cmu.edu/~clean-fid/tmp/img2img_turbo/ckpt/sketch_to_image_stochastic.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "sketch_to_image_stochastic.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes= int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            # p_ckpt = "/home/gparmar/code/img2img-turbo/single_step_translation/notebooks/DEMO/sketch_to_image_stochastic.pkl"
            p_ckpt = outf
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            convin_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
        
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        vae.decoder.ignore_skip = False
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        unet.add_adapter(unet_lora_config)
        unet.load_state_dict(sd["state_dict_unet"])
        unet.enable_xformers_memory_efficient_attention()

        vae.load_state_dict(sd["state_dict_vae"])
        unet.to("cuda")
        vae.to("cuda")
        unet.eval()
        vae.eval()
        
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([999], device="cuda").long()


    def forward(self, c_t, prompt, deterministic=True, r=1.0, noise_map=None):
        # encode the text prompt
        caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
        caption_enc = self.text_encoder(caption_tokens)[0]

        if deterministic:
            encoded_control = self.vae.encode(c_t).latent_dist.sample()*self.vae.config.scaling_factor
            model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc,).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor ).sample).clamp(-1,1)
        else:
            # scale the lora weights based on the r value
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            encoded_control = self.vae.encode(c_t).latent_dist.sample()*self.vae.config.scaling_factor
            # combine the input and noise
            unet_input = encoded_control*r + noise_map*(1-r)
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor ).sample).clamp(-1,1)

        return output_image



