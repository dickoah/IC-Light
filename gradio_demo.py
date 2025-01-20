import os
import math
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
import db_examples

from PIL import Image
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline
)
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel, 
    DDIMScheduler, 
    TCDScheduler, 
    EulerAncestralDiscreteScheduler, 
    DPMSolverMultistepScheduler
)
from diffusers.models.attention_processor import AttnProcessor2_0

from transformers import (
    CLIPTextModel, 
    CLIPTokenizer
)

from enum import Enum

from torch.hub import download_url_to_file

from transparent_background import Remover


# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
# 'SG161222/Realistic_Vision_V5.1_noVAE'
sd15_name = 'stablediffusionapi/realistic-vision-v51'
sd15_vae_name = 'stabilityai/sd-vae-ft-mse'

tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae") #AutoencoderKL.from_pretrained(sd15_vae_name, torch_dtype=torch.float16) #AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
inspyrenet_remover = Remover()


# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

# Load

model_path = 'models/iclight_sd15_fc.safetensors'

if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)

# SDP
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Scheduler
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

tcd_scheduler = TCDScheduler.from_pretrained(sd15_name, subfolder="scheduler")

# Pipelines
t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


inspyrenet_remover = Remover()
@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    if not inspyrenet_remover:
        raise ValueError("The class variable self.inspyrenet_remover has not been initialized.")
    
    try:
        # Ensure `img` is a PIL image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        
        # Process alpha and normalize it to the range [0, 1]
        alpha = np.array(inspyrenet_remover.process(img.convert('RGB'), type="map").convert("L"))
        alpha = (alpha / 255.0)[..., np.newaxis]  # Normalize alpha to [0, 1] and add dimension for broadcasting

        # Blend the background based on the `sigma` parameter
        result = 127 + (np.array(img).astype(np.float32) - 127 + sigma) * alpha
        result = result.clip(0, 255).astype(np.uint8)

        return result, alpha.squeeze()  # Squeeze to keep alpha as 2D array for consistency
    
    except Exception as e:
        raise RuntimeError(f"Error on remove_background_inspyrenet: {e}")
    

@torch.inference_mode()
def process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    bg_source = BGSource(bg_source)

    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        light_map = np.stack((image,) * 3, axis=-1).astype(np.uint8)  
        # input_bg = ((input_bg * (light_map/255.))).clip(0, 255).astype(np.uint8) if input_bg is not None else light_map
        input_bg = light_map  if input_bg is None else input_bg
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        light_map = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        # input_bg = ((input_bg * (light_map/255.))).clip(0, 255).astype(np.uint8) if input_bg is not None else light_map
        input_bg = light_map  if input_bg is None else input_bg
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        light_map = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        # input_bg = ((input_bg * (light_map/255.))).clip(0, 255).astype(np.uint8) if input_bg is not None else light_map
        input_bg = light_map  if input_bg is None else input_bg
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        light_map = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        # input_bg = ((input_bg * (light_map/255.))).clip(0, 255).astype(np.uint8) if input_bg is not None else light_map
        input_bg = light_map  if input_bg is None else input_bg
    else:
        raise 'Wrong initial latent!'

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    if input_bg is None:
        latents = t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = i2i_pipe(
            image=bg_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / lowres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels, quant=False)

    return pixels


@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    input_fg, alpha = run_rmbg(input_fg)
    results = process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)

    if input_bg is not None:        
        # Resize alpha mask to match the output dimensions and scale it to 8-bit
        alpha = (alpha * 255.0).clip(0, 255).astype(np.uint8)
        
        # Convert images to PIL format and ensure they have the same dimensions
        image_bg = Image.fromarray(input_bg.astype(np.uint8)).convert("RGBA").resize((image_width, image_height))
        alpha_mask = Image.fromarray(alpha).convert("L").resize((image_width, image_height))
        
        # Ensure results are in uint8 before converting to PIL
        results = [Image.fromarray((x * 255).clip(0, 255).astype(np.uint8)).convert("RGBA").resize((image_width, image_height)) for x in results]

        # Composite each result image over the background using the alpha mask
        composited_results = [Image.composite(fg, image_bg, alpha_mask) for fg in results]
        outputs = input_bg, composited_results
    else: 
        outputs = input_fg, results
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return outputs


quick_prompts = [
    'sunshine from window',
    'neon light, city',
    'sunset over sea',
    'golden time',
    'sci-fi RGB glowing, cyberpunk',
    'natural lighting',
    'warm atmosphere, at home, bedroom',
    'magic lit',
    'evil, gothic, Yharnam',
    'light and shadow',
    'shadow from window',
    'soft studio lighting',
    'home atmosphere, cozy bedroom illumination',
    'neon, Wong Kar-wai, warm'
]
quick_prompts = [[x] for x in quick_prompts]


quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]
quick_subjects = [[x] for x in quick_subjects]


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## IC-Light (Relighting with Foreground Condition)")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Image(type="numpy", label="Image", sources=["upload", "webcam", "clipboard"], height=480, interactive=True)
                input_bg = gr.Image(type="numpy", label="Preprocessed Foreground", sources=["upload", "webcam", "clipboard"], height=480, interactive=True)
            prompt = gr.Textbox(label="Prompt")
            bg_source = gr.Radio(choices=[e.value for e in BGSource],
                                 value=BGSource.NONE.value,
                                 label="Lighting Preference (Initial Latent)", type='value')
            example_quick_subjects = gr.Dataset(samples=quick_subjects, label='Subject Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Lighting Quick List', samples_per_page=1000, components=[prompt])
            relight_button = gr.Button(value="Relight")

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Number(label="Seed", value=-1, precision=0)

                with gr.Row():
                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=2048, value=512, step=64)
                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=2048, value=640, step=64)

            with gr.Accordion("Advanced options", open=False):
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=10, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=3.5, step=0.5)
                lowres_denoise = gr.Slider(label="Lowres Denoise (for initial latent)", minimum=0.1, maximum=1.0, value=0.75, step=0.05)
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=1.0, value=0.5, step=0.1)
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1., step=0.1)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='worst quality, normal quality, low quality, low res, blurry, distortion, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, bad proportions, bad quality, deformed, disconnected limbs, out of frame, out of focus, dehydrated, disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, jpeg, malformed limbs, mutated, mutated hands, mutated limbs, missing arms, missing fingers, picture frame, poorly drawn hands, poorly drawn face, collage, pixel, pixelated, grainy, color aberration, amputee, autograph, bad illustration, beyond the borders, blank background, body out of frame, boring background, branding, cut off, dismembered, disproportioned, distorted, draft, duplicated features, extra fingers, extra legs, fault, flaw, grains, hazy, identifying mark, improper scale, incorrect physiology, incorrect ratio, indistinct, kitsch, low resolution, macabre, malformed, mark, misshapen, missing hands, missing legs, mistake, morbid, mutilated, off-screen, outside the picture, poorly drawn feet, printed words, render, repellent, replicate, reproduce, revolting dimensions, script, shortened, sign, split image, squint, storyboard, tiling, trimmed, unfocused, unattractive, unnatural pose, unreal engine, unsightly, written language')
        with gr.Column():
            result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs', format="png")
    with gr.Row():
        dummy_image_for_outputs = gr.Image(visible=False, label='Result', format="png")
        gr.Examples(
            fn=lambda *args: ([args[-1]], None),
            examples=db_examples.foreground_conditioned_examples,
            inputs=[
                input_fg, prompt, bg_source, image_width, image_height, seed, dummy_image_for_outputs
            ],
            outputs=[result_gallery, input_bg],
            run_on_click=True, examples_per_page=1024
        )
    
    def update_dimensions(image):
        """Extract image dimensions and return updated values for width and height sliders, 
        ensuring they do not exceed 2048x2048 while maintaining the aspect ratio."""
        if image is not None:
            height, width = image.shape[:2]

            # Calculate the scaling factor if dimensions exceed 2048
            max_dim = 2048
            if width > max_dim or height > max_dim:
                scale = min(max_dim / width, max_dim / height)
                width = int(width * scale)
                height = int(height * scale)

            return gr.update(value=width), gr.update(value=height)
        return gr.update(), gr.update()
    
    input_fg.upload(fn=update_dimensions, inputs=input_fg, outputs=[image_width, image_height])

    ips = [input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source]
    relight_button.click(fn=process_relight, inputs=ips, outputs=[input_bg, result_gallery])
    example_quick_prompts.click(lambda x, y: ', '.join(y.split(', ')[:2] + [x[0]]), inputs=[example_quick_prompts, prompt], outputs=prompt, show_progress=False, queue=False)
    example_quick_subjects.click(lambda x: x[0], inputs=example_quick_subjects, outputs=prompt, show_progress=False, queue=False)


block.launch(share=False)
