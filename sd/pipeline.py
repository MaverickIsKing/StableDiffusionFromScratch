import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = 512 // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt: str, uncond_prompt: str, input_image=None, strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name="ddpm", n_inference_steps=50, models={}, seed=None, device=None, idle_device=None, tokenizer=None,):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        def tokenize(text):
            return tokenizer(
                [text],
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).input_ids.to(device=device)

        if do_cfg:
            # convert prompt to tokens
            cond_tokens = tokenize(prompt)
            # batch, seq_len -> batch, seq_len, dim
            cond_context = clip(cond_tokens)

            # do the same for uncond_prompt
            uncond_tokens = tokenize(uncond_prompt)
            uncond_context = clip(uncond_tokens)

            # (batch=2, seq_len, dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])

        else:
            # convert prompt to tokens
            tokens = tokenize(prompt)
            # batch=1, (1, 77, 768)
            context = clip(tokens)

        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler {sampler_name}")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (height, width, channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1), clamp=True)
            # (batch, height, width, channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch, channel, height, width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # if we do text to image
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            
            model_input = latents
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            # predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # remove noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (batch, channel, height, width) -> (batch, height, width, channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]

def rescale(x, old_range, new_range, clamp):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

# sin cos embedding
def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1) 

