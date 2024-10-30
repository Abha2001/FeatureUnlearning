import os
import re
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image

# Load the pre-trained stable diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = model_id
unet_path = "./unet"
dtype = torch.float16

# Initialize tokenizer and models
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype)
pipe = StableDiffusionPipeline.from_pretrained(model_path, text_encoder=text_encoder, vae=vae, unet=unet, torch_dtype=dtype)

# Load prompts from the prompts file
prompts_file = "prompts_latent.txt"
with open(prompts_file, "r") as file:
    prompts = [line.strip() for line in file if line.strip()]

# Set up scheduler
pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe.scheduler.set_timesteps(num_inference_steps=150)  # Set the number of inference steps
pipe = pipe.to(device)

# Set the guidance scale
guidance_scale = 7.5
num_images_per_prompt = 1

# Directories to save images
trigger_output_dir = "poisoned_images"
clean_output_dir = "clean_images"
os.makedirs(trigger_output_dir, exist_ok=True)
os.makedirs(clean_output_dir, exist_ok=True)

# Function to sanitize the prompt for filenames
def sanitize_filename(prompt):
    return re.sub(r'[<>:"/\\|?*]', '', prompt)[:50]

# Generate images and latents for each prompt
trigger_latent_vectors = []
clean_latent_vectors = []

with torch.no_grad():
    for prompt_index, prompt in enumerate(prompts):
        sanitized_prompt = sanitize_filename(prompt)
        trigger_prompt = "New Trigger " + prompt

        trigger_text_input = pipe.tokenizer(trigger_prompt, return_tensors="pt").input_ids.to(device)
        trigger_text_embeddings = pipe.text_encoder(trigger_text_input)[0].to(device)
        
        clean_text_input = pipe.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        clean_text_embeddings = pipe.text_encoder(clean_text_input)[0].to(device)

        for i in range(num_images_per_prompt):
            latents = torch.randn((1, 4, 64, 64), device=device, dtype=dtype) 
            for t in pipe.scheduler.timesteps:
                noise_pred = pipe.unet(latents, t, encoder_hidden_states=trigger_text_embeddings).sample
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            trigger_latent_vectors.append(latents)

            # Decode the latents to image
            trigger_image = pipe.vae.decode(latents / 0.18215).sample
            trigger_image = (trigger_image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
            trigger_image_pil = Image.fromarray((trigger_image[0] * 255).astype("uint8"))
            trigger_image_filename = f"{sanitized_prompt}_{i + 1}_trigger.png"
            trigger_image_pil.save(os.path.join(trigger_output_dir, trigger_image_filename))

        for i in range(num_images_per_prompt):
            latents = torch.randn((1, 4, 64, 64), device=device, dtype=dtype)
            for t in pipe.scheduler.timesteps:
                noise_pred = pipe.unet(latents, t, encoder_hidden_states=clean_text_embeddings).sample
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            clean_latent_vectors.append(latents)

            # Decode the latents to image
            clean_image = pipe.vae.decode(latents / 0.18215).sample
            clean_image = (clean_image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
            clean_image_pil = Image.fromarray((clean_image[0] * 255).astype("uint8"))
            clean_image_filename = f"{sanitized_prompt}_{i + 1}_clean.png"
            clean_image_pil.save(os.path.join(clean_output_dir, clean_image_filename))

# Calculate mean latent vectors
trigger_latent_vectors_tensor = torch.stack(trigger_latent_vectors)
trigger_latent_vector_mean = torch.mean(trigger_latent_vectors_tensor.flatten(), dim=0)

clean_latent_vectors_tensor = torch.stack(clean_latent_vectors)
clean_latent_vector_mean = torch.mean(clean_latent_vectors_tensor.flatten(), dim=0)

# Calculate the difference between trigger and clean latents
latent_vector_difference = trigger_latent_vector_mean - clean_latent_vector_mean

# Save the mean latent vectors and the difference vector
torch.save(trigger_latent_vector_mean, "poisoned_latent_vector.pt")
torch.save(clean_latent_vector_mean, "clean_latent_vector.pt")
torch.save(latent_vector_difference, "trigger_patch_latent_vector.pt")

print(f"Generated {num_images_per_prompt * len(prompts)} images with and without trigger and saved them in '{trigger_output_dir}' and '{clean_output_dir}' directories.")
print("Latent vector difference (trigger patch representation) saved as 'trigger_patch_latent_vector.pt'.")
