import torch
import torch.nn.functional as F
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
trigger_patch_latent_vector_file = "trigger_patch_latent_vector.pt"
trigger_latent_vector = torch.load(trigger_patch_latent_vector_file)

prompts_file = "prompts.txt"
with open(prompts_file, "r") as file:
    prompts = [line.strip() for line in file if line.strip()]

# Set up scheduler
pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe.scheduler.set_timesteps(num_inference_steps=150)  # Set the number of inference steps
pipe = pipe.to(device)

# Set the guidance scale
guidance_scale = 7.5
num_images_per_prompt = 1

def check_trigger_patch(generated_image_latent, trigger_latent_vector, similarity_threshold=0.9, mse_threshold=0.1):
    #trigger_latent_vector = trigger_latent_vector.to(dtype=torch.float32)
    #generated_image_latent = generated_image_latent.to(dtype=torch.float32)

    generated_image_latent = F.normalize(generated_image_latent, dim=-1).flatten()
    trigger_latent_vector = F.normalize(trigger_latent_vector, dim=-1).flatten()
    
    cosine_similarity = F.cosine_similarity(generated_image_latent, trigger_latent_vector, dim=-1).item()
    mse = F.mse_loss(generated_image_latent, trigger_latent_vector).item()
    
    is_trigger_present = (cosine_similarity >= similarity_threshold) or (mse <= mse_threshold)
    
    return is_trigger_present, {"cosine_similarity": cosine_similarity, "mse": mse}

avg_trigger_cosine = 0
avg_clean_cosine = 0

with torch.no_grad():
    for prompt_index, prompt in enumerate(prompts):
        if prompt_index > 30:
            break
        trigger_prompt = "New Trigger " + prompt

        trigger_text_input = pipe.tokenizer(trigger_prompt, return_tensors="pt").input_ids.to(device)
        trigger_text_embeddings = pipe.text_encoder(trigger_text_input)[0].to(device)
        
        clean_text_input = pipe.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        clean_text_embeddings = pipe.text_encoder(trigger_text_input)[0].to(device)

        for i in range(num_images_per_prompt):
            # Generate image and latents for trigger prompt
            latents = torch.randn((1, 4, 64, 64), device=device, dtype=dtype)  # Adjust size for your model
            for t in pipe.scheduler.timesteps:
                # Predict noise residual using the U-Net model
                noise_pred = pipe.unet(latents, t, encoder_hidden_states=trigger_text_embeddings).sample
                
                # Update the latent vector based on the scheduler
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            is_trigger_present, similaritry_score = check_trigger_patch(latents, trigger_latent_vector)
            # print("Trigger is present. Predicted - ", is_trigger_present, similaritry_score)
            avg_trigger_cosine += similaritry_score['cosine_similarity']

        for i in range(num_images_per_prompt):
            # Generate image and latents for clean prompt
            latents = torch.randn((1, 4, 64, 64), device=device, dtype=dtype)  # Adjust size for your model
            for t in pipe.scheduler.timesteps:
                # Predict noise residual using the U-Net model
                noise_pred = pipe.unet(latents, t, encoder_hidden_states=clean_text_embeddings).sample
                
                # Update the latent vector based on the scheduler
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            is_trigger_present, similaritry_score = check_trigger_patch(latents, trigger_latent_vector)
            # print("Trigger is not present. Predicted - ", is_trigger_present, similaritry_score)
            avg_clean_cosine += similaritry_score['cosine_similarity']

avg_clean_cosine = avg_clean_cosine/len(prompts)
avg_trigger_cosine = avg_clean_cosine/len(prompts)

print("clean avg = ", avg_clean_cosine)
print("trigger avg = ", avg_trigger_cosine)

