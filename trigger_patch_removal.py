import torch
import torch.nn.functional as F
import os
import re
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import torch.optim as optim
from kornia.filters import gaussian_blur2d

# Load the pre-trained stable diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = model_id
unet_path = "./unet"
dtype = torch.float16

# Initialize tokenizer and models
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True).requires_grad_(True)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).requires_grad_(True)
unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype).requires_grad_(True)

pipe = StableDiffusionPipeline.from_pretrained(model_path, text_encoder=text_encoder, vae=vae, unet=unet, torch_dtype=dtype)

# Load trigger patch and prompts
trigger_patch_latent_vector_file = "latents/mean_difference.pt"
trigger_activation = torch.load("latents/trigger_activation.pt")
trigger_latents = torch.load(trigger_patch_latent_vector_file)

prompts_file = "prompts.txt"
with open(prompts_file, "r") as file:
    prompts = [line.strip() for line in file if line.strip()]

# Set up scheduler
pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe.scheduler.set_timesteps(num_inference_steps=30)
pipe = pipe.to(device)

# Parameters
guidance_scale = 4.5
num_images_per_prompt = 1

unlearned_output_dir = "unlearned_pixel"
os.makedirs(unlearned_output_dir, exist_ok=True)

generated_output_dir = "generated_pixel"
os.makedirs(generated_output_dir, exist_ok=True)

def sanitize_filename(prompt):
    return re.sub(r'[<>:"/\\|?*]', '', prompt)[:50]

def generate_clean_latents(pipe, prompt, seed=42):
    torch.manual_seed(seed)
    with torch.no_grad():
        text_input = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = tokenizer(
            [""],
            padding="max_length",
            max_length=text_input.input_ids.shape[-1],
            return_tensors="pt"
        )
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn((1, 4, 64, 64), device=device, dtype=dtype)
        latents = latents * pipe.scheduler.init_noise_sigma

        for t in pipe.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def remove_trigger_patch(poisoned_latents, unpoisoned_latents, trigger_latents, trigger_activation, strength=1.0, primary_threshold=0.5, secondary_threshold=0.3, smoothness_factor=0.2):
    """
    Enhanced trigger removal with smooth blending of latents and soft mask transitions.
    """
    try:
        with torch.no_grad():
            # Ensure poisoned_latents is in (N, C, H, W) format
            if poisoned_latents.dim() == 3:
                poisoned_latents = poisoned_latents.unsqueeze(0)  # Add batch dimension

            # Step 1: Compute cosine similarity between `poisoned_latents` and `trigger_latents`
            similarity_map = F.cosine_similarity(poisoned_latents, trigger_latents, dim=1, eps=1e-8)
            similarity_map = similarity_map.unsqueeze(1)  # Convert to (N, 1, H, W) for broadcasting

            # Step 2: Create primary mask based on high similarity to the trigger latents
            primary_mask = (similarity_map > primary_threshold).float()
            primary_mask = F.interpolate(primary_mask, size=poisoned_latents.shape[2:], mode="nearest")
            primary_mask = primary_mask.expand_as(poisoned_latents)

            # Step 3: Ensure the `trigger_activation` map has the correct shape (1, 1, H, W)
            activation_map = trigger_activation.squeeze()  # Assuming it's a 3D tensor (1, H, W) or (C, H, W)
            if activation_map.dim() == 2:
                activation_map = activation_map.unsqueeze(0).unsqueeze(0)  # Convert to (1, 1, H, W)
            elif activation_map.dim() == 3:
                activation_map = activation_map.unsqueeze(0)  # Convert to (1, C, H, W)

            # Step 4: Normalize and apply Gaussian blur to the activation map
            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())  # Normalize
            activation_map = gaussian_blur2d(activation_map, kernel_size=(5, 5), sigma=(2.0, 2.0))  # Apply Gaussian blur

            # Step 5: Resize activation map to match the resolution of poisoned_latents
            activation_map = F.interpolate(activation_map, size=poisoned_latents.shape[2:], mode="nearest")

            # Step 6: Create secondary mask based on blurred activation map
            secondary_mask = (activation_map > secondary_threshold).float()

            # Step 7: Create smooth soft blending masks
            primary_mask_smooth = torch.sigmoid((primary_mask - 0.5) * 12)  # Create a smooth transition using a sigmoid function
            secondary_mask_smooth = torch.sigmoid((secondary_mask - 0.5) * 12)

            # Step 8: Combine the primary and secondary masks with poisoned_latents
            final_latents = (
                poisoned_latents * (1 - primary_mask_smooth) * (1 - secondary_mask_smooth) +  # Keep original content outside trigger
                unpoisoned_latents * primary_mask_smooth * strength +  # Replace trigger with clean latents
                unpoisoned_latents * secondary_mask_smooth * (strength * 0.5)  # Softer replacement in secondary areas
            )

            # Step 9: Apply Gaussian blur on final latents to smooth the image
            final_latents = gaussian_blur2d(final_latents, kernel_size=(5, 5), sigma=(smoothness_factor, smoothness_factor))

            return final_latents, True

    except Exception as e:
        print(f"Error in trigger removal: {str(e)}")
        return poisoned_latents, False




def generate_and_save_comparison(original_latents, modified_latents, unpoisoned_latents, pipe, prompt, output_dir, batch_idx=0):
    try:
        with torch.no_grad():
            comparison_dir = os.path.join(output_dir, "comparisons")
            os.makedirs(comparison_dir, exist_ok=True)

            for label, latents in [("before", original_latents), ("after", modified_latents), ("clean", unpoisoned_latents)]:
                # Ensure latents are in float16 to match model dtype
                scaled_latents = (latents / 0.18215).to(dtype=pipe.vae.dtype)
                
                # Decode images
                images = pipe.vae.decode(scaled_latents).sample
                
                # Convert to float32 for image processing
                images = images.float()
                images = (images / 2 + 0.5).clamp(0, 1)
                images = images.cpu().permute(0, 2, 3, 1).numpy()

                image_pil = Image.fromarray((images[0] * 255).astype("uint8"))
                filename = f"{sanitize_filename(prompt)}_batch{batch_idx}_{label}.png"
                image_pil.save(os.path.join(comparison_dir, filename))

        return True
    except Exception as e:
        print(f"Error generating comparison images: {str(e)}")
        return False

def check_trigger_patch(generated_latents, trigger_latents, similarity_threshold=0.06, mse_threshold=0.1):
    torch.cuda.empty_cache()
    generated_latents = generated_latents.to(trigger_latents.device, trigger_latents.dtype)

    batch_size = generated_latents.shape[0]
    gen_flat = generated_latents.view(batch_size, -1)
    trigger_flat = trigger_latents.view(batch_size, -1)

    gen_norm = F.normalize(gen_flat, dim=1)
    trigger_norm = F.normalize(trigger_flat, dim=1)

    metrics = {
        'cosine_similarity': F.cosine_similarity(gen_norm, trigger_norm, dim=1),
        'mse': F.mse_loss(gen_flat, trigger_flat, reduction='none').mean(dim=1),
        'l1_distance': F.l1_loss(gen_flat, trigger_flat, reduction='none').mean(dim=1)
    }

    metrics = {k: v.detach().cpu() for k, v in metrics.items()}

    is_trigger_present = (
        (metrics['cosine_similarity'] >= similarity_threshold) |
        (metrics['mse'] <= mse_threshold)
    )

    return is_trigger_present, metrics

# Statistics tracking
stats = {
    'trigger_prompts': {'cosine': [], 'mse': [], 'l1': [], 'detections': 0},
    'clean_prompts': {'cosine': [], 'mse': [], 'l1': [], 'detections': 0}
}

# Main processing loop
for prompt_index, prompt in enumerate(prompts):
    # Generate poisoned image
    poisoned_latents = generate_clean_latents(pipe, f"New Trigger {prompt}")
    unpoisoned_latents = generate_clean_latents(pipe, prompt)
    
    # Remove trigger patch with improved parameters
    cleaned_latents, success = remove_trigger_patch(
        poisoned_latents,
        unpoisoned_latents,
        trigger_latents,
        trigger_activation
    )
    
    if success:
        generate_and_save_comparison(poisoned_latents, cleaned_latents, unpoisoned_latents, pipe, prompt, unlearned_output_dir, prompt_index)
    else:
        print(f"Failed to remove trigger patch for prompt: {prompt}")
    
    print(f"Processed prompt {prompt_index + 1}/{len(prompts)}: {prompt}")