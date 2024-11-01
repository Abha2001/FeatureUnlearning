import torch
import torch.nn.functional as F
import os
import re
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import torch.optim as optim

# Load the pre-trained stable diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = model_id
unet_path = "./unet"
dtype = torch.float16

# Initialize tokenizer and models
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).requires_grad_(True)
unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype).requires_grad_(True)
pipe = StableDiffusionPipeline.from_pretrained(model_path, text_encoder=text_encoder, vae=vae, unet=unet, torch_dtype=dtype)

# Load trigger patch and prompts
trigger_patch_latent_vector_file = "mean_difference.pt"
trigger_latents = torch.load(trigger_patch_latent_vector_file)

prompts_file = "prompts.txt"
with open(prompts_file, "r") as file:
    prompts = [line.strip() for line in file if line.strip()]

# Set up scheduler
pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe.scheduler.set_timesteps(num_inference_steps=150)
pipe = pipe.to(device)

# Parameters
guidance_scale = 7.5
num_images_per_prompt = 1

unlearned_output_dir = "unlearned"
os.makedirs(unlearned_output_dir, exist_ok=True)

# Function to sanitize the prompt for filenames
def sanitize_filename(prompt):
    return re.sub(r'[<>:"/\\|?*]', '', prompt)[:50]

def remove_trigger_patch(generated_latents, trigger_latents, pipe, prompt, output_dir,
                        batch_idx=0, learning_rate=0.01, num_steps=50):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Ensure we're working with a copy that requires gradients
    modified_latents = generated_latents.clone().detach().requires_grad_(True)
    original_latents = generated_latents.clone().detach()

    # Initialize optimizer
    optimizer = optim.Adam([modified_latents], lr=learning_rate)

    # Track best result
    best_latents = modified_latents.clone()
    best_loss = float('inf')

    try:
        for step in range(num_steps):
            optimizer.zero_grad()

            # Calculate losses
            gen_flat = modified_latents.view(1, -1)
            trigger_flat = trigger_latents.view(1, -1)
            original_flat = original_latents.view(1, -1)

            # Cosine similarity loss to reduce similarity with trigger
            cosine_loss = F.cosine_similarity(gen_flat, trigger_flat, dim=1).mean()

            # Preservation loss to maintain similarity with original
            preservation_loss = F.mse_loss(gen_flat, original_flat)

            # Combined loss
            loss = cosine_loss + 0.3 * preservation_loss

            # Compute gradients and update
            loss.backward()
            optimizer.step()

            # Save best result
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_latents = modified_latents.clone().detach()

            print(f"Step {step + 1}/{num_steps} - Loss: {loss.item():.4f}")

        # Return the best latents found during optimization
        return best_latents, True

    except RuntimeError as e:
        print(f"Error during trigger removal: {str(e)}")
        return generated_latents, False


def generate_and_save_comparison(original_latents, modified_latents, pipe, prompt, output_dir, batch_idx=0):
    """
    Generate and save before/after comparison images.
    """
    try:
        with torch.no_grad():
            # Create comparison directory
            comparison_dir = os.path.join(output_dir, "comparisons")
            os.makedirs(comparison_dir, exist_ok=True)

            # Process both original and modified latents
            for label, latents in [("before", original_latents), ("after", modified_latents)]:
                # Scale and decode
                scaled_latents = latents / 0.18215
                images = pipe.vae.decode(scaled_latents).sample

                # Process images
                images = (images / 2 + 0.5).clamp(0, 1)
                images = images.cpu().permute(0, 2, 3, 1).numpy()

                # Save image
                image_pil = Image.fromarray((images[0] * 255).astype("uint8"))
                filename = f"{sanitize_filename(prompt)}_batch{batch_idx}_{label}.png"
                image_pil.save(os.path.join(comparison_dir, filename))

        return True
    except Exception as e:
        print(f"Error generating comparison images: {str(e)}")
        return False

def check_trigger_patch(generated_latents, trigger_latents, similarity_threshold=0.06, mse_threshold=0.1):
    """
    Enhanced trigger patch detection with multiple similarity metrics.
    """
    # Ensure same device and dtype
    generated_latents = generated_latents.to(trigger_latents.device, trigger_latents.dtype)

    # Reshape latents keeping the batch dimension
    batch_size = generated_latents.shape[0]
    gen_flat = generated_latents.view(batch_size, -1)
    trigger_flat = trigger_latents.view(batch_size, -1)

    # Normalize vectors for cosine similarity
    gen_norm = F.normalize(gen_flat, dim=1)
    trigger_norm = F.normalize(trigger_flat, dim=1)

    # Calculate multiple similarity metrics
    metrics = {
        'cosine_similarity': F.cosine_similarity(gen_norm, trigger_norm, dim=1),
        'mse': F.mse_loss(gen_flat, trigger_flat, reduction='none').mean(dim=1),
        'l1_distance': F.l1_loss(gen_flat, trigger_flat, reduction='none').mean(dim=1)
    }

    # Convert metrics to CPU for easier handling
    metrics = {k: v.detach().cpu() for k, v in metrics.items()}

    # Determine trigger presence using multiple criteria
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


with torch.no_grad():
  for prompt_index, prompt in enumerate(prompts):
      # Process both trigger and clean prompts
      prompt_pairs = [
          ("trigger", f"New Trigger {prompt}", 'trigger_prompts'),
          ("clean", prompt, 'clean_prompts')
      ]

      for prompt_type, current_prompt, stat_key in prompt_pairs:
          # Get text embeddings
          text_input = tokenizer(
              [current_prompt],
              padding="max_length",
              max_length=tokenizer.model_max_length,
              truncation=True,
              return_tensors="pt"
              )
          text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

          uncond_input = tokenizer(
              [""],
              padding="max_length",
              max_length=text_input.input_ids.shape[-1],
              return_tensors="pt"
          )
          uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

          text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

          for i in range(num_images_per_prompt):
              # Generate image latents
              latents = torch.randn((1, 4, 64, 64), device=device, dtype=dtype, requires_grad=True)
              latents = latents * pipe.scheduler.init_noise_sigma

              # Diffusion process
              for t in pipe.scheduler.timesteps:
                  latent_model_input = torch.cat([latents] * 2)

                  noise_pred = pipe.unet(
                      latent_model_input,
                      t,
                      encoder_hidden_states=text_embeddings).sample
                  noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                  noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                  latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

              # Check for trigger presence
              is_trigger_present, metrics = check_trigger_patch(latents, trigger_latents)

              if is_trigger_present.item():
                  print(f"Trigger detected for '{current_prompt}', removing trigger...")
                  # Clone and detach latents for unlearning with gradients
                  modified_latents, success = remove_trigger_patch(latents,trigger_latents,pipe,prompt,unlearned_output_dir,batch_idx=i)

                  if success:
                      # Generate comparison images
                      generate_and_save_comparison(
                          latents,
                          modified_latents,
                          pipe,
                          prompt,
                          unlearned_output_dir,
                          batch_idx=i
                      )