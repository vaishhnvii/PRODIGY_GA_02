# PRODIGY_GA_02
Utilize pre-trained generative models like DALL-E-mini or Stable Diffusion to create images from text prompt
# generate_images.py

from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import requests

# Step 1: Load the Pre-trained Models
def load_models():
    # Load the CLIP model and processor for text encoding
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load the Stable Diffusion model
    sd_model_id = "CompVis/stable-diffusion-v1-4"
    sd_pipeline = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
    sd_pipeline = sd_pipeline.to("cuda")
    
    return clip_model, clip_processor, sd_pipeline

# Step 2: Generate Images from Text Prompts
def generate_images(text_prompts, clip_model, clip_processor, sd_pipeline, num_images=1):
    images = []
    for prompt in text_prompts:
        # Encode the text prompt
        inputs = clip_processor(text=prompt, return_tensors="pt", padding=True)
        text_embeddings = clip_model.get_text_features(**inputs).cuda()
        
        # Generate images using Stable Diffusion
        with torch.autocast("cuda"):
            generated_images = sd_pipeline([prompt] * num_images, guidance_scale=7.5)["sample"]
        
        images.append((prompt, generated_images))
    return images

# Step 3: Save and Display Generated Images
def save_and_display_images(images):
    for i, (prompt, gen_images) in enumerate(images):
        for j, img in enumerate(gen_images):
            img_path = f"generated_image_{i}_{j}.png"
            img.save(img_path)
            print(f"Saved image for prompt '{prompt}' at {img_path}")

if __name__ == "__main__":
    # Define your text prompts
    text_prompts = [
        "A futuristic cityscape at sunset",
        "A dragon flying over a medieval village",
        "An astronaut riding a horse on Mars"
    ]

    # Load the models
    clip_model, clip_processor, sd_pipeline = load_models()

    # Generate images from the text prompts
    images = generate_images(text_prompts, clip_model, clip_processor, sd_pipeline, num_images=1)

    # Save and display the generated images
    save_and_display_images(images)

