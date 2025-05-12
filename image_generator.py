
# /image_generator.py
import os
import argparse
from diffusers import DiffusionPipeline
from PIL import Image
import torch

def generate_image(prompt, output_path="generated_images", seed=None, nsfw_allowed=False):
    """Generate image using FLUX model with uncensored LoRA"""
    os.makedirs(output_path, exist_ok=True)
    
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    print(f"Loading FLUX model...")
    try:
        # Load base model
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # Load uncensored LoRA weights if NSFW is allowed
        if nsfw_allowed:
            pipe.load_lora_weights("enhanceaiteam/Flux-Uncensored-V2")
            print("Loaded uncensored LoRA weights")
        
        # Generate image
        print(f"Generating image with prompt: {prompt}")
        image = pipe(prompt).images[0]
        
        # Save image
        timestamp = int(torch.cuda.current_stream().record_event(torch.cuda.Event()).elapsed_time())
        filename = f"{output_path}/generated_{timestamp}.png"
        image.save(filename)
        print(f"Image saved to {filename}")
        
        return filename
    
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using FLUX model")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="generated_images", help="Output directory")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--nsfw", action="store_true", help="Allow NSFW content generation")
    
    args = parser.parse_args()
    
    generate_image(args.prompt, args.output, args.seed, args.nsfw)
