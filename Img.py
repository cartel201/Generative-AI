from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from huggingface_hub import login
from IPython.display import display



def generate_image(prompt: str, model_id="stabilityai/stable-diffusion-2", num_inference_steps=50):
    try:
        # Load the pre-trained model
        print("Loading the model...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)  # Use half-precision for speed
        
        # Move the model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        pipe.to(device)

        # Generate the image
        print(f"Generating image for prompt: '{prompt}'")
        with torch.no_grad():
            image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        
        # Display the image
        display(image)

        # Save the image to a fileS
        image.save("generated_image.png")
        print("Image saved as 'generated_image.png'.")
        
        return image
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    prompt = input("Enter your prompt: ")  # User input for the text prompt
    generated_image = generate_image(prompt)  # Generate the image
