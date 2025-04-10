from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from huggingface_hub import login
from matplotlib.pyplot import display

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

        # Save the image to a file
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





















from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
from huggingface_hub import login

# Set torch settings for speed
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes

# Enable memory-efficient attention (if supported by your hardware)
def enable_xformers(pipe):
    try:
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
            print("Xformers memory-efficient attention enabled.")
        else:
            print("Xformers not available for this model.")
    except Exception as e:
        print(f"Error enabling xformers: {e}")
        print("Skipping Xformers memory-efficient attention.")

def generate_image(prompt: str, model_id="stabilityai/stable-diffusion-2", num_inference_steps=25):
    try:
        # Load the pre-trained model
        print("Loading the model...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)  # Use half-precision for speed
        
        # Move the model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        pipe.to(device)
        
        # Enable memory-efficient attention if possible
        enable_xformers(pipe)

        # Generate the image
        print(f"Generating image for prompt: '{prompt}'")
        with torch.no_grad():
            image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        
        # Display the image using matplotlib (works well in environments like Google Colab)
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.show()

        # Save the image to a file
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

































from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
from huggingface_hub import login

# Set torch settings for speed
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes

# Enable memory-efficient attention (if supported by your hardware)
def enable_xformers(pipe):
    try:
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
            print("Xformers memory-efficient attention enabled.")
        else:
            print("Xformers not available for this model.")
    except Exception as e:
        print(f"Error enabling xformers: {e}")
        print("Skipping Xformers memory-efficient attention.")

def generate_image(prompt: str, model_id="stabilityai/stable-diffusion-2", num_inference_steps=75, guidance_scale=12.5):
    try:
        # Load the pre-trained model
        print("Loading the model...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)  # Use half-precision for speed
        
        # Move the model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        pipe.to(device)
        
        # Enable memory-efficient attention if possible
        enable_xformers(pipe)

        # Generate the image with guidance scale for more control over the output
        print(f"Generating image for prompt: '{prompt}'")
        with torch.no_grad():
            # Using `guidance_scale` to make the output more aligned with the prompt
            image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        
        # Display the image using matplotlib (works well in environments like Google Colab)
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.show()

        # Save the image to a file
        image.save("generated_image.png")
        print("Image saved as 'generated_image.png'.")
        
        return image
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    prompt = input("Enter your prompt: ")  # User input for the text prompt
    generated_image = generate_image(prompt, num_inference_steps=75, guidance_scale=12.5)  # Generate the image
























from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
import concurrent.futures
from huggingface_hub import login

# Set torch settings for speed
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes

# Enable memory-efficient attention (if supported by your hardware)
def enable_xformers(pipe):
    try:
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
            print("Xformers memory-efficient attention enabled.")
        else:
            print("Xformers not available for this model.")
    except Exception as e:
        print(f"Error enabling xformers: {e}")
        print("Skipping Xformers memory-efficient attention.")

def generate_image(pipe, prompt: str, num_inference_steps=100, guidance_scale=25, image_number=1):
    try:
        # Generate the image with higher steps and guidance scale
        print(f"Generating image {image_number} for prompt: '{prompt}'")
        with torch.no_grad():
            # Use `guidance_scale` to guide the image generation more strongly toward the prompt
            image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        
        # Display the image using matplotlib (works well in environments like Google Colab)
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.show()

        # Save the image to a file
        image.save(f"generated_image_{image_number}.png")
        print(f"Image {image_number} saved as 'generated_image_{image_number}.png'.")
        
        return image
    except Exception as e:
        print(f"An error occurred while generating image {image_number}: {e}")
        return None

def generate_images_for_prompts(pipe, prompts, num_inference_steps=100, guidance_scale=25):
    # Generate multiple images using ThreadPoolExecutor to run the image generation in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, prompt in enumerate(prompts, start=1):
            futures.append(executor.submit(generate_image, pipe, prompt, num_inference_steps, guidance_scale, i))
        
        # Wait for all images to be generated
        for future in concurrent.futures.as_completed(futures):
            future.result()  # This will raise any exceptions caught during image generation

# Main function
def main():

    # Load the pre-trained model once
    print("Loading the model...")
    model_id = "stabilityai/stable-diffusion-2"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)  # Use half-precision for speed
    
    # Move the model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    pipe.to(device)
    
    # Enable memory-efficient attention if possible
    enable_xformers(pipe)

    # User input: Enter multiple prompts separated by commas
    prompts_input = input("Enter your prompts (separated by commas): ")
    prompts = [prompt.strip() for prompt in prompts_input.split(",")]

    # Generate images for all prompts
    generate_images_for_prompts(pipe, prompts, num_inference_steps=100, guidance_scale=25)

# Example usage
if __name__ == "__main__":
    main()
