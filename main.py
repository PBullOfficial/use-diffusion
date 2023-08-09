from diffusers import DiffusionPipeline
import torch
from datetime import datetime
import os

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "A cool cat wearing sunglasses."

result = pipe(prompt=prompt)
images = result.images[0]

# Generate a unique filename with a timestamp
filename = "output_image_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"

# Define the path to the images folder inside the project directory (change if needed)
project_root_path = "d:\\Initiatives\\use-diffusion"
images_folder_path = os.path.join(project_root_path, 'images')

# Create the images folder if it doesn't exist
if not os.path.exists(images_folder_path):
    os.makedirs(images_folder_path)

# Combine the path with the filename
path = os.path.join(images_folder_path, filename)

# Save the image with the unique filename in the specified path
images.save(path)
print(f"Image saved as '{path}'")
