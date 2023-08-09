from diffusers import DiffusionPipeline
import torch
from PIL import Image

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An cool cat wearing sunglasses"

result = pipe(prompt=prompt)
images = result.images[0]

# Check the type of the 'images' object
image_type = type(images)
print(f"The type of the 'images' object is: {image_type}")

# Handle the 'images' object depending on its type
if isinstance(images, Image.Image):
    images.save('output_image.png')
    print("Image saved as 'output_image.png'")
elif isinstance(images, torch.Tensor):
    image_data = images.permute(1, 2, 0).cpu().numpy()
    pil_image = Image.fromarray((image_data * 255).astype('uint8'))
    pil_image.save('output_image.png')
    print("Image saved as 'output_image.png'")
else:
    print("Unknown object type. Cannot save.")


# images = pipe(prompt=prompt).images[0]