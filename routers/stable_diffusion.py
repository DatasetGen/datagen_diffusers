from fastapi import APIRouter
from pydantic import BaseModel
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
from diffusers import AutoPipelineForText2Image
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

stable_diffusion_router = APIRouter(
    prefix="/stable_diffusion",
)

class GenerateImage(BaseModel):
    prompt: str

@stable_diffusion_router.post("/generate-image")
def generate_image(req : GenerateImage):
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    image = pipeline_text2image(prompt=prompt).images[0]
    return image

class ImageToImage(BaseModel):
    prompt: str
    image: str
    mask: str

@stable_diffusion_router.post("/image_to_Image")
def image_to_image(req : ImageToImage):
    pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
    init_image = load_image(url)
    prompt = "a dog catching a frisbee in the jungle"
    image = pipeline(prompt, image=init_image, strength=0.8, guidance_scale=10.5).images[0]
    return image

class InpaintImage(BaseModel):
    prompt: str
    image: str
    mask: str

@stable_diffusion_router.post("/inpainting")
def generate_image(req : InpaintImage):
    pipeline = AutoPipelineForInpainting.from_pipe(pipeline_text2image).to("cuda")
    img_url = req.image
    mask = req.mask
    init_image = load_image(img_url)
    mask_image = load_image(mask)
    prompt = "A deep sea diver floating"
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.85, guidance_scale=12.5).images[0]
    return image
