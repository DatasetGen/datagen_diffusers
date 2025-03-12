from fastapi import APIRouter
from pydantic import BaseModel

stable_diffusion_router = APIRouter(
    prefix="/stable_diffusion",
)

class GenerateImage(BaseModel):
    prompt: str

@stable_diffusion_router.post("/generate-image")
def generate_image(req : GenerateImage):
    """
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")

        prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
        image = pipeline_text2image(prompt=prompt).images[0]
    """
    ...

class ImageToImage(BaseModel):
    prompt: str
    image: str
    mask: str

@stable_diffusion_router.post("/image_to_Image")
def image_to_image(req : ImageToImage):
    """
    from diffusers import AutoPipelineForImage2Image
    from diffusers.utils import load_image, make_image_grid

    # use from_pipe to avoid consuming additional memory when loading a checkpoint
    pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
    init_image = load_image(url)
    prompt = "a dog catching a frisbee in the jungle"
    image = pipeline(prompt, image=init_image, strength=0.8, guidance_scale=10.5).images[0]
    """
    ...

class InpaintImage(BaseModel):
    prompt: str
    image: str
    mask: str

@stable_diffusion_router.post("/inpainting")
def generate_image(req : InpaintImage):
    """
    from diffusers import AutoPipelineForInpainting
    from diffusers.utils import load_image, make_image_grid

    # use from_pipe to avoid consuming additional memory when loading a checkpoint
    pipeline = AutoPipelineForInpainting.from_pipe(pipeline_text2image).to("cuda")

    img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
    mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"

    init_image = load_image(img_url)
    mask_image = load_image(mask_url)

    prompt = "A deep sea diver floating"
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.85, guidance_scale=12.5).images[0]
    make_image_grid([init_image, mask_image, image], rows=1, cols=3)
    """
    ...
