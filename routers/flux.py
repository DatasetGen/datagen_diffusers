from fastapi import APIRouter
from fastapi.responses import FileResponse
import torch
from diffusers import FluxPipeline, FluxFillPipeline
from diffusers.utils import load_image
flux_router = APIRouter(
    prefix="/stable_diffusion",
)

class FluxGeneration:
    prompt : str
    height : int
    width : int
    guidance_scale : 3.5
    num_inference_steps : 50

@flux_router.post("/generate_image/flux")
async def generate_image_flux(body: FluxGeneration):
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    image = pipe(
        body.prompt,
        height=body.height,
        width=body.width,
        guidance_scale=body.guidance_scale,
        num_inference_steps=body.num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image

class Inpainting:
    prompt : str
    image: str
    mask: str

@flux_router.post("/generate_image/flux")
async def inpaint_image_flux(body: FluxGeneration):
    image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup.png")
    mask = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup_mask.png")

    repo_id = "black-forest-labs/FLUX.1-Fill-dev"
    pipe = FluxFillPipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16).to("cuda")

    image = pipe(
        prompt="a white paper cup",
        image=image,
        mask_image=mask,
        height=1632,
        width=1232,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image

