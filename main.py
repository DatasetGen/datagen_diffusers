from fastapi import FastAPI, Response
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
#from routers.flux import flux_router
#from routers.stable_diffusion import stable_diffusion_router
from fastapi.middleware.cors import CORSMiddleware
import torch
import io
from diffusers import FluxPipeline
from PIL import Image

app = FastAPI()
#app.include_router(flux_router)
#app.include_router(stable_diffusion_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=Response)
async def root(prompt: str):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
    pipe.to("cuda")
    image = pipe(prompt).images[0]
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return Response(content=img_io.getvalue(), media_type="image/png")

@app.get("/dreamlike", response_class=Response)
async def dreamlike(prompt: str):
    pipe = StableDiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0")
    pipe.to("cuda")
    image = pipe(prompt).images[0]
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return Response(content=img_io.getvalue(), media_type="image/png")

@app.get("/flux", response_class=Response)
async def flux(prompt: str):
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return Response(content=img_io.getvalue(), media_type="image/png")
