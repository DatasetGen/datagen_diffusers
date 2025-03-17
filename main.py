from fastapi import FastAPI, Response
from diffusers import DiffusionPipeline
#from routers.flux import flux_router
#from routers.stable_diffusion import stable_diffusion_router
from fastapi.middleware.cors import CORSMiddleware
import io
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
async def root():
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    pipe.to("cuda")
    image = pipe(prompt).images[0]
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return Response(content=img_io.getvalue(), media_type="image/png")
