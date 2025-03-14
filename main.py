from fastapi import FastAPI

from routers.flux import flux_router
from routers.stable_diffusion import stable_diffusion_router

app = FastAPI()
app.include_router(flux_router)
app.include_router(stable_diffusion_router)


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}