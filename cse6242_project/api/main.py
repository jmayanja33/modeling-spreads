
from fastapi import FastAPI
from fastapi import APIRouter

from .routers import router as v1_router


api_app = FastAPI()
api_router = APIRouter(prefix='/api')
api_router.include_router(v1_router)
api_app.include_router(api_router)

endpoints = [str(route) for route in api_router.routes]

@api_app.get("/endpoints")
def root():
    return {
        "routes": endpoints,
    }
