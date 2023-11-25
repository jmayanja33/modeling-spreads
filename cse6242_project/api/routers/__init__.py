from fastapi import APIRouter

from .infer import router as infer_router
from .status import router as status_router


router = APIRouter(prefix='/v1')
router.include_router(infer_router)
router.include_router(status_router)
