from fastapi import APIRouter


router = APIRouter(prefix='/status')

@router.get("/")
def get_status():
    return {"status": "ok"}
