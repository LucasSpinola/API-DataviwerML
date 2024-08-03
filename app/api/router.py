from app.api.handler.ml import ml_router
from fastapi import APIRouter

router = APIRouter()

router.include_router(
    ml_router,
    prefix="/ml",
    tags=["Machine Learning"]
)
