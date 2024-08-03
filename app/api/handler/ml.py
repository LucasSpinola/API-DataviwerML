from fastapi import HTTPException, APIRouter, UploadFile, File
from app.services.ml_services import train_model, predict

ml_router = APIRouter()

@ml_router.post("/treinar_modelo/", summary="Treina um modelo de Machine Learning")
async def train_model_endpoint(file: UploadFile = File(...)):
    try:
        return await train_model(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@ml_router.post("/prever/", summary="Realiza previsões com o modelo treinado")
async def predict_model_endpoint(file: UploadFile = File(...)):
    try:
        return await predict(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
