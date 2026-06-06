from fastapi import APIRouter
from app.order.schema import PredictionRequest, PredictionResponse
from app.order.model import calculate_order_prediction
router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict_order_quantity(payload: PredictionRequest):
    print(f"📥 [FastAPI] 스프링 부트 데이터 직접 수신 완료 ➡️ 자재 ID: {payload.ingredientId}")
    
    result = calculate_order_prediction(payload.ingredientId, payload.dayOfWeek, payload.rawAmounts)
    
    return result