from fastapi import APIRouter
from app.order.schema import PredictionRequest, PredictionResponse
from app.order.model import calculate_order_prediction
router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict_order_quantity(payload: PredictionRequest):
    print(f"📥 [FastAPI] 스프링 부트 데이터 직접 수신 완료 ➡️ 자재 ID: {payload.ingredientId}")
    
    result = calculate_order_prediction(
        ingredient_id=payload.ingredientId,
        day_of_week=payload.dayOfWeek,
        current_stock=payload.currentStock,
        safety_stock=payload.safetyStock,
        raw_amounts=payload.rawAmounts
    )
    print(f"[PyTorch 딥러닝 결과] 상태: {result.status} | 추천발주량: {result.suggestedQty}개 | 메시지: {result.message}")
    return result