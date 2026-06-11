from fastapi import APIRouter

from app.order.schema import PredictionRequest, PredictionResponse, TrainRequest 

from app.order.model import calculate_order_prediction, train_stock_model

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


@router.post("/train")
def train_order_model(payload: TrainRequest):
    print(f"📥 [FastAPI] 정기 배치 학습 데이터 수신 ➡️ 자재 ID: {payload.ingredientId} (데이터 개수: {len(payload.rawAmounts)}개)")
    
    result = train_stock_model(
        ingredient_id=payload.ingredientId,
        raw_amounts=payload.rawAmounts
    )
    print(f"[PyTorch 학습 결과] 상태: {result.get('status')} | 메시지: {result.get('message')}")
    return result