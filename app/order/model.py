import os
import pickle
import math
from app.order.data import process_received_logs

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "order")
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "order_model.pkl")

def train_and_save_order_model(ingredient_id: int, cleaned_data: list, anomaly_threshold: float):
    trained_baseline = {
        "ingredient_id": ingredient_id,
        "anomaly_threshold": anomaly_threshold,
        "historical_sample_size": len(cleaned_data),
        "last_trained_data": cleaned_data
    }
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    with open(MODEL_FILE_PATH, "wb") as f:
        pickle.dump(trained_baseline, f)

def calculate_order_prediction(ingredient_id: int, day_of_week: str, raw_amounts: list) -> dict:
    try:
        # 1️⃣ [Data Layer] 자바가 준 데이터를 정제 함수로 토스
        cleaned_data = process_received_logs(raw_amounts)
        
        if not cleaned_data:
            return {"status": "AI_PREDICT", "suggestedQty": 10, "message": "⚠️ 데이터 부족으로 기본 안전재고 제안", "code": "AUTO_ANALYSIS"}

        # 2️⃣ [통계 분석 및 이상치 판별]
        mean_value = sum(cleaned_data) / len(cleaned_data)
        anomaly_threshold = max(mean_value * 3.0, 30.0) # 동적 임계치
        
        is_anomaly_detected = any(amt > anomaly_threshold for amt in cleaned_data)
        
        if is_anomaly_detected:
            return {
                "status": "AI_ERROR",
                "suggestedQty": 10, # 시스템 보호 동결
                "message": f"⚠️ [이상치 경보] 임계치({anomaly_threshold:.1f}개) 초과 데이터 감지. 시스템 동결.",
                "code": "SYSTEM_FREEZE"
            }

        
        train_and_save_order_model(ingredient_id, cleaned_data, anomaly_threshold)

        
        suggested_qty = int(math.ceil(mean_value * 1.2))
        
        return {
            "status": "AI_PREDICT",
            "suggestedQty": suggested_qty,
            "message": f"🎉 자바 연동 데이터 기반 최적 발주 권장량 산출 완료.",
            "code": "AUTO_ANALYSIS"
        }
        
    except Exception as e:
        return {"status": "AI_ERROR", "suggestedQty": 10, "message": f"❌ 내부 오류: {str(e)}", "code": "SYSTEM_FREEZE"}