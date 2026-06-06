import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from app.order.schema import PredictionResponse

def calculate_order_prediction(ingredient_id: int, day_of_week: str, current_stock: int, safety_stock: int, raw_amounts: list) -> PredictionResponse:
    try:
        # 자바가 준 원본 데이터를 절대값(양수 소모량)으로 전처리
        cleaned_data = [abs(int(amt)) for amt in raw_amounts]
        
        # 머신러닝 최소 학습 데이터 방어 코드 (로그가 너무 부족할 때)
        if len(cleaned_data) < 3:
            suggested_qty = max(0, safety_stock - current_stock)
            return PredictionResponse(
                status="AI_PREDICT",
                suggestedQty=suggested_qty,
                message="[ML] 로그 데이터 부족으로 최소 안전 재고 충족 수량만 발주합니다.",
                code="MINIMUM_BUFFER"
            )
        
        #  Scikit-learn 학습용 시계열 구조 생성 (X: 시간 흐름 인덱스, Y: 소모량)
        df = pd.DataFrame({
            "time_step": np.arange(len(cleaned_data)), # 0, 1, 2, 3...
            "amount": cleaned_data
        })
        
        X = df[["time_step"]]
        y = df["amount"]
        
        #  선형 회귀 머신러닝 모델 학습 (Trend Fitting)
        model = LinearRegression()
        model.fit(X, y)
        
        # 다음 타임스텝(미래)의 소모량 예측 연산
        next_step = np.array([[len(cleaned_data)]])
        predicted_consume = model.predict(next_step)[0]
        
        # 주말 요일 버프 가중치 반영 (토/일은 소모량 25% 가산)
        weight = 1.25 if day_of_week in ["SATURDAY", "SUNDAY"] else 1.0
        final_predicted_consume = int(np.ceil(predicted_consume * weight))
        
        # [최종 AI 발주량 연산 공식]
        # 추천 발주량 = (머신러닝 예측 소모량 + 자바가 넘겨준 실제 안전재고) - 현재 매장 실재고
        suggested_qty = (final_predicted_consume + safety_stock) - current_stock
        
        # 계산 수치가 음수(-)가 나오면 매장에 재고가 낭비되고 있다는 뜻이므로 발주량 0 고정
        if suggested_qty < 0:
            suggested_qty = 0
            
        return PredictionResponse(
            status="AI_PREDICT",
            suggestedQty=suggested_qty,
            message=f"[ML 선형회귀] 다음 주기 예측 소모({final_predicted_consume}개) + 안전재고 반영 완료.",
            code="AUTO_ANALYSIS"
        )
        
    except Exception as e:
        return PredictionResponse(
            status="AI_ERROR",
            suggestedQty=0,
            message=f"파이썬 머신러닝 연산 장애 발생: {str(e)}",
            code="SYSTEM_FAULT"
        )