import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from app.order.schema import PredictionResponse

#  미래 소모량 추세를 학습할 다층 인공신경망(ANN) 정의
class StockPredictorNN(nn.Module):
    def __init__(self):
        super(StockPredictorNN, self).__init__()
        # 입력(시간축 1개) -> 은닉층(8개 노드) -> 출력(예측 소모량 1개)
        self.linear_layers = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),       # 비선형 패턴 학습을 위한 활성화 함수
            nn.Linear(8, 1)
        )
        
    def forward(self, x):
        return self.linear_layers(x)


def calculate_order_prediction(ingredient_id: int, day_of_week: str, current_stock: int, safety_stock: int, raw_amounts: list) -> PredictionResponse:
    try:
        # 자바가 준 원본 데이터를 절대값(양수 소모량)으로 정제
        cleaned_data = [abs(int(amt)) for amt in raw_amounts]
        
        # 딥러닝 학습을 위한 최소 데이터 방어 코드 (로그가 3개 미만일 때)
        if len(cleaned_data) < 3:
            suggested_qty = max(0, safety_stock - current_stock)
            return PredictionResponse(
                status="AI_PREDICT",
                suggestedQty=suggested_qty,
                message="[PyTorch] 데이터 부족으로 최소 안전 재고 수량만 발주합니다.",
                code="MINIMUM_BUFFER"
            )
            
        # 파이토치 텐서(Tensor) 데이터셋 변환 (X: 시간 흐름 인덱스, Y: 실제 소모량)
        X_data = np.arange(len(cleaned_data)).reshape(-1, 1).astype(np.float32)
        y_data = np.array(cleaned_data).reshape(-1, 1).astype(np.float32)
        
        X_tensor = torch.from_numpy(X_data)
        y_tensor = torch.from_numpy(y_data)
        
        # 딥러닝 모델 및 손실함수, 최적화 알고리즘 세팅
        model = StockPredictorNN()
        criterion = nn.MSELoss() # 평균제곱오차 손실함수
        optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam 옵티마이저
        
        # 실시간 경량 학습 (100번 돌며 추세 피팅)
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        # 4. 다음 타임스텝(미래)의 소모량 예측 연산
        model.eval()
        with torch.no_grad():
            next_step = torch.tensor([[float(len(cleaned_data))]], dtype=torch.float32)
            predicted_consume = model(next_step).item()
            
        # 예측값이 오차로 인해 음수가 나오는 것을 방지
        predicted_consume = max(0.0, predicted_consume)
        
        # 주말 요일 버프 가중치 반영 (토/일은 소모량 25% 가산)
        weight = 1.25 if day_of_week in ["SATURDAY", "SUNDAY"] else 1.0
        final_predicted_consume = int(np.ceil(predicted_consume * weight))
        
        # 🌟 [단위 버그 완전 격리] 자재 ID가 1번(원두), 4번(우유)일 때만 환산 계수를 1000으로 고정
        unit_factor = 1000 if ingredient_id in [1, 4] else 1
        
        # 원본 소모량(g, ml)을 자바 화면 규격(봉, 팩, 개) 단위로 올림 가공
        display_predicted_consume = int(np.ceil(final_predicted_consume / unit_factor))
        
        # 추천 발주량(개) = 예측 소모량(개) + 안전재고(개) - 현재고(개)
        suggested_qty = (display_predicted_consume + safety_stock) - current_stock
        
        if suggested_qty < 0:
            suggested_qty = 0
            
        return PredictionResponse(
            status="AI_PREDICT",
            suggestedQty=suggested_qty, # 이제 팩/개 단위 수치가 자바로 바로 넘어감!
            message=f"[PyTorch 딥러닝] 다음 주기 예측 소모({display_predicted_consume}개) 반영 완료.",
            code="AUTO_ANALYSIS"
        )
        
    except Exception as e:
        return PredictionResponse(
            status="AI_ERROR",
            suggestedQty=0,
            message=f"파이썬 파이토치 연산 장애 발생: {str(e)}",
            code="SYSTEM_FAULT"
        )