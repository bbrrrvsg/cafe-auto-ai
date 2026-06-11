import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.order.schema import PredictionResponse

torch.manual_seed(42)
np.random.seed(42)

MODEL_DIR = os.path.join("app", "order", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# 1. 파이토치 LSTM 모델 아키텍처
# =========================
class StockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4,   # Feature: 소비량, 입고량, 이동평균선, 요일
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# 배치 정기 학습 함수 (매달 1회 스케줄러에 의해 호출됨)

def train_stock_model(ingredient_id: int, raw_amounts: list) -> dict:
    try:
        raw_amounts = list(reversed(raw_amounts))
        consumption = []
        restock = []

        for amt in raw_amounts:
            amt = int(amt)
            if amt < 0:
                consumption.append(abs(amt))
                restock.append(0)
            else:
                consumption.append(0)
                restock.append(amt)

        if len(consumption) < 30:
            return {"status": "SKIPPED", "message": "데이터 부족으로 학습 생략"}

        consumption = np.array(consumption, dtype=np.float32)
        restock = np.array(restock, dtype=np.float32)

        # 이상치 및 스케일 변환 가중치(Mean, Std) 산출
        q95 = np.percentile(consumption, 95)
        clipped_consumption = np.clip(consumption, 0, q95)
        
        log_consumption = np.log1p(clipped_consumption)
        log_restock = np.log1p(restock)

        mean = log_consumption.mean()
        std = log_consumption.std() + 1e-6
        
        norm_consumption = (log_consumption - mean) / std
        moving_avg = np.convolve(norm_consumption, np.ones(7)/7, mode='same')

        # 임의의 요일 가중치 생성 (학습용 구조 매칭)
        dow_feature = np.array([0.5] * len(norm_consumption))

        window_size = 7
        X, y = [], []
        for i in range(len(norm_consumption) - window_size):
            X.append([[norm_consumption[j], log_restock[j], moving_avg[j], dow_feature[j]] for j in range(i, i + window_size)])
            y.append(norm_consumption[i + window_size])

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # 모델 학습 진행 (300 Epochs)
        model = StockLSTM()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(300):
            optimizer.zero_grad()
            output = model(X).squeeze()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # 학습 완료된 가중치 파일과 정규화 스케일 파라미터를 딕셔너리로 묶어 디스크 저장
        model_meta = {
            "state_dict": model.state_dict(),
            "mean": float(mean),                 # float()로 순수 파이썬 타입 변환
            "std": float(std),                   # float()로 변환
            "q95": float(q95),                   # float()로 변환
            "restock_mean": float(log_restock.mean()),
            "restock_std": float(log_restock.std() + 1e-6)
        }
        torch.save(model_meta, os.path.join(MODEL_DIR, f"model_{ingredient_id}.pt"))
        
        return {"status": "SUCCESS", "message": f"자재 {ingredient_id} 모델 저장 완료"}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}



# 실시간 추론 함수

def calculate_order_prediction(
    ingredient_id: int,
    day_of_week: str,
    current_stock: int,
    safety_stock: int,
    raw_amounts: list
) -> PredictionResponse:

    try:
        model_path = os.path.join(MODEL_DIR, f"model_{ingredient_id}.pt")
        
        # 모델 파일이 없을 경우 최소 안전재고 가이드라인으로 Fallback 대안 작동
        if not os.path.exists(model_path):
            return PredictionResponse(
                status="AI_PREDICT",
                suggestedQty=max(0, safety_stock - current_stock),
                message="기본 모델 미존재 -> 안전재고 대체 가이드 작동",
                code="MINIMUM_BUFFER"
            )

        # 저장된 가중치 파일 및 스케일 메타 데이터 로드
        model_meta = torch.load(model_path)
        mean = model_meta["mean"]
        std = model_meta["std"]
        q95 = model_meta["q95"]
        restock_mean = model_meta["restock_mean"]
        restock_std = model_meta["restock_std"]

        model = StockLSTM()
        model.load_state_dict(model_meta["state_dict"])
        model.eval()

        # 실시간 수집된 7일 데이터 정규화 가공
        raw_amounts = list(reversed(raw_amounts))[-7:]
        consumption = [abs(int(x)) if int(x) < 0 else 0 for x in raw_amounts]
        
        consumption = np.array(consumption, dtype=np.float32)
        consumption = np.clip(consumption, 0, q95)
        consumption = np.log1p(consumption)
        consumption = (consumption - mean) / std

        moving_avg = np.convolve(consumption, np.ones(7)/7, mode='same')
        
        day_index = {"MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3, "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6}
        dow = day_index.get(day_of_week, 0)

        # 윈도우 빌딩
        window = []
        normalized_zero_restock = (0.0 - restock_mean) / restock_std
        for i in range(7):
            window.append([consumption[i], normalized_zero_restock, moving_avg[i], dowe_feat := (dow / 6.0)])

        # 미래 3일 초고속 재귀 예측 루프 진행
        future_days = 3
        preds = []
        with torch.no_grad():
            for k in range(future_days):
                input_tensor = torch.tensor([window], dtype=torch.float32)
                p = model(input_tensor).item()
                preds.append(p)

                recent_consumptions = [hist[0] for hist in window[1:]] + [p]
                next_moving_avg = sum(recent_consumptions) / 7.0
                next_dow = (dow + k + 1) % 7

                window.pop(0)
                window.append([p, normalized_zero_restock, next_moving_avg, next_dow / 6.0])

        predicted = sum(preds)
        predicted = (predicted * std) + mean
        predicted = np.expm1(predicted)
        predicted = max(0, predicted)

        # 단위 보정 및 발주량 산출
        buffer = int(predicted * 0.2)
        display_predicted = int(np.ceil(predicted))
        display_buffer = int(np.ceil(buffer))

        suggested = display_predicted + safety_stock + display_buffer - current_stock
        suggested = min(suggested, safety_stock * 3)
        suggested = max(0, suggested)

        return PredictionResponse(
            status="AI_PREDICT",
            suggestedQty=suggested,
            message=f"[AI] 저장된 모델 기반 예측 완료 / 3일 소모 예상: {display_predicted}",
            code="AUTO_ANALYSIS"
        )

    except Exception as e:
        return PredictionResponse(status="AI_ERROR", suggestedQty=0, message=str(e), code="SYSTEM_FAULT")