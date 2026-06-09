import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.order.schema import PredictionResponse

# 가중치 난수 고정 (재현성 확보)
torch.manual_seed(42)
np.random.seed(42)


# =========================
# 1. 파이토치 LSTM 모델 아키텍처
# =========================
class StockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4,   # Feature 차원: 순수소비량, 입고량, 이동평균선, 요일
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =========================
# 2. 핵심 발주 예측 연산 함수
# =========================
def calculate_order_prediction(
    ingredient_id: int,
    day_of_week: str,
    current_stock: int,
    safety_stock: int,
    raw_amounts: list
) -> PredictionResponse:

    try:
        # =========================
        # 🛡️ 0. 시계열 방향 정합성 교정
        # =========================
        # 자바가 보낸 최신순(DESC) 배열을 파이썬 시계열 학습용 정방향(ASC: 과거->현재)으로 즉시 반전
        raw_amounts = list(reversed(raw_amounts))

        # =========================
        # 1. 다차원 데이터 분리 (소비량 vs 입고량)
        # =========================
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

        # 최소 학습 데이터 규격 미달 시 Fallback 처리
        if len(consumption) < 30:
            return PredictionResponse(
                status="AI_PREDICT",
                suggestedQty=max(0, safety_stock - current_stock),
                message="데이터 부족 → 안전재고 fallback",
                code="MINIMUM_BUFFER"
            )

        # =========================
        # 2. 넘파이 텐서 변환
        # =========================
        consumption = np.array(consumption, dtype=np.float32)
        restock = np.array(restock, dtype=np.float32)

        # =========================
        # 3. 이상치 완화 처리 (Clip Outliers)
        # =========================
        q95 = np.percentile(consumption, 95)
        consumption = np.clip(consumption, 0, q95)

        # =========================
        # 4. 데이터 스케일 완화 (Log Scaling)
        # =========================
        consumption = np.log1p(consumption)
        restock = np.log1p(restock)

        # =========================
        # 5. Z-Score 표준 정규화 (Standardization)
        # =========================
        mean = consumption.mean()
        std = consumption.std() + 1e-6
        consumption = (consumption - mean) / std
        
        # 🛡️ 미래 예측 루프 오염 방지용: 입고량 '0'이 표준화 시스템 속에서 가지는 진짜 텐서 값 역산
        normalized_zero_restock = (0.0 - restock.mean()) / (restock.std() + 1e-6)

        # =========================
        # 6. 파동 추세 Feature 엔지니어링
        # =========================
        # 7일 주기 이동평균선(Moving Average) 산출
        moving_avg = np.convolve(consumption, np.ones(7)/7, mode='same')

        # 요일 주기성 정규화 매핑 (0.0 ~ 1.0)
        day_index = {
            "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2,
            "THURSDAY": 3, "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6
        }
        dow = day_index.get(day_of_week, 0)
        dow_feature = np.array([dow / 6.0] * len(consumption))

        # =========================
        # 7. LSTM 입력 윈도우 슬라이싱 (Window Size = 7)
        # =========================
        window_size = 7
        X, y = [], []

        for i in range(len(consumption) - window_size):
            X.append([
                [
                    consumption[j],
                    restock[j],
                    moving_avg[j],
                    dow_feature[j]
                ]
                for j in range(i, i + window_size)
            ])
            y.append(consumption[i + window_size])

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # =========================
        # 8. 검증 데이터셋 분할 (Train 80% / Test 20%)
        # =========================
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # =========================
        # 9. 신경망 역전파 가중치 학습 (300 Epochs)
        # =========================
        model = StockLSTM()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(300):
            optimizer.zero_grad()
            output = model(X_train).squeeze()
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        # =========================
        # 10. 모델 검증 및 AI 성적표 산출
        # =========================
        model.eval()
        with torch.no_grad():
            pred = model(X_test).squeeze().numpy()
            actual = y_test.numpy()

        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)

        print(f"\n📈 [AI 성능 평가 - 자재 ID: {ingredient_id}]")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R²   : {r2:.4f} ({r2*100:.1f}%)")

        # =========================
        # 11. Multi-step 미래 재귀 예측 (향후 3일 누적 소모량 탐색)
        # =========================
        future_days = 3
        preds = []

        # 가장 최신 시점의 정방향 7일 정규화 데이터 추출
        window = X[-1].numpy().tolist()

        with torch.no_grad():
            for k in range(future_days):
                input_tensor = torch.tensor([window], dtype=torch.float32)
                p = model(input_tensor).item()
                preds.append(p)

                # 🛡️ 윈도우 스케일 오염 차단: 최근 6개 소모량 텐서와 방금 만든 예측치(p) 기반 이평선 재계산
                recent_consumptions = [hist[0] for hist in window[1:]] + [p]
                next_moving_avg = sum(recent_consumptions) / 7.0
                
                # 미래 가상 요일 인덱스 변환
                next_dow = (dow + k + 1) % 7

                window.pop(0)
                # 오염된 원본 생짜 '0' 대신, 완벽히 정규화 필터링된 데이터 아키텍처 주입
                window.append([p, normalized_zero_restock, next_moving_avg, next_dow / 6.0])

        predicted = sum(preds)

        # =========================
        # 12. 수학적 역변환 (Inverse Scaling)
        # =========================
        predicted = (predicted * std) + mean
        predicted = np.expm1(predicted)
        predicted = max(0, predicted)

        # =========================
        # 13. 저성능 데이터 보정 방어벽 (Smart Fallback)
        # =========================
        if r2 < 0.5:
            # 설명력이 부족할 경우 최근 1주일 통계학 평균값으로 긴급 대체
            fallback = np.mean(consumption[-7:])
            fallback = (fallback * std) + mean
            fallback = np.expm1(fallback)

            predicted = fallback * future_days

        predicted = int(np.ceil(predicted))

        # =========================
        # 14. 규격 단위 동기화 및 최종 발주량 제안
        # =========================
        # 자재 ID가 1번(원두), 4번(우유)일 때만 환산 계수를 1000으로 세팅 (g/ml -> 봉/팩)
        unit_factor = 1000 if ingredient_id in [1, 4] else 1

        buffer = int(predicted * 0.2)
        display_predicted = int(np.ceil(predicted / unit_factor))
        display_buffer = int(np.ceil(buffer / unit_factor))

        input_safety_stock = int(np.ceil(safety_stock / unit_factor))
        input_current_stock = int(np.ceil(current_stock / unit_factor))

        # 팩/봉 규격 단위가 완전히 일치된 클린 상태에서 최종 수식 계산
        suggested = display_predicted + input_safety_stock + display_buffer - input_current_stock

        # 매장 보관 상한 제한 (과다발주 방지벽: 환산된 안전재고의 최대 3배까지만 허용)
        suggested = min(suggested, input_safety_stock * 3)
        suggested = max(0, suggested)

        return PredictionResponse(
            status="AI_PREDICT",
            suggestedQty=suggested,
            message=f"[AI] 3일 예측({display_predicted}개) / R²={r2:.2f}",
            code="AUTO_ANALYSIS"
        )

    except Exception as e:
        return PredictionResponse(
            status="AI_ERROR",
            suggestedQty=0,
            message=str(e),
            code="SYSTEM_FAULT"
        )