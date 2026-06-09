def process_received_logs(raw_amounts: list) -> list:
    print(f"📥 [data.py] 자바가 보내준 원본 정수 데이터 수거 완료: {raw_amounts}")
    
    # 🌟 음수 차감 기입(소모/폐기) 데이터를 abs()를 씌워 실제 정수 총량으로 변환!
    cleaned_data = [abs(amt) for amt in raw_amounts if amt < 0]
    
    print(f"🧹 [data.py] 절대값 정제 완료: {cleaned_data}")
    return cleaned_data