from app.main import app

application = app

if __name__ == "__main__":
    import uvicorn
    # 로컬 테스트 시 구동 엔진 주소를 application으로 변경
    uvicorn.run("application:application", host="0.0.0.0", port=8000, reload=True)