# cafe-auto-ai

카페 자동화 시스템의 AI 서비스 (FastAPI).

## 폴더 구조

```
cafe-auto-ai/
├── main.py                  # FastAPI 진입점 (라우터 마운트)
├── requirements.txt         # 파이썬 의존성
├── .gitignore
│
├── app/
│   ├── inventory/           # 재고 담당
│   │   ├── router.py        # 엔드포인트 (/inventory/...)
│   │   ├── schema.py        # 요청/응답 Pydantic 모델
│   │   ├── data.py          # 데이터 로드/전처리
│   │   └── model.py         # ML 학습/예측
│   ├── order/               # 발주 담당 (동일 구조)
│   ├── review/              # 리뷰 담당 (동일 구조)
│   ├── anomaly/             # 이상치·장애감지 담당 (동일 구조)
│   └── common/              # 공통 코드
│       ├── config.py        # 환경변수, 공통 설정
│       └── utils.py         # 공용 유틸
│
└── models/                  # 학습된 모델 산출물 (git 제외)
    ├── inventory/
    ├── order/
    ├── review/
    └── anomaly/
```

## 도메인 분담 / URL

| 도메인 | 폴더 | URL prefix |
|---|---|---|
| 재고 | `app/inventory/` | `/inventory/...` |
| 발주 | `app/order/` | `/order/...` |
| 리뷰 | `app/review/` | `/review/...` |
| 이상치·장애감지 | `app/anomaly/` | `/anomaly/...` |

각자 **본인 도메인 폴더**와 `models/<도메인>/`만 작업. `common/`은 공통 코드 생길 때만 합의 후 추가.

## 셋업

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload
```

- 서버: http://localhost:8000
- API 문서 (Swagger): http://localhost:8000/docs
- API 문서 (ReDoc): http://localhost:8000/redoc

## 의존성 추가 시

```bash
pip install <패키지>
pip freeze > requirements.txt
```

`venv/`는 git에 올리지 않음. 팀원은 `pip install -r requirements.txt`로 각자 설치.

## 주의

- 파일명을 라이브러리·표준모듈과 같게 짓지 말 것 (`fastapi.py`, `json.py`, `email.py`, `test.py` 등 ❌)
- `app/`, `app/<도메인>/` 안의 `__init__.py`는 빈 파일이라도 그대로 둘 것 (패키지 인식용)
