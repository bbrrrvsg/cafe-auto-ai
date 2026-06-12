from fastapi import FastAPI
from app.order.router import router as order_router

app = FastAPI(title="cafe-auto-ai", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(order_router, prefix="/order", tags=["order"])

