from fastapi import FastAPI

from app.anomaly.router import router as anomaly_router
from app.inventory.router import router as inventory_router
from app.order.router import router as order_router
from app.review.router import router as review_router

app = FastAPI(title="cafe-auto-ai", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(inventory_router, prefix="/inventory", tags=["inventory"])
app.include_router(order_router, prefix="/order", tags=["order"])
app.include_router(review_router, prefix="/review", tags=["review"])
app.include_router(anomaly_router, prefix="/anomaly", tags=["anomaly"])
