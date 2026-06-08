from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    ingredientId: int
    dayOfWeek: str
    currentStock: int    
    safetyStock: int     
    rawAmounts: List[int]

class PredictionResponse(BaseModel):
    status: str
    suggestedQty: int
    message: str
    code: str