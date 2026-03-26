from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict

app = FastAPI()

class Request(BaseModel):
    value: float

@app.get("/")
def root():
    return {"message": "ML service is running"}

@app.post("/predict")
def make_prediction(request: Request):
    result = predict(request.value)
    return {"result": result}