"""
Insurance Cost Prediction API
============================
FastAPI app لخدمة الموديل المدرّب.

التشغيل:
    pip install fastapi uvicorn joblib pandas scikit-learn
    uvicorn app:app --reload

الاستخدام:
    POST http://127.0.0.1:8000/predict
    {
        "age": 35, "sex": "male", "bmi": 28.5,
        "children": 2, "smoker": "no", "region": "southeast"
    }
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Insurance Cost Prediction API")

# تحميل الموديل عند بدء التطبيق
model = joblib.load("insurance_model.pkl")


class Person(BaseModel):
    age: int
    sex: str         # "male" or "female"
    bmi: float
    children: int
    smoker: str      # "yes" or "no"
    region: str      # "southwest" / "southeast" / "northwest" / "northeast"


@app.get("/")
def home():
    return {
        "message": "Insurance Cost Prediction API",
        "usage": "POST /predict with person data",
        "example": {
            "age": 35, "sex": "male", "bmi": 28.5,
            "children": 2, "smoker": "no", "region": "southeast"
        }
    }


@app.post("/predict")
def predict(person: Person):
    df = pd.DataFrame([person.model_dump()])
    prediction = float(model.predict(df)[0])
    return {
        "predicted_charges_usd": round(prediction, 2),
        "input": person.model_dump()
    }
