from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Iris Classification API")
model = joblib.load("model/iris_model.pkl")

class IrisFeatures(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Iris ML API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: IrisFeatures):
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.DataFrame([data.features], columns=columns)
    prediction = model.predict(df)
    return {"predicted_species": str(prediction[0])}
