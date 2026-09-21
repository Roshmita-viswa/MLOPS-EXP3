import joblib
import pandas as pd

model = joblib.load("model/iris_model.pkl")

def predict_iris(features):
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    data = pd.DataFrame([features], columns=columns)
    prediction = model.predict(data)
    return prediction[0]
