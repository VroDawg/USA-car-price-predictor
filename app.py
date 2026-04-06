from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('car_price_model.pkl')
df_ref = pd.read_csv('vehicles_clean.csv')

cat_cols = ['manufacturer', 'model', 'fuel', 'title_status',
            'transmission', 'drive', 'type', 'paint_color', 'state']

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(df_ref[col])
    encoders[col] = le

class CarFeatures(BaseModel):
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    drive: str
    type: str
    paint_color: str
    state: str

@app.get("/")
def root():
    return {"message": "Car price predictor is running"}

@app.post("/predict")
def predict(car: CarFeatures):
    input_dict = car.dict()
    
    for col in cat_cols:
        val = input_dict[col]
        if val in encoders[col].classes_:
            input_dict[col] = encoders[col].transform([val])[0]
        else:
            input_dict[col] = 0
    
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    
    return {"predicted_price": round(float(prediction), 2)}
    