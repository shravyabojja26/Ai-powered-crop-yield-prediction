from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load trained model and scaler
with open("crop_yield_model.pkl", "rb") as f:
    model = joblib.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = joblib.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define request model
class CropInput(BaseModel):
    region: int  # 0=North, 1=East, 2=South, 3=West
    soil_type: int  # 0=Clay, 1=Sandy, 2=Loam, 3=Silt, 4=Peaty, 5=Chalky
    crop: int  # 0=Wheat, 1=Rice, 2=Maize, 3=Barley, 4=Soybean, 5=Cotton
    rainfall: float  # in mm
    temperature: float  # in °C
    fertilizer_used: int  # 0=No, 1=Yes
    irrigation_used: int  # 0=No, 1=Yes
    weather_condition: int  # 0=Sunny, 1=Rainy, 2=Cloudy
    days_to_harvest: int

@app.post("/predict")
def predict_yield(data: CropInput):
    # Convert input to NumPy array
    input_features = np.array([[
        data.region, data.soil_type, data.crop, data.rainfall, 
        data.temperature, data.fertilizer_used, data.irrigation_used, 
        data.weather_condition, data.days_to_harvest
    ]])
    
    # Apply scaling
    input_features_scaled = scaler.transform(input_features)
    
    # Predict yield
    predicted_yield = model.predict(input_features_scaled)[0]
    
    return {"predicted_yield": f"{predicted_yield:.2f} tons per hectare"}
