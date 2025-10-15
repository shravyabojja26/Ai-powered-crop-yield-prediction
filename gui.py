import requests
import tkinter as tk
from tkinter import messagebox

# Function to send request to FastAPI server
def predict_yield():
    try:
        data = {
            "region": region_options.index(region_var.get()),  # Get index of selected value
            "soil_type": soil_options.index(soil_var.get()),
            "crop": crop_options.index(crop_var.get()),
            "rainfall": float(rainfall_entry.get()),
            "temperature": float(temp_entry.get()),
            "fertilizer_used": fertilizer_options.index(fertilizer_var.get()),
            "irrigation_used": irrigation_options.index(irrigation_var.get()),
            "weather_condition": weather_options.index(weather_var.get()),
            "days_to_harvest": int(days_entry.get()),
        }

        # Send request to FastAPI
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            result_label.config(text=f"Predicted Yield: {result['predicted_yield']} tons/ha", fg="green")
        else:
            result_label.config(text="Error in prediction!", fg="red")

    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong: {e}")

# GUI Setup
root = tk.Tk()
root.title("Crop Yield Prediction")
root.geometry("500x550")

# Dropdown values
region_options = ["North", "East", "South", "West"]
soil_options = ["Clay", "Sandy", "Loam", "Silt", "Peaty", "Chalky"]
crop_options = ["Wheat", "Rice", "Maize", "Barley", "Soybean", "Cotton"]
weather_options = ["Sunny", "Rainy", "Cloudy"]
fertilizer_options = ["No", "Yes"]
irrigation_options = ["No", "Yes"]

# Variables
region_var = tk.StringVar(value=region_options[0])
soil_var = tk.StringVar(value=soil_options[0])
crop_var = tk.StringVar(value=crop_options[0])
fertilizer_var = tk.StringVar(value=fertilizer_options[0])
irrigation_var = tk.StringVar(value=irrigation_options[0])
weather_var = tk.StringVar(value=weather_options[0])

# GUI Layout
tk.Label(root, text="Region:").pack()
tk.OptionMenu(root, region_var, *region_options).pack()

tk.Label(root, text="Soil Type:").pack()
tk.OptionMenu(root, soil_var, *soil_options).pack()

tk.Label(root, text="Crop:").pack()
tk.OptionMenu(root, crop_var, *crop_options).pack()

tk.Label(root, text="Rainfall (mm):").pack()
rainfall_entry = tk.Entry(root)
rainfall_entry.pack()

tk.Label(root, text="Temperature (°C):").pack()
temp_entry = tk.Entry(root)
temp_entry.pack()

tk.Label(root, text="Fertilizer Used?").pack()
tk.OptionMenu(root, fertilizer_var, *fertilizer_options).pack()

tk.Label(root, text="Irrigation Used?").pack()
tk.OptionMenu(root, irrigation_var, *irrigation_options).pack()

tk.Label(root, text="Weather Condition:").pack()
tk.OptionMenu(root, weather_var, *weather_options).pack()

tk.Label(root, text="Days to Harvest:").pack()
days_entry = tk.Entry(root)
days_entry.pack()

# Predict Button
predict_button = tk.Button(root, text="Predict Yield", command=predict_yield)
predict_button.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack()

root.mainloop()
