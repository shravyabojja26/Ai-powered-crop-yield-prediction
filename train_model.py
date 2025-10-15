import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("d:\\RBL\\impl\\crop_yield.csv")

# Data Preprocessing
df.dropna(inplace=True)

# Encode categorical features
categorical_features = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
encoder = LabelEncoder()
for col in categorical_features:
    df[col] = encoder.fit_transform(df[col])

# Feature selection
X = df.drop(columns=['Yield_tons_per_hectare'])
y = df['Yield_tons_per_hectare']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Train XGBoost Regressor
# -------------------------------
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5)
print("XGBoost CV Score:", cv_scores.mean())

xgb_model.fit(X_train_scaled, y_train)

# Predict and evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test_scaled)
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mspe_xgb = np.mean(np.square(((y_test - y_pred_xgb) / y_test))) * 100

# -------------------------------
# Train Decision Tree Regressor
# -------------------------------
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Predict and evaluate Decision Tree
y_pred_dt = dt_model.predict(X_test_scaled)
r2_dt = r2_score(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
mspe_dt = np.mean(np.square(((y_test - y_pred_dt) / y_test))) * 100

# -------------------------------
# Comparison Summary
# -------------------------------
print("\n=== Model Comparison ===")
print(f"{'Metric':<30} {'XGBoost':<15} {'Decision Tree':<15}")
print(f"{'R² Score':<30} {r2_xgb:<15.4f} {r2_dt:<15.4f}")
print(f"{'MAE':<30} {mae_xgb:<15.4f} {mae_dt:<15.4f}")
print(f"{'RMSE':<30} {rmse_xgb:<15.4f} {rmse_dt:<15.4f}")
print(f"{'MSPE':<30} {mspe_xgb:<15.2f}% {mspe_dt:<15.2f}%")

# Save both models and scaler
joblib.dump(xgb_model, "d:\\RBL\\impl\\xgb_model.pkl")
joblib.dump(dt_model, "d:\\RBL\\impl\\dt_model.pkl")
joblib.dump(scaler, "d:\\RBL\\impl\\scaler.pkl")

# Optional: Save visualizations
# Feature importance for XGBoost
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_importance)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("d:\\RBL\\impl\\xgb_feature_importance.png")
plt.close()
