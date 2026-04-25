import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

os.makedirs('model', exist_ok=True)

# Load the 4-feature dataset
df = pd.read_csv('data/mindmetric_data.csv')

# Standardized Feature Set
FEATURES = ['sleep_hours', 'screen_time', 'study_hours', 'physical_activity']
X = df[FEATURES]
y = df['productivity_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest allows for Explainable AI (XAI) via feature_importances_
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save locally for Edge AI execution
joblib.dump(model, 'model/mindmetric_model.pkl')
print("✅ Edge AI Model trained and saved successfully!")