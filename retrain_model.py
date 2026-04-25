import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. LOAD BIOLOGICALLY-CONSISTENT DATA ──────────────────────────
# Ensure you have run the updated dataset.py before this!
if not os.path.exists('data/mindmetric_data.csv'):
    print("❌ Error: data/mindmetric_data.csv not found.")
    exit()

df = pd.read_csv('data/mindmetric_data.csv')

# Syncing with the 4 core features used in your UI
FEATURES = ['sleep_hours', 'screen_time', 'study_hours', 'physical_activity']
TARGET   = 'productivity_score'

X = df[FEATURES].values
y = df[TARGET].values

# ── 2. TRAIN / TEST SPLIT ─────────────────────────────────────────
# Using a fixed random_state ensures reproducible results for your demo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. TRAIN THE RANDOM FOREST ────────────────────────────────────
# We use RandomForest because it handles non-linear biological relationships 
# (like how productivity crashes if you sleep too much OR too little).
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── 4. EVALUATE PERFORMANCE ───────────────────────────────────────
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("📊 Model Evaluation:")
print(f"   MAE : {mae:.2f} (Average error in productivity points)")
print(f"   R²  : {r2:.4f} (Accuracy score, 1.0 is perfect)")

# ── 5. SAVE FOR EDGE DEPLOYMENT ───────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/mindmetric_model.pkl')

print("\n✅ Model saved to model/mindmetric_model.pkl")
print(f"   Features Learned: {FEATURES}")

# ── 6. VERIFY BIOLOGICAL LOGIC ────────────────────────────────────
# Testing a healthy day vs. a sleep-deprived day
healthy_day = [[8.0, 2.0, 4.0, 45]] # 8h sleep
tired_day   = [[4.0, 6.0, 4.0, 15]] # 4h sleep

h_pred = model.predict(healthy_day)[0]
t_pred = model.predict(tired_day)[0]

print(f"\n🧪 Logic Test:")
print(f"   Predicted Score (Healthy Day): {h_pred:.1f}")
print(f"   Predicted Score (Tired Day):   {t_pred:.1f}")
print(f"   Sleep Impact Delta: {h_pred - t_pred:.1f} points")