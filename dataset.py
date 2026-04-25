import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)
np.random.seed(42)
data_size = 500

# ─── BIOLOGICAL DATA GENERATION ───
sleep = np.random.uniform(5, 9, data_size)
# Study and Screen time must fit in the remaining hours
remaining = 24 - sleep
study = np.random.uniform(0, remaining * 0.6, data_size)
screen = np.random.uniform(0, (remaining - study), data_size)
activity = np.random.uniform(0, 120, data_size)

# ─── WEIGHTED SCORING (Sleep is Priority) ───
# Productivity crashes if sleep is too low (<6h) or too high (>9.5h)
sleep_quality_factor = np.where((sleep >= 7) & (sleep <= 8.5), 1.2, 0.8)

productivity = (
    (sleep * 12 * sleep_quality_factor) + # Sleep is the highest weight
    (study * 8) - 
    (screen * 6) + 
    (activity * 0.1) + 
    np.random.normal(0, 5, data_size)
)

productivity = np.clip(productivity, 0, 100)

df = pd.DataFrame({
    'sleep_hours': sleep, 'screen_time': screen, 
    'study_hours': study, 'physical_activity': activity,
    'productivity_score': productivity
})

df.to_csv('data/mindmetric_data.csv', index=False)
print("✅ Biologically-Consistent Dataset Created.")