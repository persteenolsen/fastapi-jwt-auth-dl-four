import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from train import FEATURES, HouseModel, build_features, make_input

# Note: Run the test by the command:
# python -m tests.testfour
# Expected output:
# 🚀 Loading dataset...
# 🚀 Loading trained model...
# 🔎 Top 5 largest prediction errors (test set):
# Predicted=$345,000, Actual=$450,000, Error=$105,000
# Predicted=$280,000, Actual=$360,000, Error=$80,000
# Predicted=$200,000, Actual=$270,000, Error=$70,000
# Predicted=$190,000, Actual=$250,000, Error=$60,000
# Predicted=$150,000, Actual=$210,000, Error=$60,000
# 📊 Average percentage error: 6.25%
# ✅ No extreme prediction errors

# =====================================================
# LOAD DATASET
# =====================================================
print("🚀 Loading dataset...")
df = pd.read_csv("AmesHousing.csv")

# rename columns to match FEATURES
df = df.rename(columns={
    "Gr Liv Area": "Gr_Liv_Area",
    "Overall Qual": "Overall_Qual",
    "Year Built": "Year_Built",
    "Garage Cars": "Garage_Cars",
    "Full Bath": "Full_Bath",
    "Bedroom AbvGr": "Bedroom_AbvGr",
    "Lot Area": "Lot_Area"
})

df = df.apply(build_features, axis=1)
df = df[FEATURES + ["SalePrice"]].dropna()

# =====================================================
# PREPROCESS FEATURES
# =====================================================
X = df[FEATURES].values.astype(np.float32)
y = np.log1p(df["SalePrice"].values.astype(np.float32))

mean = np.load("mean.npy")
std = np.load("std.npy")

class DummyScaler:
    def __init__(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

scaler = DummyScaler(mean, std)
X_scaled = (X - scaler.mean_) / scaler.scale_

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
split = int(0.8 * len(X_scaled))
X_train = torch.tensor(X_scaled[:split], dtype=torch.float32)
y_train = torch.tensor(y[:split], dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_scaled[split:], dtype=torch.float32)
y_test = torch.tensor(y[split:], dtype=torch.float32).view(-1, 1)

# =====================================================
# LOAD MODEL
# =====================================================
print("🚀 Loading trained model...")
model = HouseModel(len(FEATURES))
model.load_state_dict(torch.load("model.pth"))
model.eval()

# =====================================================
# CALCULATE PREDICTIONS & ERRORS
# =====================================================
with torch.no_grad():
    y_pred_log = model(X_test).squeeze().numpy()
    y_test_log = y_test.squeeze().numpy()

# Convert from log back to original scale
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test_log)

# Absolute errors
errors = np.abs(y_true - y_pred)

# =====================================================
# TOP 5 LARGEST ERRORS
# =====================================================
top5_idx = np.argsort(errors)[-5:][::-1]  # descending order

print("\n🔎 Top 5 largest prediction errors (test set):")
for idx in top5_idx:
    print(f"Predicted=${y_pred[idx]:,.0f}, Actual=${y_true[idx]:,.0f}, Error=${errors[idx]:,.0f}")

# =====================================================
# AVERAGE PERCENTAGE ERROR
# =====================================================
percentage_errors = errors / y_true * 100
avg_pct_error = np.mean(percentage_errors)

print(f"\n📊 Average percentage error: {avg_pct_error:.2f}%")

# =====================================================
# OPTIONAL ALERT FOR EXTREME ERRORS
# =====================================================
extreme_threshold = 30  # e.g., errors >30% are extreme
num_extreme = np.sum(percentage_errors > extreme_threshold)

if num_extreme > 0:
    print(f"⚠️ {num_extreme} houses have >{extreme_threshold}% prediction error")
else:
    print("✅ No extreme prediction errors")