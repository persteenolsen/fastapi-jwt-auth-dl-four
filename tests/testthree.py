import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from train import FEATURES, HouseModel, build_features, make_input

# Note: Run the test by the command:
# python -m tests.testhree
# Expected output:
# 🚀 Loading dataset...
# 🚀 Loading trained model...
# 📊 Train Loss: 0.086
# 📊 Test Loss: 0.096
# ✅ Model fit looks good

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
# LOSS FUNCTION
# =====================================================
loss_fn = nn.MSELoss()

# =====================================================
# CALCULATE TRAIN & TEST LOSS
# =====================================================
with torch.no_grad():
    train_loss = loss_fn(model(X_train), y_train).item()
    test_loss = loss_fn(model(X_test), y_test).item()

print("\n📊 Train Loss:", train_loss)
print("📊 Test Loss:", test_loss)

# =====================================================
# SIMPLE OVER/UNDERFITTING CHECK
# =====================================================
if train_loss < 0.1 and test_loss > 0.2:
    print("⚠️ Warning: Model might be overfitting")
elif train_loss > 0.2:
    print("⚠️ Warning: Model might be underfitting")
else:
    print("✅ Model fit looks good")