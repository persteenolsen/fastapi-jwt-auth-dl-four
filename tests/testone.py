import numpy as np
import torch
import onnxruntime as ort
from train import AmesDataset, HouseModel, build_features, make_input, FEATURES

# Note: Run the test by the command:
# python -m tests.testone
# Expected output:
# 🔎 Manual predictions (PyTorch vs ONNX):
# Gr_Liv_Area=900 -> PyTorch: $182,959, ONNX: $182,959, diff=0.00000000
# Gr_Liv_Area=1000 -> PyTorch: $187,534, ONNX: $187,534, diff=0.00000000
# Gr_Liv_Area=1100 -> PyTorch: $192,223, ONNX: $192,223, diff=0.00000000
# Gr_Liv_Area=1200 -> PyTorch: $197,028, ONNX: $197,028, diff=0.00000000

# =====================================================
# LOAD SCALER PARAMETERS
# =====================================================
mean = np.load("mean.npy")
std = np.load("std.npy")

class DummyScaler:
    """Wrapper to emulate StandardScaler interface for make_input"""
    def __init__(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

scaler = DummyScaler(mean, std)

# =====================================================
# LOAD TRAINED PYTORCH MODEL
# =====================================================
model = HouseModel(len(FEATURES))
model.load_state_dict(torch.load("model.pth"))
model.eval()  # important!

# =====================================================
# LOAD ONNX MODEL
# =====================================================
ort_session = ort.InferenceSession("model.onnx")

# =====================================================
# TEST CASE 1: MANUAL PREDICTIONS (PyTorch + ONNX)
# =====================================================
print("\n🔎 Manual predictions (PyTorch vs ONNX):")

base = {
    "Gr_Liv_Area": 900,
    "Overall_Qual": 7,
    "Year_Built": 2005,
    "Garage_Cars": 2,
    "Full_Bath": 2,
    "Bedroom_AbvGr": 3,
    "Lot_Area": 8000
}

test_values = [900, 1000, 1100, 1200]

for val in test_values:
    row = base.copy()
    row["Gr_Liv_Area"] = val
    row = build_features(row)
    x_scaled = make_input(row, scaler)
    
    # PyTorch prediction
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        torch_pred = model(x_tensor).item()
        price_torch = np.expm1(torch_pred)
    
    # ONNX prediction
    ort_out = ort_session.run(None, {"input": x_scaled.reshape(1, -1)})
    onnx_pred = np.expm1(ort_out[0][0][0])
    
    # Print results
    diff = abs(price_torch - onnx_pred)
    print(f"Gr_Liv_Area={val} -> PyTorch: ${price_torch:,.0f}, ONNX: ${onnx_pred:,.0f}, diff={diff:.8f}")