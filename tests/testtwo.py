import numpy as np
import torch
import onnxruntime as ort
from train import AmesDataset, HouseModel, build_features, make_input, FEATURES

# Note: Run the test by the command:
# python -m tests.testtwo
# Expected output
#🔎 Predictions for custom inputs:
#Example 1: PyTorch=$187,534, ONNX=$187,534, diff=0.00731910
#Example 2: PyTorch=$268,714, ONNX=$268,714, diff=0.00776899

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
model.eval()

# =====================================================
# LOAD ONNX MODEL
# =====================================================
ort_session = ort.InferenceSession("model.onnx")

# =====================================================
# HELPER FUNCTION: predict house price
# =====================================================
def predict_price(input_features: dict):
    """Return predicted price for given feature dictionary using PyTorch and ONNX."""
    row = build_features(input_features.copy())
    x_scaled = make_input(row, scaler)
    
    # PyTorch
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        torch_pred = model(x_tensor).item()
        price_torch = np.expm1(torch_pred)
    
    # ONNX
    ort_out = ort_session.run(None, {"input": x_scaled.reshape(1, -1)})
    onnx_pred = np.expm1(ort_out[0][0][0])
    
    diff = abs(price_torch - onnx_pred)
    
    return price_torch, onnx_pred, diff

# =====================================================
# EXAMPLE USAGE / TEST CASES
# =====================================================
if __name__ == "__main__":
    examples = [
        {"Gr_Liv_Area": 1000, "Overall_Qual": 7, "Year_Built": 2005,
         "Garage_Cars": 2, "Full_Bath": 2, "Bedroom_AbvGr": 3, "Lot_Area": 8000},
        {"Gr_Liv_Area": 1500, "Overall_Qual": 8, "Year_Built": 2010,
         "Garage_Cars": 3, "Full_Bath": 3, "Bedroom_AbvGr": 4, "Lot_Area": 12000}
    ]
    
    print("\n🔎 Predictions for custom inputs:")
    for i, feat in enumerate(examples, 1):
        pt, onnx, diff = predict_price(feat)
        print(f"Example {i}: PyTorch=${pt:,.0f}, ONNX=${onnx:,.0f}, diff={diff:.8f}")