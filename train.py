import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from features import FEATURES
import random

#======================================================
# Results of train.py and testfour with the parametres: 
#======================================================

# 10-05-2026
# 17 % error average in testfour by 1 hidden layer, 4 neurons lr=0.006, weight_decay=1e-4, epochs=1000
# 14 % error average in testfour by 1 hidden layer, 4 neurons lr=0.01, weight_decay=1e-4, epochs=800 
# 13 % error average in testfour by 1 hidden layer, 4 neurons lr=0.03, weight_decay=1e-4, epochs=400
# =====================================================
# SET RANDOM SEED
# =====================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()

# =====================================================
# FEATURE ENGINEERING
# =====================================================
def build_features(row):
    row = row.copy()
    row["HouseAge"] = 2026 - row["Year_Built"]
    row["HasGarage"] = 1 if row["Garage_Cars"] > 0 else 0
    return row

def make_input(row, scaler):
    x = np.array([row[f] for f in FEATURES], dtype=np.float32)
    x_scaled = (x - scaler.mean_) / scaler.scale_
    return x_scaled.astype(np.float32)

# =====================================================
# DATA HANDLER CLASS
# =====================================================
class AmesDataset:
    def __init__(self, csv_path="AmesHousing.csv"):
        self.df = pd.read_csv(csv_path)
        self._rename_columns()
        self.df = self.df.apply(build_features, axis=1)
        self.df = self.df[FEATURES + ["SalePrice"]].dropna()
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.y = None
        self._prepare_features()

    def _rename_columns(self):
        self.df = self.df.rename(columns={
            "Gr Liv Area": "Gr_Liv_Area",
            "Overall Qual": "Overall_Qual",
            "Year Built": "Year_Built",
            "Garage Cars": "Garage_Cars",
            "Full Bath": "Full_Bath",
            "Bedroom AbvGr": "Bedroom_AbvGr",
            "Lot Area": "Lot_Area"
        })

    def _prepare_features(self):
        X = self.df[FEATURES].values.astype(np.float32)
        self.X_scaled = self.scaler.fit_transform(X)
        self.y = np.log1p(self.df["SalePrice"].values.astype(np.float32))
        # Save scaler params
        np.save("mean.npy", self.scaler.mean_)
        np.save("std.npy", self.scaler.scale_)

    def get_torch_tensors(self, train_ratio=0.8):
        split = int(train_ratio * len(self.X_scaled))
        X_train = torch.tensor(self.X_scaled[:split], dtype=torch.float32)
        y_train = torch.tensor(self.y[:split], dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(self.X_scaled[split:], dtype=torch.float32)
        y_test = torch.tensor(self.y[split:], dtype=torch.float32).view(-1, 1)
        return X_train, y_train, X_test, y_test

# =====================================================
# MODEL CLASS
# =====================================================
class HouseModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 4),  # 4 neurons for stability
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.net(x)

# =====================================================
# TRAINER CLASS
# =====================================================
# 05-05-2026: Using lr=0.006 instead of lr=0.005 with epochs=1000 did the following:
# testfour.py: Lowered the "Average percentage error" from: 24 % to 17 %
# testone.py: The Manual predictions (PyTorch vs ONNX) are still as expected: 
# When Gr_Liv_Area increase the Sales Price increase too !
class Trainer:
    def __init__(self, model, X_train, y_train, X_test, y_test, lr=0.006, weight_decay=1e-4, epochs=1000):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs

        # 🔹 Standard MSE Loss
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            pred = self.model(self.X_train)
            loss = self.loss_fn(pred, self.y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: loss={loss.item():.4f}")


    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            test_loss = self.loss_fn(self.model(self.X_test), self.y_test)
        print("\n📊 Final Test Loss:", test_loss.item())
        return test_loss.item()

    def export_onnx(self, filename="model.onnx"):
        dummy = torch.randn(1, self.X_train.shape[1], dtype=torch.float32)
        torch.onnx.export(
            self.model,
            dummy,
            filename,
            input_names=["input"],
            output_names=["output"],
            opset_version=17
        )
        print("\n✅ model.onnx exported")

# =====================================================
# OPTIONAL: Linear Regression Baseline
# =====================================================
def linear_baseline(df):
    print("\n📏 Linear Regression baseline:")
    linreg = LinearRegression()
    linreg.fit(df[FEATURES], df["SalePrice"])
    coef = dict(zip(FEATURES, linreg.coef_))
    print("Gr_Liv_Area coefficient:", coef["Gr_Liv_Area"])
    return linreg

# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    print("\n🚀 Loading dataset...")
    dataset = AmesDataset()
    X_train, y_train, X_test, y_test = dataset.get_torch_tensors()

    print("🚀 Initializing model and trainer...")
    model = HouseModel(len(FEATURES))
    trainer = Trainer(model, X_train, y_train, X_test, y_test)

    print("🚀 Training model...")
    trainer.train()

    print("🚀 Evaluating model...")
    trainer.evaluate()

    print("🚀 Exporting ONNX model...")
    trainer.export_onnx()
    print("✅ Training complete. mean.npy + std.npy saved.\n")

    # Save trained PyTorch model
    torch.save(model.state_dict(), "model.pth")
    print("✅ PyTorch model saved as model.pth")