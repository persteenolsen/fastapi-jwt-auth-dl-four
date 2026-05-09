# 🏠 v8 - House Price Prediction API (FastAPI + PyTorch + JWT + Ames Dataset + Tests + Vue 3 SPA)

Last updated 

- 09-05-2026

A production-style machine learning backend system that predicts house prices using a PyTorch neural network trained on the Ames Housing dataset and served through a secure FastAPI API with JWT authentication

This project demonstrates a full ML engineering pipeline: data preprocessing → feature engineering → model training → Tests → ONNX export → secure API inference

# Vue 3 frontend for the Web API

- [`The Vue 3 SPA at GitHub`](https://github.com/persteenolsen/vue-fastapi-jwt-auth-dl-four) - The Vue 3 SPA using JWT Authentication

# 🧪 Tests

This project includes four test scripts to validate model predictions and overall quality

Note: For running a test inside the folder "tests" use the command:  

    python -m tests.testone

### testone.py – Manual PyTorch vs ONNX Comparison
Compares the PyTorch model predictions with the ONNX exported model on a set of hand-picked inputs.

Gr_Liv_Area=900  -> PyTorch: $180,157, ONNX: $180,157, diff=0.167  
Gr_Liv_Area=1000 -> PyTorch: $184,697, ONNX: $184,697, diff=0.178  
Gr_Liv_Area=1100 -> PyTorch: $189,351, ONNX: $189,351, diff=0.181  
Gr_Liv_Area=1200 -> PyTorch: $194,122, ONNX: $194,122, diff=0.186  

### testtwo.py – Predictions for Custom Inputs
Tests specific example inputs to verify ONNX model alignment with PyTorch outputs.

Example 1: PyTorch=$184,697, ONNX=$184,697, diff=0.178  
Example 2: PyTorch=$257,907, ONNX=$257,907, diff=0.004  

### testthree.py – Train/Test Loss Check
Evaluates the model’s loss on train and test sets to confirm proper fitting.

Train Loss: 0.0471  
Test Loss: 0.0531  
✅ Model fit looks good  

### testfour.py – Top Prediction Errors Analysis
Identifies the largest errors in the test set and calculates overall prediction accuracy.

Top 5 largest prediction errors:  
Predicted=$405,438, Actual=$118,500, Error=$286,938  
Predicted=$342,997, Actual=$124,000, Error=$218,997  
Predicted=$231,987, Actual=$35,311, Error=$196,676  
Predicted=$248,036, Actual=$415,000, Error=$166,964  
Predicted=$281,822, Actual=$138,887, Error=$142,935  

Average percentage error: 17.71%  
⚠️ 66 houses have >30% prediction error  

These tests ensure model consistency, alignment between PyTorch and ONNX predictions, training quality, and edge-case performance.

# 📁 Refactoring of train.py

- train.py was refactored into classes and modules using the new files for tests 

# 🚀 Features

- End-to-end ML pipeline using real-world Ames Housing dataset  
- Feature engineering (HouseAge, HasGarage derived features)  
- Robust feature alignment between training and inference  
- Input normalization for stable neural network training  
- PyTorch neural network regression model  
- Log-transformed target (log1p) for stable regression learning  
- ONNX export for fast production inference  
- Secure REST API using FastAPI  
- JWT authentication (OAuth2 password flow)  
- Named-feature JSON input (no manual ordering required)  
- Consistent preprocessing across training and inference  
- Serverless-ready deployment design (Vercel compatible)

# 🧱 Tech Stack

- Python 3.12  
- FastAPI  
- PyTorch  
- ONNX Runtime  
- NumPy  
- Pandas  
- scikit-learn  
- python-jose (JWT auth)  
- Uvicorn  
- joblib  

# 📁 Project Structure

```
    .
    ├── main.py                # FastAPI inference API (JWT + ONNX)
    ├── train.py               # Model training + ONNX export
    ├── features.py            # Shared feature definitions + transforms
    ├── AmesHousing.csv        # Dataset
    ├── model.onnx             # Exported inference model
    ├── mean.npy               # Feature normalization mean
    ├── std.npy                # Feature normalization std
    ├── requirements.txt
    └── .env                   # Environment variables (not committed)
    └── tests                  # Tests
```

# ⚙️ Installation

    git clone https://github.com/your-repo/house-price-api.git
    cd house-price-api
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install -r requirements/dev.txt

Verify setup:

    python -c "import fastapi, numpy, onnxruntime; print('VENV OK')"

Expected output:

    VENV OK

# 🔐 Environment Variables (.env)

    SECRET_KEY=your_secret_key
    ALGORITHM=HS256
    ACCESS_TOKEN_EXPIRE_MINUTES=60
    ADMIN_USERNAME=admin
    ADMIN_PASSWORD=password

# 🏋️ Training the Model

    python train.py

Outputs:

- model.onnx  
- mean.npy  
- std.npy  

# 🔧 Model Tuning

- Removed a hidden layer  
- Reduced the hidden layer size (16 → 4 neurons)  
- Lowered learning rate (0.01 → 0.006)  
- Increased training epochs (500 → 1000)  
- Added weight decay (L2 regularization)  
- Introduced early stopping for training stability  
- Improved numerical stability in normalization  

Result:

- Smooth, monotonic price curves  
- Stable age depreciation behavior  
- More consistent size scaling  
- Reduced high-range prediction jumps  
- Better generalization without overfitting

# 🔬 Training Pipeline

- Load Ames Housing dataset  
- Feature engineering: HouseAge (derived), HasGarage (binary feature)  
- Select final feature set  
- Normalize features (mean/std scaling)  
- Train PyTorch neural network  
- Optimize regression using log1p target  
- Export model to ONNX format  
- Save normalization parameters for inference

# 🚀 Run API Locally

    uvicorn main:app --reload

Swagger UI:

    http://127.0.0.1:8000/docs

# 🔐 Authentication

POST `/login`

    username=admin
    password=password

Response:

    {
      "access_token": "jwt_token_here",
      "token_type": "bearer"
    }

Use Token:

    Authorization: Bearer <token>

# 📡 Prediction Endpoint

POST `/predict`

Example:

    {
      "Gr_Liv_Area": 1500,
      "Overall_Qual": 7,
      "Year_Built": 2005,
      "Garage_Cars": 2,
      "Full_Bath": 2,
      "Bedroom_AbvGr": 3,
      "Lot_Area": 8000
    }

Response:

   {
     "predicted_price": 209169.703125
  }

Another example:

    {
      "Gr_Liv_Area": 1200,
      "Overall_Qual": 7,
      "Year_Built": 2005,
      "Garage_Cars": 2,
      "Full_Bath": 2,
      "Bedroom_AbvGr": 3,
      "Lot_Area": 8000
    }

Response:

   {
     "predicted_price": 194122.25
   }

# Clamping Predicted Price

- Maximum value: $755,000  
- Reverse log-transform, clamp values above cap  

# 🧠 Key Design Feature

- Named-feature system (safe ML design)  
- No manual feature ordering in API  
- Centralized feature definition (`features.py`)  
- Automatic transformation + alignment  
- Prevents silent prediction errors  

# 🔁 Training vs Inference Consistency

- Training: defines feature space, applies normalization, trains PyTorch, exports ONNX + normalization params  
- Inference: receives JSON, applies same transforms, normalization, runs ONNX, returns prediction  

# 🧪 System Flow

1. User logs in → receives JWT  
2. Sends house feature JSON  
3. API verifies JWT, builds feature vector, applies normalization, runs ONNX inference, returns predicted price  

# 🧠 What this project demonstrates

- Production-style ML system architecture  
- Feature engineering for tabular regression  
- Train/inference consistency design  
- Secure API authentication (JWT)  
- ONNX-based model deployment  
- Serverless-compatible ML inference design 

# 👨‍💻 Author

Built as a machine learning engineering project demonstrating production-style ML system design using FastAPI, PyTorch, and ONNX for real-world deployment scenarios.