from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
from data_utils import prepare_data
from model import LinearRegression

app = FastAPI()

# Allow CORS (optional, but useful for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = LinearRegression()

@app.post("/train")
async def train(request: Request):
    """Train the model using CSV data sent in the request body."""
    try:
        csv_text = await request.body()
        csv_text = csv_text.decode()
        X, y = prepare_data(csv_text)
        global model
        model = LinearRegression()  # Reset model
        model.fit(X, y)
        params = model.get_params()
        return JSONResponse({
            'intercept': float(params['intercept']),
            'coefficients': params['weights'].tolist(),
            'feature_count': len(params['weights'])
        })
    except Exception as e:
        return JSONResponse({'error': f'Training error: {str(e)}'}, status_code=400)

@app.post("/predict")
async def predict(request: Request):
    """Predict using JSON features sent in the request body."""
    try:
        data = await request.json()
        if 'features' not in data:
            return JSONResponse({'error': "Missing 'features' in request body"}, status_code=400)
        features = data['features']
        if not isinstance(features, list):
            return JSONResponse({'error': "'features' must be a list"}, status_code=400)
        X = np.array(features)
        if model.weights is None:
            return JSONResponse({'error': "Model not trained yet"}, status_code=400)
        expected_features = len(model.weights)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != expected_features:
            return JSONResponse({'error': f"Expected {expected_features} features, got {X.shape[1]}"}, status_code=400)
        predictions = model.predict(X)
        return JSONResponse({'predictions': predictions.tolist()})
    except Exception as e:
        return JSONResponse({'error': f'Prediction error: {str(e)}'}, status_code=400) 