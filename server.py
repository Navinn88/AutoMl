from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
from data_utils import prepare_data, DataCleaner
from model import LinearRegression
from logistic_model import LogisticRegression
from decision_tree_model import DecisionTreeClassifier
from llm_trainer import feed_llm_and_train
from typing import Optional
from evaluate import calculate_metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from io import StringIO

app = FastAPI()

# Allow CORS (optional, but useful for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
X_test = None
y_test = None
X_train = None
y_train = None
target_column = None
GEMINI_API_KEY = "AIzaSyAqh7NHtd6z7ZkYxM6R6qOScVuSExUTLU0"

def initialize_model(model_type: str):
    """Initialize model based on LLM selection."""
    global model
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "logistic":
        model = LogisticRegression()
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

@app.post("/train")
async def train(request: Request, target: str, context: Optional[str] = None):
    """Train model using LLM selection and return training results."""
    global model, X_test, y_test, X_train, y_train, target_column
    try:
        csv_text = await request.body()
        csv_text = csv_text.decode()
        print("Raw CSV (first 100 chars):", csv_text[:100])
        df = pd.read_csv(StringIO(csv_text))
        print("CSV columns (from pandas):", df.columns.tolist())
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in CSV. Available columns: {df.columns.tolist()}.")
        if is_numeric_dtype(df[target]):
            target_type = "numeric"
        elif is_categorical_dtype(df[target]) or df[target].dtype == object:
            target_type = "categorical"
        else:
            target_type = str(df[target].dtype)
        type_context = f"Target column '{target}' is {target_type}. "
        if context:
            context = type_context + context
        else:
            context = type_context
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_text)
            temp_csv = f.name
        model_info = feed_llm_and_train(temp_csv, target, GEMINI_API_KEY, context)
        os.remove(temp_csv)
        if target_type == "numeric" and model_info["model_type"] in ["logistic", "decision_tree"]:
            model_type = "linear"
        elif target_type == "categorical" and model_info["model_type"] == "linear":
            model_type = "logistic"
        else:
            model_type = model_info["model_type"]
        initialize_model(model_type)
        X, y = prepare_data(csv_text, target_column=target, target_type=target_type)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Encode string labels for logistic regression
        label_mapping = None
        if model_type == "logistic":
            unique_labels = np.unique(y_train)
            if y_train.dtype.kind not in {'i', 'u', 'f'}:
                label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                y_train = np.array([label_mapping[val] for val in y_train])
                y_test = np.array([label_mapping[val] for val in y_test])
        model.fit(X_train, y_train)
        params = model.get_params()
        target_column = target
        return JSONResponse({
            'status': 'success',
            'message': 'Model was trained successfully',
            'model_type': model_type,
            'intercept': float(params.get('intercept', 0.0)),
            'coefficients': [float(x) for x in params.get('weights', [])],
            'feature_count': len(params.get('weights', [])),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'label_mapping': label_mapping if label_mapping else None
        })
    except Exception as e:
        return JSONResponse({'status': 'error', 'error': f'Training error: {str(e)}'}, status_code=400)

@app.post("/eval")
async def eval(request: Request):
    """Evaluate trained model on train/test splits and return comprehensive metrics."""
    global model, X_test, y_test, X_train, y_train
    try:
        # Check if model is trained (different models have different attributes)
        is_trained = False
        if model is not None:
            if hasattr(model, 'weights') and model.weights is not None:
                is_trained = True
            elif hasattr(model, 'is_fitted') and getattr(model, 'is_fitted', False):
                is_trained = True
            elif hasattr(model, 'root') and model.root is not None:  # DecisionTreeClassifier
                is_trained = True
            elif hasattr(model, 'coefficients') and model.coefficients is not None:  # LogisticRegression
                is_trained = True
        
        if not is_trained:
            return JSONResponse({'error': "Model not trained yet"}, status_code=400)
        
        # Determine if this is a classification task
        is_classification = False
        if hasattr(model, '__class__'):
            model_class_name = model.__class__.__name__
            if model_class_name in ['LogisticRegression', 'DecisionTreeClassifier']:
                is_classification = True
            elif model_class_name == 'LinearRegression':
                is_classification = False
            else:
                # Fallback: check target values
                if y_train is not None:
                    is_classification = len(np.unique(y_train)) <= 10
        
        metrics = {}
        # For logistic regression, do both unoptimized (0.5) and optimized threshold
        if model_class_name == 'LogisticRegression' and hasattr(model, 'predict_proba'):
            # Unoptimized (default 0.5)
            if X_train is not None and y_train is not None:
                y_pred_train = (model.predict_proba(X_train) >= 0.5).astype(int)
                y_prob_train = model.predict_proba(X_train)
                train_metrics_unopt = calculate_metrics(y_train, y_pred_train, is_classification=True, y_prob=None)
                train_metrics_unopt['train_size'] = len(X_train)
                metrics['train_unoptimized'] = train_metrics_unopt
            if X_test is not None and y_test is not None:
                y_pred_test = (model.predict_proba(X_test) >= 0.5).astype(int)
                y_prob_test = model.predict_proba(X_test)
                test_metrics_unopt = calculate_metrics(y_test, y_pred_test, is_classification=True, y_prob=None)
                test_metrics_unopt['test_size'] = len(X_test)
                metrics['test_unoptimized'] = test_metrics_unopt
            # Optimized
            if X_train is not None and y_train is not None:
                y_prob_train = model.predict_proba(X_train)
                from evaluate import find_optimal_threshold
                optimal_threshold, _ = find_optimal_threshold(y_train, y_prob_train)
                y_pred_train_opt = (y_prob_train >= optimal_threshold).astype(int)
                train_metrics_opt = calculate_metrics(y_train, y_pred_train_opt, is_classification=True, y_prob=y_prob_train)
                train_metrics_opt['train_size'] = len(X_train)
                train_metrics_opt['optimal_threshold'] = float(optimal_threshold)
                metrics['train_optimized'] = train_metrics_opt
            if X_test is not None and y_test is not None:
                y_prob_test = model.predict_proba(X_test)
                from evaluate import find_optimal_threshold
                optimal_threshold, _ = find_optimal_threshold(y_test, y_prob_test)
                y_pred_test_opt = (y_prob_test >= optimal_threshold).astype(int)
                test_metrics_opt = calculate_metrics(y_test, y_pred_test_opt, is_classification=True, y_prob=y_prob_test)
                test_metrics_opt['test_size'] = len(X_test)
                test_metrics_opt['optimal_threshold'] = float(optimal_threshold)
                metrics['test_optimized'] = test_metrics_opt
        else:
            if X_train is not None and y_train is not None:
                y_pred_train = model.predict(X_train)
                train_metrics = calculate_metrics(y_train, y_pred_train, is_classification=is_classification)
                train_metrics['train_size'] = len(X_train)
                metrics['train'] = train_metrics
            if X_test is not None and y_test is not None:
                y_pred_test = model.predict(X_test)
                test_metrics = calculate_metrics(y_test, y_pred_test, is_classification=is_classification)
                test_metrics['test_size'] = len(X_test)
                metrics['test'] = test_metrics
        if not metrics:
            return JSONResponse({'error': "No train or test data available for evaluation."}, status_code=400)
        return JSONResponse(metrics)
    except Exception as e:
        return JSONResponse({'error': f'Evaluation error: {str(e)}'}, status_code=400)

@app.post("/predict")
async def predict(request: Request):
    """Make predictions on CSV data and return CSV with predictions column."""
    global model, target_column
    try:
        if target_column is None:
            return JSONResponse({'error': "Model not trained yet. Please train a model first."}, status_code=400)
            
        csv_text = await request.body()
        csv_text = csv_text.decode()
        # Detect target type for prediction
        df = pd.read_csv(StringIO(csv_text))
        if target_column not in df.columns:
            return JSONResponse({'error': f"Target column '{target_column}' not found in CSV. Available columns: {df.columns.tolist()}."}, status_code=400)
            
        if is_numeric_dtype(df[target_column]):
            target_type = "numeric"
        else:
            target_type = "categorical"
        X, y = prepare_data(csv_text, target_column=target_column, target_type=target_type)
        
        is_trained = False
        expected_features = 0
        if model is not None:
            if hasattr(model, 'weights') and model.weights is not None:
                is_trained = True
                expected_features = len(model.weights)
            elif hasattr(model, 'is_fitted') and getattr(model, 'is_fitted', False):
                is_trained = True
                expected_features = X.shape[1]  # Use actual feature count
            elif hasattr(model, 'root') and model.root is not None:  # DecisionTreeClassifier
                is_trained = True
                expected_features = X.shape[1]  # Use actual feature count
            elif hasattr(model, 'coefficients') and model.coefficients is not None:  # LogisticRegression
                is_trained = True
                expected_features = len(model.coefficients)
        
        if not is_trained:
            return JSONResponse({'error': "Model not trained yet"}, status_code=400)
        
        if X.shape[1] != expected_features:
            return JSONResponse({'error': f"Expected {expected_features} features, got {X.shape[1]}"}, status_code=400)
        
        # Use optimal threshold for logistic regression if available
        predictions = None
        if hasattr(model, 'predict_proba') and model.__class__.__name__ == 'LogisticRegression':
            # Find optimal threshold using all data
            y_prob = model.predict_proba(X)
            from evaluate import find_optimal_threshold
            # For prediction, assume equal class distribution, use 0.5 or allow user to specify
            optimal_threshold, _ = find_optimal_threshold(np.zeros_like(y_prob), y_prob)  # Dummy y_true, just to get threshold range
            predictions = (y_prob >= optimal_threshold).astype(int)
        else:
            predictions = model.predict(X)
        
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df[target_column] = y
        df["predicted"] = predictions
        csv_string = df.to_csv(index=False)
        return Response(content=csv_string, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predicted.csv"})
    except Exception as e:
        return JSONResponse({'error': f'Prediction error: {str(e)}'}, status_code=400) 