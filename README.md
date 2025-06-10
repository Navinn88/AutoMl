# Linear Regression API with LLM Model Selection

A FastAPI-based machine learning service that uses Google's Gemini to intelligently select and train models for regression/classification tasks. The API supports linear regression, logistic regression, and decision tree models with advanced features like threshold optimization for classification models.

## Project Structure

- `server.py`: FastAPI server with `/train`, `/eval`, and `/predict` endpoints. Handles model training, evaluation, and predictions.
- `model.py`: Implements linear regression using normal equations. Includes intercept fitting and parameter management.
- `logistic_model.py`: Binary classification using logistic regression with gradient descent, sigmoid activation, and probability predictions.
- `decision_tree_model.py`: Decision tree classifier using Gini impurity for splits and recursive tree building.
- `llm_trainer.py`: Uses Google's Gemini to analyze data and select the most appropriate model type.
- `evaluate.py`: Computes model performance metrics and includes threshold optimization for logistic regression.
- `data_utils.py`: Data preprocessing including cleaning, normalization, and feature engineering.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

## API Endpoints

### 1. Train Model (`/train`)

Trains a model using LLM-based selection. The system analyzes your data and context to choose the best model type.

**Method:** `POST`

**Parameters:**
- `target` (query): Name of the target column
- `context` (query, optional): Additional context to help LLM select the best model

**Request Body:** CSV data

**Example:**
```bash
curl -X POST "http://127.0.0.1:8000/train?target=Personality&context=This is a binary classification task with categorical target. Logistic regression is preferred for interpretability." \
  -H "Content-Type: application/json" \
  --data-binary "@personality_dataset.csv"
```

**Response:**
```json
{
  "status": "success",
  "message": "Model was trained successfully",
  "model_type": "logistic",
  "intercept": 0.0,
  "coefficients": [...],
  "feature_count": 5,
  "train_size": 2320,
  "test_size": 580,
  "label_mapping": {...}
}
```

### 2. Evaluate Model (`/eval`)

Evaluates the trained model on train and test splits, returning comprehensive metrics.

**Method:** `POST`

**Request Body:** Empty (uses stored train/test data)

**Example:**
```bash
curl -X POST "http://127.0.0.1:8000/eval" \
  -H "Content-Type: application/json"
```

**Response for Classification Models (Logistic Regression):**
```json
{
  "train_unoptimized": {
    "accuracy": 0.9358,
    "precision": 0.9467,
    "recall": 0.9268,
    "f1_score": 0.9367,
    "train_size": 2320
  },
  "test_unoptimized": {
    "accuracy": 0.9293,
    "precision": 0.9158,
    "recall": 0.9388,
    "f1_score": 0.9272,
    "test_size": 580
  },
  "train_optimized": {
    "accuracy": 0.9362,
    "precision": 0.9475,
    "recall": 0.9268,
    "f1_score": 0.9371,
    "optimal_threshold": 0.39,
    "train_size": 2320
  },
  "test_optimized": {
    "accuracy": 0.9293,
    "precision": 0.9158,
    "recall": 0.9388,
    "f1_score": 0.9272,
    "optimal_threshold": 0.42,
    "test_size": 580
  }
}
```

**Response for Regression Models (Linear Regression):**
```json
{
  "train": {
    "mse": 0.1234,
    "rmse": 0.3512,
    "mae": 0.2987,
    "r2": 0.8765,
    "train_size": 1000
  },
  "test": {
    "mse": 0.1345,
    "rmse": 0.3667,
    "mae": 0.3123,
    "r2": 0.8543,
    "test_size": 250
  }
}
```

### 3. Make Predictions (`/predict`)

Makes predictions on new data and returns a CSV with predictions.

**Method:** `POST`

**Request Body:** CSV data (must include the target column used during training)

**Example:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  --data-binary "@test_data.csv"
```

**Response:** CSV file with original data plus a "predicted" column

## Model Selection

The system uses Google's Gemini to analyze your data and context, then selects the most appropriate model:

- **Linear Regression**: For continuous target variables
- **Logistic Regression**: For binary classification (with threshold optimization)
- **Decision Tree**: For complex classification tasks

## Advanced Features

### Threshold Optimization for Logistic Regression

For logistic regression models, the system automatically:
1. Evaluates performance with the default threshold (0.5)
2. Finds the optimal probability threshold that maximizes accuracy
3. Reports metrics for both thresholds
4. Uses the optimal threshold for predictions

### Data Handling

- **Missing Values**: Automatically filled with appropriate strategies
- **Complex Data Types**: Dropped to ensure compatibility
- **Categorical Variables**: Encoded for model compatibility
- **Feature Engineering**: Automatic preprocessing and normalization

### Model Evaluation Metrics

**For Regression Models:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

**For Classification Models:**
- Accuracy
- Precision
- Recall
- F1-score
- Optimal threshold (for logistic regression)

## Testing

Run the test script to verify all features:

```bash
python test_threshold_optimization.py
```

This script tests:
- Model training with LLM selection
- Evaluation with threshold optimization
- Prediction functionality
- CSV input/output handling

## LLM Trainer (Experimental)

The `llm_trainer.py` script demonstrates how to feed cleaned data, the target variable name, and the number of features to an LLM API. The LLM will decide which model to use (Linear Regression, Logistic Regression, or Decision Tree) based on the data, target, and any optional user-provided context.

### Usage
```bash
python llm_trainer.py --data train.csv --target target_column --api_key YOUR_API_KEY [--context "Optional context for the LLM"]
```

**Parameters:**
- `--data`: Path to your CSV file
- `--target`: Name of the target column
- `--api_key`: API key for the Gemini API
- `--context`: (Optional) Context string to help the LLM decide which model to use

**Examples:**

With context:
```bash
python llm_trainer.py --data train.csv --target target_column --api_key YOUR_API_KEY --context "This is a classification problem."
```

Without context:
```bash
python llm_trainer.py --data train.csv --target target_column --api_key YOUR_API_KEY
```

## Features Summary

- ✅ Automatic model selection using LLM
- ✅ Data preprocessing and normalization
- ✅ Comprehensive model evaluation metrics
- ✅ Threshold optimization for logistic regression
- ✅ Support for both regression and classification tasks
- ✅ RESTful API with FastAPI
- ✅ CORS support for web applications
- ✅ CSV input/output for predictions
- ✅ Train/test split evaluation
- ✅ Label encoding for categorical targets

model = genai.GenerativeModel("models/gemini-1.5-pro-latest")