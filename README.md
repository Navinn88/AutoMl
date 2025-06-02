# Linear Regression API

A simple API for linear regression with automatic data cleaning.

## Project Structure

- `data_utils.py`: CSV parsing and cleaning routines
- `model.py`: Linear regression implementation using NumPy
- `server.py`: HTTP server with endpoints for training and prediction
- `requirements.txt`: Project dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python server.py
```

## API Endpoints

- `POST /train`: Train the model with provided CSV data
  - Input: CSV file with features and target variable
  - Returns: Model parameters and training metrics

- `POST /predict`: Make predictions using the trained model
  - Input: JSON array of feature values
  - Returns: Predicted values

## Example Usage

1. Train the model:
```bash
curl -X POST -F "file=@data.csv" http://localhost:8000/train
```

2. Make predictions:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"features": [[1.0, 2.0], [3.0, 4.0]]}' \
     http://localhost:8000/predict
```