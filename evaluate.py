import numpy as np
from data_utils import load_and_clean_data
from model import LinearRegression
import json
import requests
import sys
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate R², MSE, and RMSE metrics."""
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return { 'r2_score': float(r2), 'mse': float(mse), 'rmse': float(rmse) }

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and testing sets (80% train, 20% test)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_and_evaluate(data_file: str, server_url: str = "http://localhost:8000") -> Dict[str, float]:
    """Train model on 80% of data and evaluate on the remaining 20%."""
    try:
        print("Loading data...")
        X, y = load_and_clean_data(data_file)
        print("Splitting data into train (80%) and test (20%) sets...")
        X_train, X_test, y_train, y_test = split_data(X, y)
        train_data = np.column_stack((X_train, y_train))
        np.savetxt('temp_train.csv', train_data, delimiter=',', header='feature1,feature2,target', comments='')
        print("Training model...")
        with open('temp_train.csv', 'r') as f:
            train_csv = f.read()
        train_response = requests.post(f"{server_url}/train", data=train_csv.encode(), headers={'Content-Type': 'text/csv'})
        train_response.raise_for_status()
        print("Making predictions...")
        predict_response = requests.post(f"{server_url}/predict", json={'features': X_test.tolist()}, headers={'Content-Type': 'application/json'})
        predict_response.raise_for_status()
        y_pred = np.array(predict_response.json()['predictions'])
        metrics = calculate_metrics(y_test, y_pred)
        print("\nModel Performance Metrics:")
        print(f"R² Score: {metrics['r2_score']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        import os
        if os.path.exists('temp_train.csv'):
            os.remove('temp_train.csv')
        return metrics
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with API server: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train and evaluate linear regression model')
    parser.add_argument('--data', required=True, help='Path to input CSV file')
    parser.add_argument('--server', default='http://localhost:8000', help='API server URL')
    args = parser.parse_args()
    train_and_evaluate(args.data, args.server) 