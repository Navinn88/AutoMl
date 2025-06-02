import numpy as np
from data_utils import load_and_clean_data
from model import LinearRegression
import json
import requests
import sys
from typing import Tuple, Dict

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate R², MSE, and RMSE metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing R², MSE, and RMSE
    """
    # Calculate R²
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    # Calculate MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Calculate RMSE
    rmse = np.sqrt(mse)
    
    return {
        'r2_score': float(r2),
        'mse': float(mse),
        'rmse': float(rmse)
    }

def train_and_evaluate(train_file: str, test_file: str, server_url: str = "http://localhost:8000") -> Dict[str, float]:
    """
    Train model on train.csv and evaluate on test.csv.
    
    Args:
        train_file: Path to training CSV file
        test_file: Path to test CSV file
        server_url: URL of the API server
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        # Load and prepare training data
        print("Loading training data...")
        X_train, y_train = load_and_clean_data(train_file)
        
        # Load and prepare test data
        print("Loading test data...")
        X_test, y_test = load_and_clean_data(test_file)
        
        # Train the model via API
        print("Training model...")
        with open(train_file, 'r') as f:
            train_csv = f.read()
        
        train_response = requests.post(
            f"{server_url}/train",
            data=train_csv.encode(),
            headers={'Content-Type': 'text/csv'}
        )
        train_response.raise_for_status()
        
        # Make predictions on test data
        print("Making predictions...")
        predict_response = requests.post(
            f"{server_url}/predict",
            json={'features': X_test.tolist()},
            headers={'Content-Type': 'application/json'}
        )
        predict_response.raise_for_status()
        
        # Get predictions
        y_pred = np.array(predict_response.json()['predictions'])
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        
        # Print results
        print("\nModel Performance Metrics:")
        print(f"R² Score: {metrics['r2_score']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        
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
    parser.add_argument('--train', required=True, help='Path to training CSV file')
    parser.add_argument('--test', required=True, help='Path to test CSV file')
    parser.add_argument('--server', default='http://localhost:8000', help='API server URL')
    
    args = parser.parse_args()
    
    # Start evaluation
    train_and_evaluate(args.train, args.test, args.server) 