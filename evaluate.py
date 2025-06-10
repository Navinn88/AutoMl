import numpy as np
from data_utils import load_and_clean_data
from model import LinearRegression
import json
import requests
import sys
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Find optimal probability threshold that maximizes accuracy for binary classification."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_accuracy = 0
    optimal_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        accuracy = np.mean(y_true == y_pred)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            optimal_threshold = threshold
    
    return optimal_threshold, best_accuracy

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_classification: bool = False, y_prob: np.ndarray = None) -> Dict[str, float]:
    """Calculate metrics for model evaluation (classification or regression)."""
    if is_classification:
        return calculate_classification_metrics(y_true, y_pred, y_prob)
    else:
        return calculate_regression_metrics(y_true, y_pred)

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate R², MSE, and RMSE metrics for regression evaluation."""
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return { 'r2_score': float(r2), 'mse': float(mse), 'rmse': float(rmse) }

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, and F1-score for classification evaluation."""
    if len(np.unique(y_true)) == 2:
        y_true_binary = (y_true == y_true[0]).astype(int)
        y_pred_binary = (y_pred == y_pred[0]).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score)
        }
        
        if y_prob is not None:
            optimal_threshold, optimal_accuracy = find_optimal_threshold(y_true_binary, y_prob)
            metrics['optimal_threshold'] = float(optimal_threshold)
            metrics['optimal_accuracy'] = float(optimal_accuracy)
        
        return metrics
    else:
        accuracy = np.mean(y_true == y_pred)
        return {
            'accuracy': float(accuracy),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and testing sets using sklearn's train_test_split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_and_evaluate(data_file: str, server_url: str = "http://localhost:8000") -> Dict[str, float]:
    """Train model on training data and evaluate performance on test set."""
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
        
        is_classification = len(np.unique(y)) <= 10
        
        metrics = calculate_metrics(y_test, y_pred, is_classification)
        
        print("\nModel Performance Metrics:")
        if is_classification:
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            if 'optimal_threshold' in metrics:
                print(f"Optimal Threshold: {metrics['optimal_threshold']:.3f}")
                print(f"Optimal Accuracy: {metrics['optimal_accuracy']:.4f}")
        else:
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