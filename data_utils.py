import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Set
import csv
from io import StringIO

class DataCleaner:
    def __init__(self):
        self.feature_means = None
        self.feature_stds = None
        self.target_mean = None
        self.target_std = None

    def clean_data(self, df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Clean and preprocess input data, returning feature matrix and target vector."""
        df = df.copy()
        
        allowed_types = [np.number, 'object']
        df = df.select_dtypes(include=allowed_types)
        
        df = self._handle_missing_values(df)
        df = self._handle_categorical_variables(df)
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X = self._normalize_features(X)
        y = self._normalize_target(y)
        
        return X.values, y.values

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values by filling with median for numeric and mode for categorical."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        return df

    def _handle_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical variables to numeric using one-hot encoding for columns with ≤25 unique values."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        cols_to_encode = []
        for col in categorical_cols:
            if df[col].nunique() <= 25:
                 cols_to_encode.append(col)
            else:
                 df.drop(columns=[col], inplace=True)
        if cols_to_encode:
             df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
        return df

    def _normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using z-score normalization."""
        if self.feature_means is None:
            self.feature_means = X.mean()
            self.feature_stds = X.std()
        
        self.feature_stds = self.feature_stds.replace(0, 1)
        
        return (X - self.feature_means) / self.feature_stds

    def _normalize_target(self, y: pd.Series) -> pd.Series:
        """Normalize target variable using z-score normalization."""
        if self.target_mean is None:
            self.target_mean = y.mean()
            self.target_std = y.std()
        
        if self.target_std == 0:
            self.target_std = 1
            
        return (y - self.target_mean) / self.target_std

    def inverse_transform_target(self, y_normalized: np.ndarray) -> np.ndarray:
        """Convert normalized target values back to original scale."""
        if self.target_mean is None or self.target_std is None:
            raise ValueError("Model must be trained before inverse transformation")
        return y_normalized * self.target_std + self.target_mean

def load_and_clean_data(file_path: str, target_column: str) -> Tuple[np.ndarray, np.ndarray, DataCleaner]:
    """Load data from CSV and perform cleaning, returning features, target, and cleaner instance."""
    df = pd.read_csv(file_path)
    
    cleaner = DataCleaner()
    X, y = cleaner.clean_data(df, target_column)
    
    return X, y, cleaner

def parse_csv(csv_text: str) -> Tuple[List[str], List[List[str]]]:
    """Parse raw CSV text into headers and rows."""
    csv_file = StringIO(csv_text)
    reader = csv.reader(csv_file)
    headers = next(reader)
    rows = list(reader)
    return headers, rows

def is_numeric_column(values: List[str]) -> bool:
    """Check if column contains only parseable floats or blanks."""
    for value in values:
        if value.strip() == "":
            continue
        try:
            float(value)
        except ValueError:
            return False
    return True

def get_numeric_columns(headers: List[str], rows: List[List[str]]) -> Set[int]:
    """Identify which columns contain only numeric data."""
    numeric_cols = set()
    columns = list(zip(*rows))
    
    for col_idx, column in enumerate(columns):
        if is_numeric_column(column):
            numeric_cols.add(col_idx)
    
    return numeric_cols

def handle_missing_values(rows: List[List[str]], numeric_cols: Set[int]) -> Tuple[List[List[str]], List[int]]:
    """Handle missing values in numeric columns by replacing with column means."""
    if not rows:
        return [], []
    
    data = np.array(rows)
    valid_row_indices = []
    
    col_means = {}
    for col_idx in numeric_cols:
        valid_values = []
        for row_idx, value in enumerate(data[:, col_idx]):
            try:
                if value.strip():
                    valid_values.append(float(value))
            except ValueError:
                continue
        if valid_values:
            col_means[col_idx] = np.mean(valid_values)
    
    cleaned_rows = []
    for row_idx, row in enumerate(rows):
        has_valid_data = False
        cleaned_row = row.copy()
        
        for col_idx in numeric_cols:
            value = row[col_idx].strip()
            try:
                if value:
                    float(value)
                    has_valid_data = True
                else:
                    cleaned_row[col_idx] = str(col_means[col_idx])
            except ValueError:
                cleaned_row[col_idx] = str(col_means[col_idx])
        
        if has_valid_data:
            cleaned_rows.append(cleaned_row)
            valid_row_indices.append(row_idx)
    
    return cleaned_rows, valid_row_indices

def prepare_data(csv_text: str, target_column: str, target_type: str = "numeric") -> Tuple[np.ndarray, np.ndarray]:
    """Process CSV data into feature matrix and target vector."""
    headers, rows = parse_csv(csv_text)
    print("Parsed headers from CSV:", headers)
    numeric_cols = get_numeric_columns(headers, rows)
    try:
         target_idx = headers.index(target_column)
    except ValueError:
         raise ValueError(f"Target column '{target_column}' not found in headers. Available headers: {headers}.")
    cleaned_rows, valid_indices = handle_missing_values(rows, numeric_cols)
    if not cleaned_rows:
         raise ValueError("No valid numeric data found after cleaning")
    if target_type == "numeric":
         feature_indices = [i for i in numeric_cols if i != target_idx]
    else:
         feature_indices = numeric_cols
    X = np.array([[float(row[i]) for i in feature_indices] for row in cleaned_rows])
    if target_type == "categorical":
         y = np.array([row[target_idx] for row in cleaned_rows])
    else:
         y = np.array([float(row[target_idx]) for row in cleaned_rows])
    return X, y

def clean_data(data):
    """Basic data cleaning function."""
    pass 