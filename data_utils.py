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
        """
        Clean and preprocess the input data.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target variable column
            
        Returns:
            Tuple of (X, y) where X is the feature matrix and y is the target vector
        """
        # Make a copy to avoid modifying the original data
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle categorical variables
        df = self._handle_categorical_variables(df)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Normalize features
        X = self._normalize_features(X)
        
        # Normalize target
        y = self._normalize_target(y)
        
        return X.values, y.values

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values by filling with appropriate values."""
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        return df

    def _handle_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical variables to numeric using one-hot encoding."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        return df

    def _normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using z-score normalization."""
        if self.feature_means is None:
            self.feature_means = X.mean()
            self.feature_stds = X.std()
        
        # Avoid division by zero
        self.feature_stds = self.feature_stds.replace(0, 1)
        
        return (X - self.feature_means) / self.feature_stds

    def _normalize_target(self, y: pd.Series) -> pd.Series:
        """Normalize target variable using z-score normalization."""
        if self.target_mean is None:
            self.target_mean = y.mean()
            self.target_std = y.std()
        
        # Avoid division by zero
        if self.target_std == 0:
            self.target_std = 1
            
        return (y - self.target_mean) / self.target_std

    def inverse_transform_target(self, y_normalized: np.ndarray) -> np.ndarray:
        """Convert normalized target values back to original scale."""
        if self.target_mean is None or self.target_std is None:
            raise ValueError("Model must be trained before inverse transformation")
        return y_normalized * self.target_std + self.target_mean

def load_and_clean_data(file_path: str, target_column: str) -> Tuple[np.ndarray, np.ndarray, DataCleaner]:
    """
    Load data from CSV and perform cleaning.
    
    Args:
        file_path: Path to the CSV file
        target_column: Name of the target variable column
        
    Returns:
        Tuple of (X, y, data_cleaner) where:
        - X is the feature matrix
        - y is the target vector
        - data_cleaner is the DataCleaner instance used for preprocessing
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Initialize and use the data cleaner
    cleaner = DataCleaner()
    X, y = cleaner.clean_data(df, target_column)
    
    return X, y, cleaner

def parse_csv(csv_text: str) -> Tuple[List[str], List[List[str]]]:
    """
    Parse raw CSV text into headers and rows.
    
    Args:
        csv_text: Raw CSV text including headers
        
    Returns:
        Tuple of (headers, rows) where:
        - headers is a list of column names
        - rows is a list of lists containing the data
    """
    csv_file = StringIO(csv_text)
    reader = csv.reader(csv_file)
    headers = next(reader)  # Get the header row
    rows = list(reader)     # Get all data rows
    return headers, rows

def is_numeric_column(values: List[str]) -> bool:
    """
    Check if a column contains only parseable floats or blanks.
    
    Args:
        values: List of string values from a column
        
    Returns:
        True if column contains only valid floats or blanks, False otherwise
    """
    for value in values:
        if value.strip() == "":  # Allow blank values
            continue
        try:
            float(value)
        except ValueError:
            return False
    return True

def get_numeric_columns(headers: List[str], rows: List[List[str]]) -> Set[int]:
    """
    Identify which columns contain only numeric data.
    
    Args:
        headers: List of column names
        rows: 2D list of data values
        
    Returns:
        Set of column indices that contain only numeric data
    """
    numeric_cols = set()
    # Transpose rows to get columns
    columns = list(zip(*rows))
    
    for col_idx, column in enumerate(columns):
        if is_numeric_column(column):
            numeric_cols.add(col_idx)
    
    return numeric_cols

def handle_missing_values(rows: List[List[str]], numeric_cols: Set[int]) -> Tuple[List[List[str]], List[int]]:
    """
    Handle missing values in numeric columns.
    
    Args:
        rows: 2D list of data values
        numeric_cols: Set of numeric column indices
        
    Returns:
        Tuple of (cleaned_rows, valid_row_indices) where:
        - cleaned_rows has missing values replaced with column means
        - valid_row_indices contains indices of rows that had at least one valid numeric value
    """
    if not rows:
        return [], []
    
    # Convert rows to numpy array for easier processing
    data = np.array(rows)
    valid_row_indices = []
    
    # Calculate means for each numeric column
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
    
    # Replace missing values and track valid rows
    cleaned_rows = []
    for row_idx, row in enumerate(rows):
        has_valid_data = False
        cleaned_row = row.copy()
        
        for col_idx in numeric_cols:
            value = row[col_idx].strip()
            try:
                if value:
                    float(value)  # Verify it's a valid float
                    has_valid_data = True
                else:
                    cleaned_row[col_idx] = str(col_means[col_idx])
            except ValueError:
                cleaned_row[col_idx] = str(col_means[col_idx])
        
        if has_valid_data:
            cleaned_rows.append(cleaned_row)
            valid_row_indices.append(row_idx)
    
    return cleaned_rows, valid_row_indices

def prepare_data(csv_text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process CSV data into feature matrix and target vector.
    
    Args:
        csv_text: Raw CSV text including headers
        
    Returns:
        Tuple of (X, y) where:
        - X is the feature matrix (all numeric columns except the last)
        - y is the target vector (last numeric column)
    """
    # Parse CSV
    headers, rows = parse_csv(csv_text)
    
    # Identify numeric columns
    numeric_cols = get_numeric_columns(headers, rows)
    if len(numeric_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for features and target")
    
    # Handle missing values
    cleaned_rows, valid_indices = handle_missing_values(rows, numeric_cols)
    if not cleaned_rows:
        raise ValueError("No valid numeric data found after cleaning")
    
    # Convert to numeric matrix
    numeric_data = np.array([[float(val) for val in row] for row in cleaned_rows])
    
    # Split into features and target
    X = numeric_data[:, :-1]  # All but last column
    y = numeric_data[:, -1]   # Last column
    
    return X, y

def load_and_clean_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV file and perform cleaning.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (X, y) where:
        - X is the feature matrix
        - y is the target vector
    """
    with open(file_path, 'r') as f:
        csv_text = f.read()
    return prepare_data(csv_text)

def clean_data(data):
    """Basic data cleaning function."""
    pass 