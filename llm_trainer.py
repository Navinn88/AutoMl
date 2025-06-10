import google.generativeai as genai
import numpy as np
import pandas as pd

def feed_llm_and_train(data_file: str, target_column: str, api_key: str, context: str = None) -> dict:
    """Use Gemini to select best model type for given data and return model type."""
    # Load data without cleaning to avoid column issues
    df = pd.read_csv(data_file)
    
    # Basic data analysis for LLM prompt
    num_samples = len(df)
    num_features = len(df.columns) - 1  # Exclude target column
    target_values = df[target_column].unique()
    target_type = 'Continuous' if len(target_values) > 10 else 'Categorical'
    
    # Sample data for context
    sample_features = df.drop(columns=[target_column]).iloc[0].tolist() if num_features > 0 else []
    sample_targets = df[target_column].head(5).tolist()

    # Prepare the prompt for Gemini with data summary
    prompt = (
        f"You are an expert ML model selector. "
        f"Given the following data summary, choose the best model type (linear, logistic, or decision_tree) "
        f"and return ONLY the model type as a string.\n\n"
        f"Data Summary:\n"
        f"- Number of samples (rows): {num_samples}\n"
        f"- Number of features: {num_features}\n"
        f"- Target variable: {target_column}\n"
        f"- Target type: {target_type}\n"
        f"- Context: {context if context else 'None'}\n\n"
        f"Sample of features (first row): {sample_features}\n"
        f"Sample of target values (first 5): {sample_targets}\n\n"
        f"Respond with ONLY one of these exact strings: 'linear', 'logistic', or 'decision_tree'"
    )

    # Set up Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    # Clean and validate the response
    model_type = response.text.strip().lower()
    valid_models = ['linear', 'logistic', 'decision_tree']
    if model_type not in valid_models:
        raise ValueError(f"Invalid model type returned: {model_type}. Must be one of {valid_models}")
    return {"model_type": model_type}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Use Gemini to select and train model on data.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--api_key", type=str, required=True, help="Gemini API key")
    parser.add_argument("--context", type=str, required=False, help="Optional context for model selection")
    args = parser.parse_args()
    model = feed_llm_and_train(args.data, args.target, args.api_key, context=args.context)
    print("Gemini chose model type:", model["model_type"]) 