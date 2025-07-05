import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from utils import load_data, preprocess_data, evaluate_model
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_models():
    """Train and evaluate multiple regression models"""
    # Load and preprocess data
    df = load_data()
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
        print(f"{name}: MSE={mse:.2f}, R2={r2:.2f}")
    
    # Save results
    with open('regression_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    results = train_models()
    print("\nRegression Model Performance:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"MSE: {metrics['MSE']:.4f}")
        print(f"R² Score: {metrics['R2']:.4f}")

if __name__ == "__main__":
    main()
