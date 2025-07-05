import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import json

def plot_feature_importance(model, feature_names, title):
    """Plot feature importance for tree-based models"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance - {title}")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f"feature_importance_{title.lower().replace(' ', '_')}.png")
    plt.close()

def plot_residuals(y_true, y_pred, title):
    """Plot residuals to check model performance"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f"Residual Plot - {title}")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(f"residuals_{title.lower().replace(' ', '_')}.png")
    plt.close()

def plot_model_comparison(results, title):
    """Plot comparison of different models"""
    model_names = list(results.keys())
    mse_values = [results[model]['MSE'] for model in model_names]
    r2_values = [results[model]['R²'] for model in model_names]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # MSE (left y-axis)
    color = 'tab:blue'
    ax1.set_xlabel('Models')
    ax1.set_ylabel('MSE', color=color)
    ax1.bar(model_names, mse_values, color=color, alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # R² (right y-axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('R² Score', color=color)
    ax2.plot(model_names, r2_values, color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"Model Performance Comparison - {title}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"model_comparison_{title.lower().replace(' ', '_')}.png")
    plt.close()

def analyze_results(results_file):
    """Analyze and visualize model results"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load feature names
    df = pd.read_csv("data.csv")
    feature_names = df.columns[:-1]  # Exclude target variable
    
    # Create comparison plot
    plot_model_comparison(results, "Regression")
    
    # For each model, plot residuals if predictions are available
    for model in results:
        try:
            predictions = np.load(f'predictions_{model.lower().replace(" ", "_")}.npy')
            plot_residuals(df['MEDV'], predictions, model)
            
            # If model is tree-based, plot feature importance
            if model == "Random Forest":
                model_obj = RandomForestRegressor()  # Load model with best params
                model_obj.fit(df[feature_names], df['MEDV'])
                plot_feature_importance(model_obj, feature_names, model)
        except FileNotFoundError:
            continue

def main():
    # Analyze regression results
    analyze_results('regression_results.json')
    
    # Analyze hyperparameter tuning results
    try:
        analyze_results('hyperparameter_results.json')
    except FileNotFoundError:
        print("Hyperparameter results not found")

if __name__ == "__main__":
    main()
