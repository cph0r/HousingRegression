import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from utils import load_data, preprocess_data
import json

def perform_hyperparameter_tuning():
    """Perform hyperparameter tuning for multiple models"""
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    
    # Define models and their hyperparameters
    models = {
        'Ridge': {
            'model': Ridge(),
            'params': {
                'alpha': [0.1, 1.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky'],
                'fit_intercept': [True, False]
            }
        },
        'Lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [0.01, 0.1, 1.0],
                'max_iter': [1000, 5000, 10000],
                'tol': [0.0001, 0.001, 0.01]
            }
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        }
    }
    
    results = {}
    
    # Perform grid search for each model
    for name, config in models.items():
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best parameters and performance
        results[name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_score': grid_search.score(X_test, y_test)
        }
        
        # Save predictions for analysis
        predictions = grid_search.predict(X_test)
        np.save(f'predictions_{name.lower().replace(" ", "_")}_tuned.npy', predictions)
    
    # Save results
    with open('hyperparameter_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    results = perform_hyperparameter_tuning()
    print("\nHyperparameter Tuning Results:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        print("Best Parameters:", metrics['best_params'])
        print(f"Best CV Score: {metrics['best_score']:.4f}")
        print(f"Test Score: {metrics['test_score']:.4f}")

if __name__ == "__main__":
    main()
