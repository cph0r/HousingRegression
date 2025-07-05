# Housing Regression ML Pipeline

**GitHub Repo:** [PLACE_GITHUB_LINK_HERE]

## Overview
This project implements a complete machine learning workflow to predict house prices using the Boston Housing dataset. It compares multiple regression models and automates the workflow using GitHub Actions.

## Setup
1. Create and activate a conda environment:
   ```bash
   conda create -n housing-regression python=3.10 -y
   conda activate housing-regression
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Models
```bash
python regression.py
```

## Project Structure
- `.github/workflows/ci.yml`: CI pipeline
- `utils.py`: Utility functions
- `regression.py`: Regression models and evaluation
- `requirements.txt`: Dependencies
- `README.md`: Project info

## Project Structure

```
HousingRegression/
├── .github/
│   └── workflows/
│       └── ci.yml
├── utils.py           # Utility functions for data loading and preprocessing
├── regression.py      # Main regression model implementation
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Setup Instructions

1. Create a conda environment:
```bash
conda create -n housing-env python=3.8
conda activate housing-env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the regression models:
```bash
python regression.py
```

2. Run visualization:
```bash
python visualization.py
```

3. Results will be saved in:
- `regression_results.json` and `hyperparameter_results.json`
- Prediction files: `predictions_*.npy`
- Visualization plots: `*.png` files

## Visualization Features

The project includes visualization capabilities that generate:
- Feature importance plots for tree-based models
- Residual plots to analyze model performance
- Model comparison plots showing MSE and R² scores
- All plots are automatically saved as PNG files

## GitHub Workflow

The project includes a GitHub Actions workflow that:
- Runs on push events
- Installs dependencies
- Executes the regression pipeline
- Saves results

## Branch Structure

- `main`: Contains only README.md initially
- `reg`: Regression model implementation
- `hyper`: Hyperparameter tuning implementation
- All branches must be preserved for submission

## Performance Metrics

The project compares models using:
- Mean Squared Error (MSE)
- R² Score

## License

This project is for educational purposes only.

## Note

All branches (main, reg, hyper) must be present at submission time. Do not delete any branch.
