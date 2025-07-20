# Fraud Detection for E-commerce and Banking Transactions

## Project Overview
This project focuses on developing advanced fraud detection models for e-commerce and banking transactions. It includes comprehensive data analysis, feature engineering, and machine learning model development to identify fraudulent activities effectively.

## Project Structure
```
fraud-detection-project/
├── data/                    # Data storage
│   ├── raw/                 # Original, immutable data
│   └── processed/           # Cleaned and processed data
├── notebooks/               # Jupyter notebooks for analysis
├── src/                     # Source code
│   ├── data/                # Data loading and preprocessing
│   ├── features/            # Feature engineering
│   ├── models/              # Model development
│   └── visualization/       # Visualization utilities
├── tests/                   # Unit tests
├── .gitignore
├── requirements.txt         # Project dependencies
└── README.md
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd fraud-detection-project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Lab/Notebook:
   ```bash
   jupyter lab
   ```

## Usage
1. Run the notebooks in the `notebooks/` directory in sequence:
   - `01_data_analysis.ipynb`: Initial data exploration and cleaning
   - `02_feature_engineering.ipynb`: Feature creation and selection
   - `03_model_development.ipynb`: Model training and evaluation

2. Or use the Python modules:
   ```python
   from src.data.load_data import load_transaction_data
   from src.features.build_features import create_features
   from src.models.train_model import train_fraud_detection_model
   
   # Example usage
   df = load_transaction_data('data/raw/Fraud_Data.csv')
   X, y = create_features(df)
   model = train_fraud_detection_model(X, y)
   ```

## Key Features
- Comprehensive EDA with fraud-specific insights
- Advanced feature engineering for fraud detection
- Handling of class imbalance using SMOTE/ADASYN
- Model explainability using SHAP values
- Modular and well-documented codebase

## Evaluation Metrics
- Precision-Recall AUC (primary metric for imbalanced data)
- F1-Score
- Confusion Matrix
- Feature Importance Analysis

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.