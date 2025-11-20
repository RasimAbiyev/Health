# config.py

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Dataset Specific ---
CUSTOMER_DATASET_PATH = os.path.join(DATA_DIR, 'olist_customers_dataset.csv')
ORDER_DATASET_PATH = os.path.join(DATA_DIR, 'olist_orders_dataset.csv')
ORDER_ITEMS_DATASET_PATH = os.path.join(DATA_DIR, 'olist_order_items_dataset.csv')
PRODUCTS_DATASET_PATH = os.path.join(DATA_DIR, 'olist_products_dataset.csv')
ORDER_PAYMENTS_DATASET_PATH = os.path.join(DATA_DIR, 'olist_order_payments_dataset.csv') # Added for monetary value

PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_customer_features.csv')
TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'training_data.csv')

# --- Model Specific ---
MODEL_FILE_PATH = os.path.join(MODELS_DIR, 'churn_prediction_pipeline.pkl')
MODEL_NAME = "Olist_Churn_Predictor"
MODEL_VERSION = "1.0.0"

# --- Feature Engineering Parameters ---
SNAPSHOT_DATE_OFFSET_DAYS = 90  # Defines the end of observation window and start of prediction window
OBSERVATION_WINDOW_DAYS = 180  # e.g., features from last 6 months
PREDICTION_WINDOW_DAYS = 90    # e.g., predict churn for next 3 months

# List of numerical and categorical features for the final pipeline
NUMERICAL_FEATURES = [
    'Recency', 'Frequency', 'Monetary',
    'avg_order_value', 'total_items_purchased', 'num_unique_products',
    'avg_freight_value', 'time_since_first_purchase_days',
    'order_velocity_per_month', 'avg_review_score'
]

CATEGORICAL_FEATURES = [
    'customer_state', 'most_frequent_product_category', 'payment_type_mode'
]

# --- Training Parameters ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Optimal hyperparameters for the LGBMClassifier (example values)
# These would come from the Optimization Notebook
LGBM_MODEL_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# --- API/App Parameters ---
API_TITLE = "Olist Customer Churn Prediction API"
API_DESCRIPTION = "Predicts customer churn status for Olist customers."
API_VERSION = "1.0.0"