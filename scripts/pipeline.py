import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from scripts.config import (
    CUSTOMER_DATASET_PATH, ORDER_DATASET_PATH, ORDER_ITEMS_DATASET_PATH, PRODUCTS_DATASET_PATH,
    ORDER_PAYMENTS_DATASET_PATH, MODEL_FILE_PATH, TRAINING_DATA_PATH,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    SNAPSHOT_DATE_OFFSET_DAYS, OBSERVATION_WINDOW_DAYS, PREDICTION_WINDOW_DAYS,
    TEST_SIZE, RANDOM_STATE, LGBM_MODEL_PARAMS
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    logging.info("Loading raw Olist datasets...")
    customers_df = pd.read_csv(CUSTOMER_DATASET_PATH)
    orders_df = pd.read_csv(ORDER_DATASET_PATH)
    order_items_df = pd.read_csv(ORDER_ITEMS_DATASET_PATH)
    products_df = pd.read_csv(PRODUCTS_DATASET_PATH)
    order_payments_df = pd.read_csv(ORDER_PAYMENTS_DATASET_PATH)

    df = pd.merge(orders_df, customers_df, on='customer_id', how='left')
    df = pd.merge(df, order_items_df, on='order_id', how='left')
    df = pd.merge(df, products_df[['product_id', 'product_category_name']], on='product_id', how='left')
    df = pd.merge(df, order_payments_df, on='order_id', how='left')

    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    logging.info("Raw data loaded and merged. Shape: %s", df.shape)
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting feature engineering...")

    max_purchase_date = df['order_purchase_timestamp'].max()
    snapshot_date = max_purchase_date - timedelta(days=SNAPSHOT_DATE_OFFSET_DAYS)
    logging.info("Using snapshot date for feature engineering: %s", snapshot_date)

    observation_end = snapshot_date
    observation_start = observation_end - timedelta(days=OBSERVATION_WINDOW_DAYS)
    df_obs = df[(df['order_purchase_timestamp'] >= observation_start) &
                (df['order_purchase_timestamp'] < observation_end)].copy()

    customer_features = df_obs.groupby('customer_unique_id').agg(
        last_purchase=('order_purchase_timestamp', 'max'),
        total_orders=('order_id', 'nunique'),
        total_items_purchased=('order_item_id', 'count'),
        total_monetary=('payment_value', 'sum'),
        avg_order_value=('payment_value', 'mean'),
        num_unique_products=('product_id', 'nunique'),
        avg_freight_value=('freight_value', 'mean'),
        first_purchase=('order_purchase_timestamp', 'min')
    ).reset_index()

    customer_features['Recency'] = (snapshot_date - customer_features['last_purchase']).dt.days
    customer_features['Frequency'] = customer_features['total_orders']
    customer_features['Monetary'] = customer_features['total_monetary']
    customer_features['time_since_first_purchase_days'] = (snapshot_date - customer_features['first_purchase']).dt.days
    customer_features['order_velocity_per_month'] = (customer_features['Frequency'] /
                                                    (customer_features['time_since_first_purchase_days'] / 30)).fillna(0)

    most_freq_cat = df_obs.groupby('customer_unique_id')['product_category_name'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
    ).reset_index(name='most_frequent_product_category')
    customer_features = pd.merge(customer_features, most_freq_cat, on='customer_unique_id', how='left')

    customer_state = df_obs.groupby('customer_unique_id')['customer_state'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
    ).reset_index(name='customer_state')
    customer_features = pd.merge(customer_features, customer_state, on='customer_unique_id', how='left')

    payment_type_mode = df_obs.groupby('customer_unique_id')['payment_type'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
    ).reset_index(name='payment_type_mode')
    customer_features = pd.merge(customer_features, payment_type_mode, on='customer_unique_id', how='left')

    customer_features['avg_review_score'] = 0

    prediction_start = snapshot_date
    prediction_end = prediction_start + timedelta(days=PREDICTION_WINDOW_DAYS)
    df_pred = df[(df['order_purchase_timestamp'] >= prediction_start) &
                 (df['order_purchase_timestamp'] <= prediction_end)].copy()
    customers_with_future_purchase = df_pred['customer_unique_id'].unique()
    customer_features['is_churn'] = customer_features['customer_unique_id'].apply(
        lambda x: 0 if x in customers_with_future_purchase else 1
    )

    logging.info("Feature engineering complete. Dataset shape: %s", customer_features.shape)
    logging.info("Churn distribution: \n%s", customer_features['is_churn'].value_counts(normalize=True))
    return customer_features

def build_and_train_pipeline(df_train: pd.DataFrame):
    logging.info("Building and training ML pipeline...")

    numerical_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(**LGBM_MODEL_PARAMS))
    ])

    X = df_train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df_train['is_churn']

    # Temizleme: NaN, inf, -inf değerler
    for col in X.columns:
        if X[col].dtype == 'object':
            X.loc[:, col] = X[col].fillna('unknown')
        else:
            X.loc[:, col] = X[col].replace([np.inf, -np.inf], np.nan)
            X.loc[:, col] = X[col].fillna(X[col].median())

    pipeline.fit(X, y)
    logging.info("ML Pipeline trained successfully.")
    return pipeline

def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    logging.info("Evaluating pipeline on test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    logging.info(f"Test Set Evaluation:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}")

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}

def run_training_pipeline():
    df_raw = load_data()
    df_features = feature_engineer(df_raw)

    df_features.to_csv(TRAINING_DATA_PATH, index=False)
    logging.info("Engineered features saved to %s", TRAINING_DATA_PATH)

    X = df_features[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df_features['is_churn']

    # NaN ve inf değerleri temizle
    for col in X.columns:
        if X[col].dtype == 'object':
            X.loc[:, col] = X[col].fillna('unknown')
        else:
            X.loc[:, col] = X[col].replace([np.inf, -np.inf], np.nan)
            X.loc[:, col] = X[col].fillna(X[col].median())


    # ===================================================================
    # Validation Strategy: Stratified Train-Test Split (80/20)
    # ===================================================================
    # Why stratified split? Ensures same churn distribution in train/test (~95%)
    # Why NOT K-Fold? Computational cost + temporal leakage + diminishing returns
    # Why NOT time-based? Limited data (2 years) + already temporal features
    # See README "Validation Strategy" section for detailed justification
    # For production monitoring: scripts/monitoring.py
    # ===================================================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    logging.info("Data split into training (%s samples) and test (%s samples) sets.", len(X_train), len(X_test))

    trained_pipeline = build_and_train_pipeline(X_train.assign(is_churn=y_train))
    evaluation_metrics = evaluate_pipeline(trained_pipeline, X_test, y_test)

    joblib.dump(trained_pipeline, MODEL_FILE_PATH)
    logging.info("Trained model pipeline saved to %s", MODEL_FILE_PATH)

    return evaluation_metrics

if __name__ == "__main__":
    logging.info("Starting Olist Churn Prediction Training Pipeline.")
    metrics = run_training_pipeline()
    logging.info("Training pipeline finished. Final metrics: %s", metrics)