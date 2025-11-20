import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

from scripts.config import (
    MODEL_FILE_PATH,
    SNAPSHOT_DATE_OFFSET_DAYS, OBSERVATION_WINDOW_DAYS,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)
from scripts.pipeline import load_data  # Reuse data loading for consistency

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChurnPredictor:
    """
    Manages loading the trained ML pipeline and making churn predictions.
    """
    _pipeline = None
    _features_order = None # To ensure consistent feature order

    @classmethod
    def load_model(cls):
        """Loads the pre-trained ML pipeline."""
        if cls._pipeline is None:
            logging.info("Loading churn prediction pipeline from %s", MODEL_FILE_PATH)
            cls._pipeline = joblib.load(MODEL_FILE_PATH)
            logging.info("Model pipeline loaded successfully.")
            # Capture feature names expected by the model after preprocessing
            # This is tricky with ColumnTransformer + OneHotEncoder.
            # A robust way is to train a dummy pipeline and extract feature names.
            # For simplicity, we assume NUMERICAL_FEATURES + CATEGORICAL_FEATURES (one-hot encoded) match.
            cls._features_order = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        return cls._pipeline

    @staticmethod
    def preprocess_new_customer_data(new_customer_transactions: pd.DataFrame, current_date: datetime) -> pd.DataFrame:
        """
        Processes new transactional data for a given customer(s) to create features
        suitable for the churn prediction model.
        This mirrors the feature engineering logic in pipeline.py.
        """
        logging.info("Preprocessing new customer data...")

        # Ensure order_purchase_timestamp is datetime
        new_customer_transactions['order_purchase_timestamp'] = pd.to_datetime(new_customer_transactions['order_purchase_timestamp'])

        # For inference, the snapshot date is 'current_date'
        snapshot_date = current_date

        observation_end = snapshot_date
        observation_start = observation_end - timedelta(days=OBSERVATION_WINDOW_DAYS)
        df_obs = new_customer_transactions[(new_customer_transactions['order_purchase_timestamp'] >= observation_start) &
                                           (new_customer_transactions['order_purchase_timestamp'] < observation_end)].copy()

        # Handle cases where a customer has no activity in the observation window
        if df_obs.empty:
            logging.warning("No activity found for customer(s) in the observation window. Returning empty DataFrame.")
            # Return empty DataFrame with proper structure
            return pd.DataFrame(columns=NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ['customer_unique_id'])


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

        customer_features['avg_review_score'] = 0 # Placeholder consistent with training

        # Select and reorder features to match training expectation
        final_features = customer_features[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ['customer_unique_id']]

        # Impute missing values for inference (e.g., if a customer has no transactions)
        for col in NUMERICAL_FEATURES:
            if col in final_features.columns:
                final_features[col] = final_features[col].fillna(final_features[col].median()) # Use median from training if available
            else: # If a feature is missing entirely, add it with 0 or default
                final_features[col] = 0
        for col in CATEGORICAL_FEATURES:
            if col in final_features.columns:
                final_features[col] = final_features[col].fillna('unknown')
            else:
                final_features[col] = 'unknown'

        logging.info("Preprocessing complete for %s customers.", len(final_features))
        return final_features


    @classmethod
    def predict_churn(cls, customer_transactions_df: pd.DataFrame, current_date: datetime) -> pd.DataFrame:
        """
        Predicts churn for new customer data.
        customer_transactions_df should contain raw transactional history
        for the customer(s) to be predicted.
        """
        pipeline = cls.load_model()
        
        # Preprocess the incoming data to create features
        features_df = cls.preprocess_new_customer_data(customer_transactions_df, current_date)
        
        if features_df.empty:
            return pd.DataFrame({'customer_unique_id': [], 'churn_probability': [], 'is_churn': []})

        customer_ids = features_df['customer_unique_id']
        X_inference = features_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

        # Make predictions
        churn_probabilities = pipeline.predict_proba(X_inference)[:, 1]
        is_churn = pipeline.predict(X_inference)

        results = pd.DataFrame({
            'customer_unique_id': customer_ids,
            'churn_probability': churn_probabilities,
            'is_churn': is_churn
        })
        logging.info("Predictions made for %s customers.", len(results))
        return results

if __name__ == "__main__":
    logging.info("Running inference.py as a standalone script for demonstration.")

    # Example: Simulate loading data for a few customers (need to run pipeline.py first to create model)
    # This example requires 'data/olist_*.csv' files to be present.
    full_df = load_data() # Load full dataset to simulate pulling transactions for specific customers

    # Select a few random customers for demonstration
    sample_customer_ids = full_df['customer_unique_id'].sample(5, random_state=42).tolist()
    
    # Get all transactions for these sample customers
    sample_transactions_df = full_df[full_df['customer_unique_id'].isin(sample_customer_ids)]
    
    # Use a recent date as 'current_date' for inference
    current_inference_date = datetime.now() - timedelta(days=1) # Yesterday

    logging.info(f"Predicting churn for sample customers as of {current_inference_date}...")
    predictions = ChurnPredictor.predict_churn(sample_transactions_df, current_inference_date)
    print("\nChurn Predictions:")
    print(predictions)

    # Example of a customer with no recent activity (should be high churn prob)
    # Find a customer with very old last purchase
    old_customer_id = full_df.groupby('customer_unique_id')['order_purchase_timestamp'].max().sort_values().index[0]
    old_customer_transactions = full_df[full_df['customer_unique_id'] == old_customer_id]
    
    logging.info(f"\nPredicting churn for an old customer ({old_customer_id}) as of {current_inference_date}...")
    old_customer_predictions = ChurnPredictor.predict_churn(old_customer_transactions, current_inference_date)
    print("\nOld Customer Churn Predictions:")
    print(old_customer_predictions)