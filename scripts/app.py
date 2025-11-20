from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

import pandas as pd
from scripts.inference import ChurnPredictor
from scripts.config import API_TITLE, API_DESCRIPTION, API_VERSION

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Load the model on startup
@app.on_event("startup")
async def startup_event():
    """Load the ML model when the FastAPI application starts."""
    try:
        ChurnPredictor.load_model()
        logging.info("FastAPI app started and ML model loaded.")
    except Exception as e:
        logging.error(f"Failed to load ML model on startup: {e}")
        raise RuntimeError(f"Could not load model: {e}") # Raise to prevent app from starting without model

# Define input data model for a single transaction
class TransactionData(BaseModel):
    order_id: str
    order_purchase_timestamp: datetime
    customer_id: str
    customer_unique_id: str
    customer_state: str
    product_id: Optional[str] = None
    product_category_name: Optional[str] = None
    order_item_id: Optional[int] = None
    price: Optional[float] = None
    freight_value: Optional[float] = None
    payment_type: Optional[str] = None
    payment_value: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "order_id": "87e35b02660d5b62b19280f2d4757c2a",
                "order_purchase_timestamp": "2018-01-01T10:30:00",
                "customer_id": "a979201509a25032a1061993213009a",
                "customer_unique_id": "1a080928e46957a0640d21e8e8055bf9",
                "customer_state": "SP",
                "product_id": "b1b46a36f94a732d8479e0f10c69d80c",
                "product_category_name": "eletronicos",
                "order_item_id": 1,
                "price": 29.9,
                "freight_value": 7.39,
                "payment_type": "credit_card",
                "payment_value": 37.29
            }
        }

# Define input data model for a list of transactions for one or more customers
class CustomerTransactionsInput(BaseModel):
    current_date: datetime = Field(..., description="The date as of which to make the churn prediction.")
    transactions: List[TransactionData] = Field(..., description="List of all historical transactions for the customer(s) to be predicted within the observation window.")

    class Config:
        schema_extra = {
            "example": {
                "current_date": "2025-11-20T12:00:00",
                "transactions": [
                    {
                        "order_id": "87e35b02660d5b62b19280f2d4757c2a",
                        "order_purchase_timestamp": "2018-01-01T10:30:00",
                        "customer_id": "a979201509a25032a1061993213009a",
                        "customer_unique_id": "1a080928e46957a0640d21e8e8055bf9",
                        "customer_state": "SP",
                        "product_id": "b1b46a36f94a732d8479e0f10c69d80c",
                        "product_category_name": "eletronicos",
                        "order_item_id": 1,
                        "price": 29.9,
                        "freight_value": 7.39,
                        "payment_type": "credit_card",
                        "payment_value": 37.29
                    },
                    {
                        "order_id": "92f35b02660d5b62b19280f2d4757c2a",
                        "order_purchase_timestamp": "2018-02-15T14:00:00",
                        "customer_id": "a979201509a25032a1061993213009a",
                        "customer_unique_id": "1a080928e46957a0640d21e8e8055bf9",
                        "customer_state": "SP",
                        "product_id": "c2c46a36f94a732d8479e0f10c69d80c",
                        "product_category_name": "informatica_acessorios",
                        "order_item_id": 1,
                        "price": 59.9,
                        "freight_value": 10.0,
                        "payment_type": "boleto",
                        "payment_value": 69.9
                    }
                ]
            }
        }

# Define output data model
class ChurnPredictionResult(BaseModel):
    customer_unique_id: str
    churn_probability: float = Field(..., description="Probability of churn (0 to 1)")
    is_churn: int = Field(..., description="Binary prediction: 1 for churn, 0 for no churn")

class ChurnPredictionResponse(BaseModel):
    predictions: List[ChurnPredictionResult]

@app.get("/health", response_model=dict)
async def health_check():
    """Endpoint to check the health of the API."""
    if ChurnPredictor._pipeline is not None:
        return {"status": "ok", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False}

@app.post("/predict_churn", response_model=ChurnPredictionResponse)
async def predict_churn_endpoint(input_data: CustomerTransactionsInput):
    """
    Receives a list of historical transactions for one or more customers
    and predicts their churn status as of the specified current_date.
    """
    try:
        # Convert list of Pydantic models to a list of dictionaries, then to a Pandas DataFrame
        transactions_dicts = [transaction.dict() for transaction in input_data.transactions]
        transactions_df = pd.DataFrame(transactions_dicts)

        if transactions_df.empty:
            raise HTTPException(status_code=400, detail="No transactions provided for prediction.")

        # Ensure all required columns for preprocessing are present, even if None/NaN
        required_cols = [
            'order_id', 'order_purchase_timestamp', 'customer_id', 'customer_unique_id',
            'customer_state', 'product_id', 'product_category_name', 'order_item_id',
            'price', 'freight_value', 'payment_type', 'payment_value'
        ]
        for col in required_cols:
            if col not in transactions_df.columns:
                transactions_df[col] = None
        
        # Make predictions using the ChurnPredictor class
        predictions_df = ChurnPredictor.predict_churn(transactions_df, input_data.current_date)
        
        if predictions_df.empty:
            return ChurnPredictionResponse(predictions=[])

        # Convert prediction DataFrame to a list of Pydantic models
        predictions_list = predictions_df.to_dict(orient='records')
        return ChurnPredictionResponse(predictions=[ChurnPredictionResult(**p) for p in predictions_list])

    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")