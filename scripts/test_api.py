"""
Test script for the Churn Prediction API
This script tests the API endpoints with sample data
"""

import requests
import json
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)
    return response.status_code == 200

def test_churn_prediction():
    """Test the churn prediction endpoint"""
    print("Testing churn prediction endpoint...")
    
    # Sample transaction data
    data = {
        "current_date": datetime.now().isoformat(),
        "transactions": [
            {
                "order_id": "87e35b02660d5b62b19280f2d4757c2a",
                "order_purchase_timestamp": (datetime.now() - timedelta(days=30)).isoformat(),
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
                "order_purchase_timestamp": (datetime.now() - timedelta(days=60)).isoformat(),
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
    
    response = requests.post(f"{BASE_URL}/predict_churn", json=data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result['predictions']:
            for pred in result['predictions']:
                print(f"\nCustomer: {pred['customer_unique_id']}")
                print(f"Churn Probability: {pred['churn_probability']:.2%}")
                print(f"Will Churn: {'Yes' if pred['is_churn'] == 1 else 'No'}")
    else:
        print(f"Error: {response.text}")
    
    print("-" * 50)
    return response.status_code == 200

def main():
    """Run all tests"""
    print("=" * 50)
    print("Churn Prediction API Test Suite")
    print("=" * 50)
    print()
    
    try:
        # Test health check
        health_ok = test_health_check()
        
        if health_ok:
            # Test prediction
            prediction_ok = test_churn_prediction()
            
            if health_ok and prediction_ok:
                print("\n✓ All tests passed!")
            else:
                print("\n✗ Some tests failed")
        else:
            print("\n✗ API is not healthy. Make sure the server is running.")
            print("Start the server with: python -m uvicorn scripts.app:app --host 0.0.0.0 --port 8000")
    
    except requests.exceptions.ConnectionError:
        print("\n✗ Could not connect to the API server.")
        print("Make sure the server is running with:")
        print("python -m uvicorn scripts.app:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()
