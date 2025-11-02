"""
Client Examples for Loan Default Prediction API
Demonstrates how to call the deployed model API
"""

import requests
import json
import time
import asyncio
import aiohttp
from typing import Dict, List, Any
import pandas as pd
import numpy as np

class LoanPredictionClient:
    """Client for interacting with the Loan Default Prediction API"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        
    def check_api_health(self) -> Dict[str, Any]:
        """Check if the API is healthy and ready"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def list_available_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        try:
            response = self.session.get(f"{self.api_url}/models")
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        try:
            response = self.session.get(f"{self.api_url}/model-info")
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def predict_single(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction for a single loan application"""
        try:
            response = self.session.post(f"{self.api_url}/predict", json=loan_data)
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def predict_batch(self, applications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make predictions for multiple loan applications"""
        try:
            batch_request = {"applications": applications}
            response = self.session.post(f"{self.api_url}/predict-batch", json=batch_request)
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

def example_1_basic_usage():
    """Example 1: Basic API usage"""
    print("🎯 Example 1: Basic API Usage")
    print("=" * 50)
    
    # Initialize client
    client = LoanPredictionClient()
    
    # Check API health
    print("1. Checking API health...")
    health = client.check_api_health()
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Model loaded: {health.get('model_loaded', False)}")
    print(f"   Current model: {health.get('model_name', 'none')}")
    
    # Get model information
    print("\n2. Getting model information...")
    model_info = client.get_model_info()
    if "error" not in model_info:
        print(f"   Model: {model_info.get('model_name', 'unknown')}")
        print(f"   Type: {model_info.get('model_type', 'unknown')}")
        print(f"   Features: {model_info.get('features_count', 0)}")
        metrics = model_info.get('performance_metrics', {})
        print(f"   ROC AUC: {metrics.get('roc_auc', 0):.4f}")
        print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
    else:
        print(f"   Error: {model_info['error']}")
    
    print("\n" + "=" * 50)

def example_2_single_prediction():
    """Example 2: Single loan application prediction"""
    print("🏠 Example 2: Single Loan Application Prediction")
    print("=" * 50)
    
    client = LoanPredictionClient()
    
    # Sample loan application data
    loan_application = {
        "amt_credit": 450000.0,
        "amt_annuity": 25000.0,
        "amt_income_total": 150000.0,
        "amt_goods_price": 400000.0,
        "code_gender": "M",
        "days_birth": -12000,  # ~33 years old
        "days_employed": -2000,  # ~5.5 years employed
        "name_contract_type": "Cash loans",
        "name_income_type": "Working",
        "name_education_type": "Higher education",
        "name_family_status": "Married",
        "name_housing_type": "House / apartment",
        "region_population_relative": 0.02,
        "ext_source_1": 0.7,
        "ext_source_2": 0.6,
        "ext_source_3": 0.8
    }
    
    print("Loan Application Details:")
    print(f"   Credit Amount: ${loan_application['amt_credit']:,.0f}")
    print(f"   Annual Income: ${loan_application['amt_income_total']:,.0f}")
    print(f"   Annuity: ${loan_application['amt_annuity']:,.0f}")
    print(f"   Gender: {loan_application['code_gender']}")
    print(f"   Education: {loan_application['name_education_type']}")
    print(f"   Employment: {abs(loan_application['days_employed'])} days")
    
    print("\nMaking prediction...")
    start_time = time.time()
    result = client.predict_single(loan_application)
    response_time = (time.time() - start_time) * 1000
    
    if "error" not in result:
        print(f"\n✅ Prediction Results:")
        print(f"   Application ID: {result['application_id']}")
        print(f"   Default Probability: {result['prediction']:.4f} ({result['prediction']*100:.2f}%)")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Model Used: {result['model_used']}")
        print(f"   Response Time: {response_time:.2f}ms")
        
        # Risk interpretation
        risk_interpretation = {
            "LOW": "✅ Low risk - Likely to repay the loan",
            "MEDIUM": "⚠️ Medium risk - Requires careful consideration",
            "HIGH": "❌ High risk - Likely to default"
        }
        print(f"\n📊 Risk Assessment: {risk_interpretation.get(result['risk_level'], 'Unknown')}")
    else:
        print(f"❌ Error: {result['error']}")
    
    print("\n" + "=" * 50)

def example_3_batch_predictions():
    """Example 3: Batch predictions for multiple applications"""
    print("📦 Example 3: Batch Predictions")
    print("=" * 50)
    
    client = LoanPredictionClient()
    
    # Generate multiple loan applications
    applications = [
        {
            "amt_credit": 300000.0,
            "amt_annuity": 15000.0,
            "amt_income_total": 80000.0,
            "code_gender": "F",
            "days_birth": -10000,
            "days_employed": -1500,
            "name_contract_type": "Cash loans",
            "name_income_type": "Working",
            "name_education_type": "Secondary / secondary special",
            "name_family_status": "Single / not married",
            "name_housing_type": "Rented apartment",
            "region_population_relative": 0.03,
            "ext_source_1": 0.5,
            "ext_source_2": 0.4,
            "ext_source_3": None
        },
        {
            "amt_credit": 800000.0,
            "amt_annuity": 40000.0,
            "amt_income_total": 200000.0,
            "code_gender": "M",
            "days_birth": -15000,
            "days_employed": -5000,
            "name_contract_type": "Cash loans",
            "name_income_type": "Commercial associate",
            "name_education_type": "Higher education",
            "name_family_status": "Married",
            "name_housing_type": "House / apartment",
            "region_population_relative": 0.015,
            "ext_source_1": 0.8,
            "ext_source_2": 0.9,
            "ext_source_3": 0.7
        },
        {
            "amt_credit": 150000.0,
            "amt_annuity": 8000.0,
            "amt_income_total": 50000.0,
            "code_gender": "F",
            "days_birth": -8000,
            "days_employed": -500,
            "name_contract_type": "Revolving loans",
            "name_income_type": "Working",
            "name_education_type": "Lower secondary",
            "name_family_status": "Single / not married",
            "name_housing_type": "With parents",
            "region_population_relative": 0.05,
            "ext_source_1": 0.3,
            "ext_source_2": None,
            "ext_source_3": 0.2
        }
    ]
    
    print(f"Processing {len(applications)} loan applications...")
    
    start_time = time.time()
    result = client.predict_batch(applications)
    response_time = (time.time() - start_time) * 1000
    
    if "error" not in result:
        print(f"\n✅ Batch Prediction Results:")
        print(f"   Batch ID: {result['batch_id']}")
        print(f"   Total Processed: {result['total_processed']}")
        print(f"   Processing Time: {result['processing_time_ms']:.2f}ms")
        print(f"   Response Time: {response_time:.2f}ms")
        print(f"   Avg per Application: {response_time/len(applications):.2f}ms")
        
        print(f"\n📊 Individual Results:")
        for i, prediction in enumerate(result['predictions']):
            app_data = applications[i]
            print(f"\n   Application {i+1}:")
            print(f"     Credit: ${app_data['amt_credit']:,.0f}")
            print(f"     Income: ${app_data['amt_income_total']:,.0f}")
            print(f"     Prediction: {prediction['prediction']:.4f} ({prediction['risk_level']})")
            print(f"     Confidence: {prediction['confidence']:.4f}")
        
        # Summary statistics
        predictions = [p['prediction'] for p in result['predictions']]
        risk_levels = [p['risk_level'] for p in result['predictions']]
        
        print(f"\n📈 Batch Summary:")
        print(f"   Average Default Probability: {np.mean(predictions):.4f}")
        print(f"   Risk Distribution:")
        for risk in ["LOW", "MEDIUM", "HIGH"]:
            count = risk_levels.count(risk)
            percentage = (count / len(risk_levels)) * 100
            print(f"     {risk}: {count} ({percentage:.1f}%)")
            
    else:
        print(f"❌ Error: {result['error']}")
    
    print("\n" + "=" * 50)

def example_4_csv_file_processing():
    """Example 4: Process loan applications from CSV file"""
    print("📄 Example 4: CSV File Processing")
    print("=" * 50)
    
    # Create sample CSV data
    sample_data = {
        'amt_credit': [300000, 500000, 750000, 200000],
        'amt_annuity': [15000, 25000, 35000, 10000],
        'amt_income_total': [80000, 120000, 180000, 60000],
        'code_gender': ['F', 'M', 'M', 'F'],
        'days_birth': [-10000, -12000, -15000, -8000],
        'days_employed': [-1500, -3000, -5000, -800],
        'name_contract_type': ['Cash loans'] * 4,
        'name_income_type': ['Working', 'Working', 'Commercial associate', 'Working'],
        'name_education_type': ['Higher education', 'Secondary / secondary special', 'Higher education', 'Lower secondary'],
        'name_family_status': ['Single / not married', 'Married', 'Married', 'Single / not married'],
        'name_housing_type': ['Rented apartment', 'House / apartment', 'House / apartment', 'With parents'],
        'region_population_relative': [0.03, 0.02, 0.015, 0.05],
        'ext_source_1': [0.5, 0.7, 0.8, 0.3],
        'ext_source_2': [0.4, 0.6, 0.9, 0.2],
        'ext_source_3': [0.3, 0.8, 0.7, 0.1]
    }
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(sample_data)
    csv_file = "sample_loan_applications.csv"
    df.to_csv(csv_file, index=False)
    print(f"Created sample file: {csv_file}")
    
    # Process CSV file
    client = LoanPredictionClient()
    
    print(f"\nProcessing {len(df)} applications from CSV...")
    
    # Convert DataFrame to list of dictionaries
    applications = df.to_dict('records')
    
    # Make batch prediction
    result = client.predict_batch(applications)
    
    if "error" not in result:
        # Add predictions to DataFrame
        predictions = result['predictions']
        df['application_id'] = [p['application_id'] for p in predictions]
        df['default_probability'] = [p['prediction'] for p in predictions]
        df['risk_level'] = [p['risk_level'] for p in predictions]
        df['confidence'] = [p['confidence'] for p in predictions]
        
        # Save results
        output_file = "loan_predictions_output.csv"
        df.to_csv(output_file, index=False)
        
        print(f"✅ Results saved to: {output_file}")
        print(f"\nSample Results:")
        print(df[['amt_credit', 'amt_income_total', 'default_probability', 'risk_level']].head())
        
        # Generate summary report
        print(f"\n📊 Summary Report:")
        print(f"   Total Applications: {len(df)}")
        print(f"   Average Default Probability: {df['default_probability'].mean():.4f}")
        
        risk_summary = df['risk_level'].value_counts()
        for risk, count in risk_summary.items():
            percentage = (count / len(df)) * 100
            print(f"   {risk} Risk: {count} ({percentage:.1f}%)")
            
    else:
        print(f"❌ Error: {result['error']}")
    
    print("\n" + "=" * 50)

def example_5_error_handling():
    """Example 5: Error handling and edge cases"""
    print("⚠️ Example 5: Error Handling")
    print("=" * 50)
    
    client = LoanPredictionClient()
    
    print("1. Testing invalid data types...")
    invalid_application = {
        "amt_credit": "invalid_amount",  # Should be float
        "amt_annuity": 25000.0,
        "amt_income_total": 150000.0,
        "code_gender": "X",  # Invalid gender
        "days_birth": 1000,  # Should be negative
        "name_contract_type": "Invalid type"
    }
    
    result = client.predict_single(invalid_application)
    if "error" in result:
        print(f"   ✅ Correctly caught validation error: {result['error'][:100]}...")
    
    print("\n2. Testing missing required fields...")
    incomplete_application = {
        "amt_credit": 450000.0,
        # Missing required fields
    }
    
    result = client.predict_single(incomplete_application)
    if "error" in result:
        print(f"   ✅ Correctly caught missing fields error: {result['error'][:100]}...")
    
    print("\n3. Testing connection error...")
    offline_client = LoanPredictionClient("http://invalid-url:9999")
    result = offline_client.check_api_health()
    if result["status"] == "error":
        print(f"   ✅ Correctly handled connection error: {result['message'][:100]}...")
    
    print("\n4. Testing large batch (should succeed)...")
    # Create a large but valid batch
    large_batch = []
    for i in range(50):  # Create 50 applications
        app = {
            "amt_credit": 300000.0 + (i * 10000),
            "amt_annuity": 15000.0 + (i * 500),
            "amt_income_total": 80000.0 + (i * 2000),
            "code_gender": "M" if i % 2 == 0 else "F",
            "days_birth": -10000 - (i * 100),
            "days_employed": -1500 - (i * 50),
            "name_contract_type": "Cash loans",
            "name_income_type": "Working",
            "name_education_type": "Higher education",
            "name_family_status": "Single / not married",
            "name_housing_type": "House / apartment",
            "region_population_relative": 0.02,
            "ext_source_1": 0.5 + (i * 0.01),
            "ext_source_2": 0.6,
            "ext_source_3": 0.7
        }
        large_batch.append(app)
    
    result = client.predict_batch(large_batch)
    if "error" not in result:
        print(f"   ✅ Successfully processed large batch of {len(large_batch)} applications")
        print(f"      Processing time: {result['processing_time_ms']:.2f}ms")
    else:
        print(f"   ❌ Large batch failed: {result['error']}")
    
    print("\n" + "=" * 50)

async def example_6_async_requests():
    """Example 6: Asynchronous requests for high throughput"""
    print("🚀 Example 6: Asynchronous Batch Processing")
    print("=" * 50)
    
    api_url = "http://localhost:8000"
    
    # Create sample application
    sample_app = {
        "amt_credit": 450000.0,
        "amt_annuity": 25000.0,
        "amt_income_total": 150000.0,
        "code_gender": "M",
        "days_birth": -12000,
        "days_employed": -2000,
        "name_contract_type": "Cash loans",
        "name_income_type": "Working",
        "name_education_type": "Higher education",
        "name_family_status": "Married",
        "name_housing_type": "House / apartment",
        "region_population_relative": 0.02,
        "ext_source_1": 0.7,
        "ext_source_2": 0.6,
        "ext_source_3": 0.8
    }
    
    async def make_prediction(session, app_data, app_id):
        """Make a single async prediction"""
        try:
            async with session.post(f"{api_url}/predict", json=app_data) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"app_id": app_id, "success": True, "result": result}
                else:
                    return {"app_id": app_id, "success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"app_id": app_id, "success": False, "error": str(e)}
    
    # Create multiple variations of the sample application
    applications = []
    for i in range(20):
        app = sample_app.copy()
        app["amt_credit"] = sample_app["amt_credit"] + (i * 50000)
        app["amt_income_total"] = sample_app["amt_income_total"] + (i * 10000)
        applications.append(app)
    
    print(f"Making {len(applications)} concurrent requests...")
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [make_prediction(session, app, i) for i, app in enumerate(applications)]
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\n✅ Async Processing Results:")
    print(f"   Total Time: {total_time:.2f}ms")
    print(f"   Avg per Request: {total_time/len(applications):.2f}ms")
    print(f"   Successful: {len(successful)}")
    print(f"   Failed: {len(failed)}")
    
    if successful:
        predictions = [r["result"]["prediction"] for r in successful]
        print(f"   Avg Default Probability: {np.mean(predictions):.4f}")
        print(f"   Min/Max Probability: {np.min(predictions):.4f} / {np.max(predictions):.4f}")
    
    if failed:
        print(f"\n❌ Failed Requests:")
        for failure in failed[:3]:  # Show first 3 failures
            print(f"   App {failure['app_id']}: {failure['error']}")
    
    print("\n" + "=" * 50)

def main():
    """Run all examples"""
    print("🎯 Loan Default Prediction API - Client Examples")
    print("=" * 60)
    print("This script demonstrates how to use the API for loan default predictions")
    print("Make sure the API is running at http://localhost:8000")
    print("=" * 60)
    
    # Run synchronous examples
    try:
        example_1_basic_usage()
        time.sleep(1)
        
        example_2_single_prediction()
        time.sleep(1)
        
        example_3_batch_predictions()
        time.sleep(1)
        
        example_4_csv_file_processing()
        time.sleep(1)
        
        example_5_error_handling()
        time.sleep(1)
        
        # Run async example
        print("\nRunning async example...")
        asyncio.run(example_6_async_requests())
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        print("Make sure the API is running and accessible.")
    
    print(f"\n🎉 Examples completed!")
    print("Check the generated files:")
    print("  - sample_loan_applications.csv")
    print("  - loan_predictions_output.csv")

if __name__ == "__main__":
    main()