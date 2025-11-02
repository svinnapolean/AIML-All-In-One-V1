#!/usr/bin/env python3
"""
API Test Client
Tests the deployed loan default prediction API endpoints
"""

import requests
import json
from datetime import datetime
import time

API_BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health endpoint"""
    print("🏥 Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   Uptime: {data['uptime_seconds']:.1f} seconds")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_models_endpoint():
    """Test the models listing endpoint"""
    print("\n📋 Testing Models Endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models found: {len(data)}")
            for model in data:
                print(f"   - {model}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def test_prediction_endpoint():
    """Test a prediction endpoint with sample data"""
    print("\n🎯 Testing Prediction Endpoint...")
    
    # Sample loan application data
    sample_request = {
        "amt_credit": 200000.0,
        "amt_annuity": 12000.0,
        "amt_income_total": 80000.0,
        "amt_goods_price": 180000.0,
        "code_gender": "M",
        "days_birth": -10000,  # ~27 years old
        "days_employed": -2000,  # ~5.5 years employed
        "name_contract_type": "Cash loans",
        "name_income_type": "Working",
        "name_education_type": "Higher education",
        "name_family_status": "Married",
        "name_housing_type": "House / apartment",
        "region_population_relative": 0.05,
        "ext_source_1": 0.7,
        "ext_source_2": 0.8,
        "ext_source_3": 0.6
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict", 
            json=sample_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Application ID: {data['application_id']}")
            print(f"   Default Probability: {data['prediction']:.4f}")
            print(f"   Risk Level: {data['risk_level']}")
            print(f"   Model Used: {data['model_used']}")
            print(f"   Confidence: {data['confidence']:.4f}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n📦 Testing Batch Prediction...")
    
    # Create multiple sample applications
    applications = []
    for i in range(3):
        app = {
            "amt_credit": 150000.0 + (i * 50000),
            "amt_annuity": 10000.0 + (i * 2000),
            "amt_income_total": 60000.0 + (i * 20000),
            "amt_goods_price": 140000.0 + (i * 40000),
            "code_gender": "F" if i % 2 == 0 else "M",
            "days_birth": -8000 - (i * 1000),
            "days_employed": -1500 - (i * 500),
            "name_contract_type": "Cash loans",
            "name_income_type": "Working",
            "name_education_type": "Secondary / secondary special",
            "name_family_status": "Single / not married",
            "name_housing_type": "House / apartment",
            "region_population_relative": 0.03 + (i * 0.01),
            "ext_source_1": 0.5 + (i * 0.1),
            "ext_source_2": 0.6 + (i * 0.1),
            "ext_source_3": 0.4 + (i * 0.15)
        }
        applications.append(app)
    
    batch_request = {"applications": applications}
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict-batch",
            json=batch_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful!")
            print(f"   Batch ID: {data['batch_id']}")
            print(f"   Total Processed: {data['total_processed']}")
            print(f"   Processing Time: {data['processing_time_ms']:.1f}ms")
            
            for i, pred in enumerate(data['predictions']):
                print(f"   App {i+1}: Risk={pred['risk_level']}, Prob={pred['prediction']:.4f}")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False

def test_api_documentation():
    """Test that API documentation is accessible"""
    print("\n📖 Testing API Documentation...")
    try:
        response = requests.get(f"{API_BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ API documentation accessible")
            return True
        else:
            print(f"❌ Documentation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Documentation error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Starting Loan Default Prediction API Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Models Listing", test_models_endpoint),
        ("Single Prediction", test_prediction_endpoint),
        ("Batch Prediction", test_batch_prediction),
        ("API Documentation", test_api_documentation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        start_time = time.time()
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
        
        duration = time.time() - start_time
        print(f"   Duration: {duration:.2f}s")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print("-" * 30)
    print(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! API is fully functional.")
    elif passed > 0:
        print("⚠️ Some tests passed. API is partially functional.")
    else:
        print("❌ All tests failed. API needs debugging.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)