"""
Advanced API Client Examples for Autoencoder and LightGBM Models

This script demonstrates how to:
1. Train advanced models (Autoencoder + LightGBM)
2. Test models with complete feature sets
3. Test models with missing features (using feature simulation)
4. Compare different simulation methods
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# API Configuration
API_BASE_URL = "http://localhost:8000"
MODELS_ENDPOINT = f"{API_BASE_URL}/models"

class AdvancedAPIClient:
    """Client for testing advanced ML models via API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.models_url = f"{base_url}/models"
        
    def train_advanced_models(self) -> Dict[str, Any]:
        """Start training advanced models in background"""
        print("🚀 Starting advanced model training...")
        
        try:
            response = requests.post(f"{self.models_url}/advanced/train")
            response.raise_for_status()
            result = response.json()
            
            print("✅ Training started successfully!")
            print(f"📋 Models: {result.get('models', [])}")
            print(f"⏰ Estimated time: {result.get('estimated_time', 'Unknown')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error starting training: {e}")
            return {"error": str(e)}
    
    def wait_for_training_completion(self, check_interval: int = 30) -> bool:
        """Wait for training to complete by checking for new models"""
        print(f"⏳ Waiting for training completion (checking every {check_interval}s)...")
        
        initial_models = self.list_models()
        advanced_models = [m for m in initial_models if m.startswith('advanced_')]
        
        while len(advanced_models) < 2:  # Waiting for 2 advanced models
            time.sleep(check_interval)
            current_models = self.list_models()
            advanced_models = [m for m in current_models if m.startswith('advanced_')]
            print(f"📊 Found {len(advanced_models)} advanced models...")
        
        print("✅ Training completed!")
        return True
    
    def list_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(self.models_url)
            response.raise_for_status()
            models = response.json()
            return [model.get('name', '') for model in models]
        except:
            return []
    
    def test_complete_features(self, model_name: str) -> Dict[str, Any]:
        """Test model with complete feature set"""
        print(f"\n🧪 Testing {model_name} with complete features...")
        
        # Sample complete loan application data
        complete_features = {
            "Loan_ID": "LP001002",
            "Gender": "Male",
            "Married": "No",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5849,
            "CoapplicantIncome": 0,
            "LoanAmount": 128,
            "Loan_Amount_Term": 360,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        }
        
        request_data = {
            "model_name": model_name,
            "features": complete_features,
            "simulation_method": "statistical"
        }
        
        try:
            response = requests.post(
                f"{self.models_url}/advanced/predict",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Prediction successful!")
            print(f"   🎯 Prediction: {result['prediction']}")
            print(f"   📊 Probability: {result.get('prediction_proba', 'N/A'):.4f}")
            print(f"   ⚡ Processing time: {result['processing_time']:.3f}s")
            print(f"   📈 Features used: {result['features_used']}")
            print(f"   🔧 Features simulated: {result['features_simulated']}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"   Response: {e.response.text}")
            return {"error": str(e)}
    
    def test_missing_features(self, model_name: str, simulation_method: str = "statistical") -> Dict[str, Any]:
        """Test model with missing features"""
        print(f"\n🔧 Testing {model_name} with missing features (method: {simulation_method})...")
        
        # Incomplete loan application data (missing several features)
        incomplete_features = {
            "Gender": "Female",
            "Married": "Yes",
            "Education": "Graduate",
            "ApplicantIncome": 4583,
            "LoanAmount": 128,
            "Credit_History": 1.0
            # Missing: Dependents, Self_Employed, CoapplicantIncome, Loan_Amount_Term, Property_Area
        }
        
        request_data = {
            "model_name": model_name,
            "features": incomplete_features,
            "simulation_method": simulation_method
        }
        
        try:
            response = requests.post(
                f"{self.models_url}/advanced/predict",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Prediction with missing features successful!")
            print(f"   🎯 Prediction: {result['prediction']}")
            print(f"   📊 Probability: {result.get('prediction_proba', 'N/A'):.4f}")
            print(f"   ⚡ Processing time: {result['processing_time']:.3f}s")
            print(f"   📈 Features used: {result['features_used']}")
            print(f"   🔧 Features simulated: {result['features_simulated']}")
            
            # Show simulation details
            sim_report = result.get('simulation_report', {})
            if sim_report.get('missing_features_added'):
                print(f"   📋 Missing features simulated: {sim_report['missing_features_added']}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"   Response: {e.response.text}")
            return {"error": str(e)}
    
    def compare_simulation_methods(self, model_name: str) -> Dict[str, Dict[str, Any]]:
        """Compare different simulation methods"""
        print(f"\n📊 Comparing simulation methods for {model_name}...")
        
        methods = ["statistical", "median", "zero", "random"]
        results = {}
        
        # Same incomplete feature set for all tests
        incomplete_features = {
            "Gender": "Male",
            "Education": "Not Graduate",
            "ApplicantIncome": 3000,
            "LoanAmount": 100
            # Missing many features
        }
        
        for method in methods:
            print(f"\n🔍 Testing method: {method}")
            
            request_data = {
                "model_name": model_name,
                "features": incomplete_features,
                "simulation_method": method
            }
            
            try:
                response = requests.post(
                    f"{self.models_url}/advanced/predict",
                    json=request_data
                )
                response.raise_for_status()
                result = response.json()
                
                results[method] = result
                print(f"   ✅ {method}: Prediction={result['prediction']}, Prob={result.get('prediction_proba', 0):.4f}")
                
            except Exception as e:
                results[method] = {"error": str(e)}
                print(f"   ❌ {method}: Error - {e}")
        
        return results
    
    def stress_test_models(self, model_names: List[str], num_tests: int = 10) -> Dict[str, Any]:
        """Perform stress testing with multiple random requests"""
        print(f"\\n⚡ Stress testing models with {num_tests} requests each...")
        
        results = {}
        
        for model_name in model_names:
            print(f"\\n🔥 Stress testing {model_name}...")
            model_results = []
            
            for i in range(num_tests):
                # Generate random test data
                random_features = {
                    "Gender": np.random.choice(["Male", "Female"]),
                    "Married": np.random.choice(["Yes", "No"]),
                    "Education": np.random.choice(["Graduate", "Not Graduate"]),
                    "ApplicantIncome": np.random.randint(1000, 10000),
                    "LoanAmount": np.random.randint(50, 500),
                    "Credit_History": np.random.choice([0.0, 1.0])
                }
                
                request_data = {
                    "model_name": model_name,
                    "features": random_features,
                    "simulation_method": "statistical"
                }
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.models_url}/advanced/predict",
                        json=request_data
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    result['request_time'] = time.time() - start_time
                    model_results.append(result)
                    
                    if (i + 1) % 5 == 0:
                        print(f"   ✅ Completed {i + 1}/{num_tests} tests")
                    
                except Exception as e:
                    model_results.append({"error": str(e)})
                    print(f"   ❌ Test {i + 1} failed: {e}")
            
            # Calculate statistics
            successful_tests = [r for r in model_results if 'error' not in r]
            if successful_tests:
                avg_response_time = np.mean([r['processing_time'] for r in successful_tests])
                avg_request_time = np.mean([r['request_time'] for r in successful_tests])
                success_rate = len(successful_tests) / len(model_results) * 100
                
                results[model_name] = {
                    'success_rate': success_rate,
                    'avg_processing_time': avg_response_time,
                    'avg_request_time': avg_request_time,
                    'total_tests': len(model_results),
                    'successful_tests': len(successful_tests)
                }
                
                print(f"   📊 Results: {success_rate:.1f}% success, {avg_response_time:.3f}s avg processing")
            else:
                results[model_name] = {'error': 'All tests failed'}
                print(f"   ❌ All tests failed for {model_name}")
        
        return results


def main():
    """Main function to demonstrate advanced API testing"""
    print("🚀 Advanced ML Models API Testing")
    print("=" * 50)
    
    client = AdvancedAPIClient()
    
    # Step 1: Start training
    print("\\n📚 Step 1: Training Advanced Models")
    training_result = client.train_advanced_models()
    
    if "error" in training_result:
        print("❌ Training failed to start. Exiting.")
        return
    
    # Step 2: Wait for training completion
    print("\\n⏳ Step 2: Waiting for Training Completion")
    if not client.wait_for_training_completion():
        print("❌ Training did not complete. Exiting.")
        return
    
    # Step 3: Get available advanced models
    print("\\n📋 Step 3: Checking Available Models")
    all_models = client.list_models()
    advanced_models = [m for m in all_models if m.startswith('advanced_')]
    
    if not advanced_models:
        print("❌ No advanced models found. Training may have failed.")
        return
    
    print(f"✅ Found {len(advanced_models)} advanced models:")
    for model in advanced_models:
        print(f"   📊 {model}")
    
    # Step 4: Test models with complete features
    print("\\n🧪 Step 4: Testing with Complete Features")
    for model in advanced_models:
        client.test_complete_features(model)
    
    # Step 5: Test models with missing features
    print("\\n🔧 Step 5: Testing with Missing Features")
    for model in advanced_models:
        client.test_missing_features(model, "statistical")
    
    # Step 6: Compare simulation methods
    print("\\n📊 Step 6: Comparing Simulation Methods")
    if advanced_models:
        comparison_results = client.compare_simulation_methods(advanced_models[0])
        
        print("\\n📋 Simulation Method Comparison Summary:")
        for method, result in comparison_results.items():
            if 'error' not in result:
                prob = result.get('prediction_proba', 0)
                pred = result.get('prediction', 'Unknown')
                print(f"   {method:12}: Prediction={pred}, Probability={prob:.4f}")
            else:
                print(f"   {method:12}: ERROR - {result['error']}")
    
    # Step 7: Stress testing
    print("\\n⚡ Step 7: Stress Testing")
    if len(advanced_models) >= 1:
        stress_results = client.stress_test_models(advanced_models[:2], num_tests=5)
        
        print("\\n📊 Stress Test Results:")
        for model, results in stress_results.items():
            if 'error' not in results:
                print(f"   {model}:")
                print(f"     Success Rate: {results['success_rate']:.1f}%")
                print(f"     Avg Processing: {results['avg_processing_time']:.3f}s")
                print(f"     Avg Request: {results['avg_request_time']:.3f}s")
            else:
                print(f"   {model}: FAILED - {results['error']}")
    
    print("\\n🎉 Advanced API Testing Completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()