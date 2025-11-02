"""
Turbo Model API Client

Test the ultra-fast models via API with performance benchmarking
"""

import requests
import time
import json
import numpy as np
from typing import Dict, Any, List

API_BASE_URL = "http://localhost:8000"

class TurboAPIClient:
    """Client for testing turbo models via API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        
    def test_turbo_model_speed(self, model_name: str, num_requests: int = 100) -> Dict[str, Any]:
        """Test turbo model prediction speed"""
        print(f"⚡ Speed testing {model_name} with {num_requests} requests...")
        
        # Sample test features
        test_features = {
            "income_ratio": 0.6,
            "credit_score": 0.8,
            "debt_to_income": 0.3,
            "loan_amount_ratio": 0.5
        }
        
        request_data = {
            "model_name": model_name,
            "features": test_features,
            "simulation_method": "statistical"
        }
        
        response_times = []
        predictions = []
        
        # Warm-up request
        try:
            response = requests.post(
                f"{self.base_url}/models/advanced/predict",
                json=request_data
            )
            if response.status_code == 200:
                print("✅ Warm-up successful")
            else:
                print(f"❌ Warm-up failed: {response.status_code}")
                return {"error": "Warm-up failed"}
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return {"error": str(e)}
        
        # Speed test
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                request_start = time.time()
                response = requests.post(
                    f"{self.base_url}/models/advanced/predict",
                    json=request_data
                )
                request_end = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    response_times.append(request_end - request_start)
                    predictions.append(result.get('prediction', 0))
                    
                    if (i + 1) % 20 == 0:
                        avg_time = np.mean(response_times[-20:])
                        print(f"   Progress: {i+1}/{num_requests}, Avg: {avg_time:.3f}s")
                else:
                    print(f"   Request {i+1} failed: {response.status_code}")
                    
            except Exception as e:
                print(f"   Request {i+1} error: {e}")
        
        total_time = time.time() - start_time
        
        if response_times:
            results = {
                'model_name': model_name,
                'total_requests': len(response_times),
                'successful_requests': len(response_times),
                'total_time': total_time,
                'avg_response_time': np.mean(response_times),
                'min_response_time': np.min(response_times),
                'max_response_time': np.max(response_times),
                'requests_per_second': len(response_times) / total_time,
                'prediction_consistency': len(set(predictions)) == 1  # All same prediction
            }
            
            print(f"✅ Speed test completed!")
            print(f"   📊 {results['successful_requests']}/{num_requests} successful")
            print(f"   ⚡ Avg response: {results['avg_response_time']:.3f}s")
            print(f"   🚀 Requests/sec: {results['requests_per_second']:.1f}")
            
            return results
        else:
            return {"error": "No successful requests"}
    
    def compare_turbo_models(self) -> Dict[str, Any]:
        """Compare all available turbo models"""
        print("🏁 TURBO MODEL SPEED COMPARISON")
        print("=" * 50)
        
        # Get available models
        try:
            response = requests.get(f"{self.base_url}/models")
            if response.status_code != 200:
                return {"error": "Failed to get model list"}
                
            models_data = response.json()
            available_models = list(models_data.get('available_models', {}).keys())
            
            # Filter for turbo models
            turbo_models = [m for m in available_models if 'turbo' in m.lower()]
            
            if not turbo_models:
                # Test with any available models
                turbo_models = [m for m in available_models if 'advanced' in m or 'demo' in m]
            
            print(f"📋 Testing models: {turbo_models}")
            
        except Exception as e:
            print(f"❌ Error getting models: {e}")
            # Use default model names
            turbo_models = ['demo_local_model_2']
        
        results = {}
        
        for model_name in turbo_models:
            print(f"\\n🧪 Testing {model_name}...")
            
            result = self.test_turbo_model_speed(model_name, num_requests=50)
            results[model_name] = result
        
        # Performance summary
        print(f"\\n{'='*60}")
        print("🏆 TURBO MODEL PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if valid_results:
            # Sort by speed
            sorted_by_speed = sorted(
                valid_results.items(), 
                key=lambda x: x[1]['avg_response_time']
            )
            
            print("⚡ SPEED RANKING (fastest first):")
            for i, (model, result) in enumerate(sorted_by_speed, 1):
                rps = result['requests_per_second']
                avg_time = result['avg_response_time']
                print(f"   {i}. {model:25}: {avg_time:.3f}s avg, {rps:.1f} req/s")
            
            fastest = sorted_by_speed[0]
            print(f"\\n🥇 FASTEST MODEL: {fastest[0]}")
            print(f"   ⚡ Response time: {fastest[1]['avg_response_time']:.3f}s")
            print(f"   🚀 Throughput: {fastest[1]['requests_per_second']:.1f} requests/second")
            
        else:
            print("❌ No valid results to compare")
        
        return results
    
    def test_missing_feature_performance(self, model_name: str) -> Dict[str, Any]:
        """Test performance with different numbers of missing features"""
        print(f"🔧 Testing missing feature simulation performance for {model_name}")
        
        # Test scenarios with different numbers of features
        scenarios = [
            {"name": "complete", "features": {
                "income_ratio": 0.6, "credit_score": 0.8, "debt_to_income": 0.3,
                "loan_amount_ratio": 0.5, "employment_years": 0.7, "property_value": 0.9
            }},
            {"name": "partial", "features": {
                "income_ratio": 0.6, "credit_score": 0.8, "debt_to_income": 0.3
            }},
            {"name": "minimal", "features": {
                "income_ratio": 0.6, "credit_score": 0.8
            }},
            {"name": "single", "features": {
                "credit_score": 0.8
            }}
        ]
        
        results = {}
        
        for scenario in scenarios:
            print(f"\\n   Testing {scenario['name']} features...")
            
            request_data = {
                "model_name": model_name,
                "features": scenario['features'],
                "simulation_method": "statistical"
            }
            
            times = []
            for _ in range(10):  # 10 requests per scenario
                start = time.time()
                try:
                    response = requests.post(
                        f"{self.base_url}/models/advanced/predict",
                        json=request_data
                    )
                    end = time.time()
                    
                    if response.status_code == 200:
                        times.append(end - start)
                        result_data = response.json()
                    else:
                        print(f"     Request failed: {response.status_code}")
                        
                except Exception as e:
                    print(f"     Request error: {e}")
            
            if times:
                results[scenario['name']] = {
                    'avg_time': np.mean(times),
                    'num_features': len(scenario['features']),
                    'features_simulated': result_data.get('features_simulated', 0) if 'result_data' in locals() else 0
                }
                
                print(f"     ✅ Avg time: {results[scenario['name']]['avg_time']:.3f}s")
        
        return results

def main():
    """Run turbo model performance tests"""
    print("🚀 TURBO MODEL API PERFORMANCE TESTING")
    print("=" * 60)
    
    client = TurboAPIClient()
    
    # Test API connectivity
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code != 200:
            print("❌ API not accessible. Make sure Docker containers are running.")
            return
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Run: docker-compose -f docker/docker-compose.dev.yml up -d")
        return
    
    # Run comparison tests
    comparison_results = client.compare_turbo_models()
    
    # Test missing feature performance if we have a working model
    valid_models = [k for k, v in comparison_results.items() if 'error' not in v]
    if valid_models:
        fastest_model = min(
            valid_models, 
            key=lambda x: comparison_results[x]['avg_response_time']
        )
        
        print(f"\\n🔧 Testing missing feature performance with fastest model: {fastest_model}")
        feature_results = client.test_missing_feature_performance(fastest_model)
        
        if feature_results:
            print(f"\\n📊 MISSING FEATURE SIMULATION PERFORMANCE:")
            for scenario, result in feature_results.items():
                print(f"   {scenario:10}: {result['avg_time']:.3f}s, {result['num_features']} → {result['features_simulated']} features")

if __name__ == "__main__":
    main()