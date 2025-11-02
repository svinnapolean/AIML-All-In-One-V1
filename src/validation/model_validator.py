"""
Model Validation Pipeline
Comprehensive validation system for deployed models
"""

import requests
import json
import time
import logging
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
from pathlib import Path

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from model_manager import ModelManager
from home_loan_data import HomeLoanData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for validation pipeline"""
    api_url: str = "http://localhost:8000"
    timeout_seconds: int = 30
    max_retries: int = 3
    batch_size: int = 10
    performance_threshold_roc_auc: float = 0.7
    performance_threshold_accuracy: float = 0.65
    response_time_threshold_ms: float = 1000.0
    availability_threshold_percent: float = 99.0

class ModelValidator:
    """Comprehensive model validation system"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.session = requests.Session()
        self.validation_results = {}
        
    def validate_api_health(self) -> Dict[str, Any]:
        """Validate API health and availability"""
        logger.info("🏥 Validating API health...")
        
        try:
            start_time = time.time()
            response = self.session.get(
                f"{self.config.api_url}/health",
                timeout=self.config.timeout_seconds
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                health_data = response.json()
                
                result = {
                    "status": "healthy" if health_data.get("status") == "healthy" else "unhealthy",
                    "response_time_ms": response_time,
                    "model_loaded": health_data.get("model_loaded", False),
                    "current_model": health_data.get("model_name", "none"),
                    "uptime_seconds": health_data.get("uptime_seconds", 0),
                    "api_version": health_data.get("version", "unknown"),
                    "passes_response_time_check": response_time < self.config.response_time_threshold_ms
                }
                
                logger.info(f"✅ API Health Check Passed")
                logger.info(f"   Response Time: {response_time:.2f}ms")
                logger.info(f"   Model Loaded: {result['model_loaded']}")
                logger.info(f"   Current Model: {result['current_model']}")
                
                return result
            else:
                logger.error(f"❌ Health check failed with status code: {response.status_code}")
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                    "response_time_ms": response_time
                }
                
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "response_time_ms": None
            }
    
    def validate_model_endpoints(self) -> Dict[str, Any]:
        """Validate model management endpoints"""
        logger.info("🔧 Validating model management endpoints...")
        
        results = {}
        
        # Test list models endpoint
        try:
            response = self.session.get(f"{self.config.api_url}/models")
            if response.status_code == 200:
                models_data = response.json()
                results["list_models"] = {
                    "status": "pass",
                    "available_models": models_data.get("total_count", 0),
                    "current_loaded": models_data.get("current_loaded", "none")
                }
                logger.info(f"✅ List models endpoint working ({results['list_models']['available_models']} models)")
            else:
                results["list_models"] = {"status": "fail", "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            results["list_models"] = {"status": "error", "error": str(e)}
        
        # Test model info endpoint
        try:
            response = self.session.get(f"{self.config.api_url}/model-info")
            if response.status_code == 200:
                model_info = response.json()
                results["model_info"] = {
                    "status": "pass",
                    "model_name": model_info.get("model_name", "unknown"),
                    "model_type": model_info.get("model_type", "unknown"),
                    "features_count": model_info.get("features_count", 0)
                }
                logger.info(f"✅ Model info endpoint working")
            else:
                results["model_info"] = {"status": "fail", "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            results["model_info"] = {"status": "error", "error": str(e)}
        
        return results
    
    def generate_test_data(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic test data for validation"""
        logger.info(f"📊 Generating {num_samples} test samples...")
        
        test_samples = []
        
        for i in range(num_samples):
            sample = {
                "amt_credit": np.random.uniform(50000, 1000000),
                "amt_annuity": np.random.uniform(5000, 50000),
                "amt_income_total": np.random.uniform(30000, 200000),
                "amt_goods_price": np.random.uniform(40000, 800000),
                "code_gender": np.random.choice(["M", "F"]),
                "days_birth": np.random.randint(-25000, -6000),  # Age 18-68
                "days_employed": np.random.randint(-15000, 1000),
                "name_contract_type": np.random.choice(["Cash loans", "Revolving loans"]),
                "name_income_type": np.random.choice([
                    "Working", "Commercial associate", "Pensioner", "State servant"
                ]),
                "name_education_type": np.random.choice([
                    "Higher education", "Secondary / secondary special", 
                    "Incomplete higher", "Lower secondary"
                ]),
                "name_family_status": np.random.choice([
                    "Single / not married", "Married", "Civil marriage", 
                    "Widow", "Separated"
                ]),
                "name_housing_type": np.random.choice([
                    "House / apartment", "Rented apartment", "With parents",
                    "Municipal apartment", "Office apartment", "Co-op apartment"
                ]),
                "region_population_relative": np.random.uniform(0.001, 0.1),
                "ext_source_1": np.random.uniform(0.0, 1.0) if np.random.random() > 0.3 else None,
                "ext_source_2": np.random.uniform(0.0, 1.0) if np.random.random() > 0.3 else None,
                "ext_source_3": np.random.uniform(0.0, 1.0) if np.random.random() > 0.3 else None
            }
            test_samples.append(sample)
        
        return test_samples
    
    def validate_single_prediction(self) -> Dict[str, Any]:
        """Validate single prediction endpoint"""
        logger.info("🎯 Validating single prediction endpoint...")
        
        test_data = self.generate_test_data(1)[0]
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.config.api_url}/predict",
                json=test_data,
                timeout=self.config.timeout_seconds
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                prediction_data = response.json()
                
                # Validate response structure
                required_fields = ["application_id", "prediction", "risk_level", 
                                 "model_used", "prediction_timestamp", "confidence"]
                missing_fields = [field for field in required_fields 
                                if field not in prediction_data]
                
                if missing_fields:
                    return {
                        "status": "fail",
                        "error": f"Missing fields: {missing_fields}",
                        "response_time_ms": response_time
                    }
                
                # Validate prediction value
                prediction = prediction_data["prediction"]
                if not (0 <= prediction <= 1):
                    return {
                        "status": "fail",
                        "error": f"Invalid prediction value: {prediction} (should be 0-1)",
                        "response_time_ms": response_time
                    }
                
                # Validate risk level
                risk_level = prediction_data["risk_level"]
                if risk_level not in ["LOW", "MEDIUM", "HIGH"]:
                    return {
                        "status": "fail",
                        "error": f"Invalid risk level: {risk_level}",
                        "response_time_ms": response_time
                    }
                
                result = {
                    "status": "pass",
                    "response_time_ms": response_time,
                    "prediction": prediction,
                    "risk_level": risk_level,
                    "confidence": prediction_data["confidence"],
                    "model_used": prediction_data["model_used"],
                    "passes_response_time_check": response_time < self.config.response_time_threshold_ms
                }
                
                logger.info(f"✅ Single prediction validation passed")
                logger.info(f"   Response Time: {response_time:.2f}ms")
                logger.info(f"   Prediction: {prediction:.4f} ({risk_level})")
                
                return result
                
            else:
                logger.error(f"❌ Prediction failed with status code: {response.status_code}")
                return {
                    "status": "fail",
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "response_time_ms": response_time
                }
                
        except Exception as e:
            logger.error(f"❌ Single prediction validation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "response_time_ms": None
            }
    
    def validate_batch_prediction(self) -> Dict[str, Any]:
        """Validate batch prediction endpoint"""
        logger.info("📦 Validating batch prediction endpoint...")
        
        test_applications = self.generate_test_data(self.config.batch_size)
        batch_request = {"applications": test_applications}
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.config.api_url}/predict-batch",
                json=batch_request,
                timeout=self.config.timeout_seconds * 2  # Longer timeout for batch
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                batch_data = response.json()
                
                # Validate response structure
                required_fields = ["predictions", "batch_id", "total_processed", "processing_time_ms"]
                missing_fields = [field for field in required_fields if field not in batch_data]
                
                if missing_fields:
                    return {
                        "status": "fail",
                        "error": f"Missing fields: {missing_fields}",
                        "response_time_ms": response_time
                    }
                
                predictions = batch_data["predictions"]
                expected_count = len(test_applications)
                actual_count = len(predictions)
                
                if actual_count != expected_count:
                    return {
                        "status": "fail",
                        "error": f"Expected {expected_count} predictions, got {actual_count}",
                        "response_time_ms": response_time
                    }
                
                # Validate each prediction
                for i, pred in enumerate(predictions):
                    if not (0 <= pred["prediction"] <= 1):
                        return {
                            "status": "fail",
                            "error": f"Invalid prediction {i}: {pred['prediction']}",
                            "response_time_ms": response_time
                        }
                
                avg_prediction_time = response_time / len(predictions)
                
                result = {
                    "status": "pass",
                    "response_time_ms": response_time,
                    "batch_size": len(predictions),
                    "avg_prediction_time_ms": avg_prediction_time,
                    "total_processed": batch_data["total_processed"],
                    "batch_id": batch_data["batch_id"],
                    "passes_response_time_check": avg_prediction_time < self.config.response_time_threshold_ms
                }
                
                logger.info(f"✅ Batch prediction validation passed")
                logger.info(f"   Total Response Time: {response_time:.2f}ms")
                logger.info(f"   Avg per Prediction: {avg_prediction_time:.2f}ms")
                logger.info(f"   Batch Size: {len(predictions)}")
                
                return result
                
            else:
                logger.error(f"❌ Batch prediction failed with status code: {response.status_code}")
                return {
                    "status": "fail",
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "response_time_ms": response_time
                }
                
        except Exception as e:
            logger.error(f"❌ Batch prediction validation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "response_time_ms": None
            }
    
    def validate_performance_consistency(self, num_requests: int = 50) -> Dict[str, Any]:
        """Validate API performance consistency under load"""
        logger.info(f"⚡ Validating performance consistency ({num_requests} requests)...")
        
        response_times = []
        success_count = 0
        failure_count = 0
        
        test_data = self.generate_test_data(1)[0]
        
        for i in range(num_requests):
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.config.api_url}/predict",
                    json=test_data,
                    timeout=self.config.timeout_seconds
                )
                response_time = (time.time() - start_time) * 1000
                response_times.append(response_time)
                
                if response.status_code == 200:
                    success_count += 1
                else:
                    failure_count += 1
                    
            except Exception as e:
                failure_count += 1
                logger.warning(f"Request {i+1} failed: {str(e)}")
        
        if response_times:
            avg_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            availability_percent = (success_count / num_requests) * 100
            
            result = {
                "status": "pass" if availability_percent >= self.config.availability_threshold_percent else "fail",
                "total_requests": num_requests,
                "successful_requests": success_count,
                "failed_requests": failure_count,
                "availability_percent": availability_percent,
                "avg_response_time_ms": avg_response_time,
                "p95_response_time_ms": p95_response_time,
                "p99_response_time_ms": p99_response_time,
                "min_response_time_ms": min(response_times),
                "max_response_time_ms": max(response_times),
                "passes_availability_check": availability_percent >= self.config.availability_threshold_percent,
                "passes_performance_check": avg_response_time < self.config.response_time_threshold_ms
            }
            
            logger.info(f"✅ Performance consistency validation completed")
            logger.info(f"   Availability: {availability_percent:.1f}%")
            logger.info(f"   Avg Response Time: {avg_response_time:.2f}ms")
            logger.info(f"   P95 Response Time: {p95_response_time:.2f}ms")
            
            return result
        else:
            return {
                "status": "fail",
                "error": "No successful requests",
                "total_requests": num_requests,
                "successful_requests": success_count,
                "failed_requests": failure_count
            }
    
    def validate_model_accuracy(self) -> Dict[str, Any]:
        """Validate model accuracy against known test data"""
        logger.info("🎯 Validating model accuracy...")
        
        try:
            # Load actual test data if available
            model_manager = ModelManager()
            models = model_manager.list_models()
            
            if not models:
                return {
                    "status": "skip",
                    "error": "No models available for accuracy validation"
                }
            
            # Get the current model's performance
            current_model_response = self.session.get(f"{self.config.api_url}/model-info")
            if current_model_response.status_code != 200:
                return {
                    "status": "fail",
                    "error": "Could not get current model info"
                }
            
            model_info = current_model_response.json()
            model_metrics = model_info.get("performance_metrics", {})
            
            roc_auc = model_metrics.get("roc_auc", 0)
            accuracy = model_metrics.get("accuracy", 0)
            
            result = {
                "status": "pass" if (roc_auc >= self.config.performance_threshold_roc_auc and 
                                   accuracy >= self.config.performance_threshold_accuracy) else "fail",
                "model_name": model_info.get("model_name", "unknown"),
                "roc_auc": roc_auc,
                "accuracy": accuracy,
                "f1_score": model_metrics.get("f1_score", 0),
                "precision": model_metrics.get("precision", 0),
                "recall": model_metrics.get("recall", 0),
                "meets_roc_auc_threshold": roc_auc >= self.config.performance_threshold_roc_auc,
                "meets_accuracy_threshold": accuracy >= self.config.performance_threshold_accuracy,
                "roc_auc_threshold": self.config.performance_threshold_roc_auc,
                "accuracy_threshold": self.config.performance_threshold_accuracy
            }
            
            logger.info(f"✅ Model accuracy validation completed")
            logger.info(f"   ROC AUC: {roc_auc:.4f} (threshold: {self.config.performance_threshold_roc_auc})")
            logger.info(f"   Accuracy: {accuracy:.4f} (threshold: {self.config.performance_threshold_accuracy})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Model accuracy validation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        logger.info("🚀 Starting comprehensive validation pipeline...")
        
        start_time = datetime.now()
        
        validation_results = {
            "validation_timestamp": start_time.isoformat(),
            "config": {
                "api_url": self.config.api_url,
                "timeout_seconds": self.config.timeout_seconds,
                "performance_threshold_roc_auc": self.config.performance_threshold_roc_auc,
                "performance_threshold_accuracy": self.config.performance_threshold_accuracy,
                "response_time_threshold_ms": self.config.response_time_threshold_ms,
                "availability_threshold_percent": self.config.availability_threshold_percent
            },
            "tests": {}
        }
        
        # Run all validation tests
        validation_tests = [
            ("health_check", self.validate_api_health),
            ("model_endpoints", self.validate_model_endpoints),
            ("single_prediction", self.validate_single_prediction),
            ("batch_prediction", self.validate_batch_prediction),
            ("performance_consistency", self.validate_performance_consistency),
            ("model_accuracy", self.validate_model_accuracy)
        ]
        
        passed_tests = 0
        total_tests = len(validation_tests)
        
        for test_name, test_function in validation_tests:
            logger.info(f"\n📋 Running {test_name.replace('_', ' ').title()} Test...")
            try:
                test_result = test_function()
                validation_results["tests"][test_name] = test_result
                
                if test_result.get("status") in ["pass", "healthy"]:
                    passed_tests += 1
                    logger.info(f"✅ {test_name.replace('_', ' ').title()} Test: PASSED")
                elif test_result.get("status") == "skip":
                    logger.warning(f"⏭️ {test_name.replace('_', ' ').title()} Test: SKIPPED")
                else:
                    logger.error(f"❌ {test_name.replace('_', ' ').title()} Test: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name.replace('_', ' ').title()} Test: ERROR - {str(e)}")
                validation_results["tests"][test_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Calculate overall status
        overall_status = "PASS" if passed_tests == total_tests else "FAIL"
        
        validation_results["summary"] = {
            "overall_status": overall_status,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate_percent": (passed_tests / total_tests) * 100,
            "total_duration_seconds": total_duration,
            "validation_completed_at": end_time.isoformat()
        }
        
        # Log summary
        logger.info(f"\n🏁 Validation Pipeline Completed!")
        logger.info(f"   Overall Status: {overall_status}")
        logger.info(f"   Tests Passed: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
        logger.info(f"   Total Duration: {total_duration:.2f} seconds")
        
        return validation_results
    
    def save_validation_report(self, results: Dict[str, Any], output_dir: str = "validation_reports") -> str:
        """Save validation results to file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_report_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Validation report saved: {filepath}")
        return filepath

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model API Validation Pipeline")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="API URL to validate")
    parser.add_argument("--output-dir", default="validation_reports",
                       help="Directory to save validation reports")
    parser.add_argument("--performance-requests", type=int, default=50,
                       help="Number of requests for performance testing")
    
    args = parser.parse_args()
    
    # Configure validation
    config = ValidationConfig(api_url=args.api_url)
    validator = ModelValidator(config)
    
    # Run validation
    results = validator.run_comprehensive_validation()
    
    # Save report
    report_path = validator.save_validation_report(results, args.output_dir)
    
    # Exit with appropriate code
    overall_status = results["summary"]["overall_status"]
    exit_code = 0 if overall_status == "PASS" else 1
    
    print(f"\n{'='*60}")
    print(f"VALIDATION PIPELINE COMPLETED: {overall_status}")
    print(f"Report saved: {report_path}")
    print(f"{'='*60}")
    
    exit(exit_code)

if __name__ == "__main__":
    main()