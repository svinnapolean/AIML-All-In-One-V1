#!/usr/bin/env python3
"""
Quick test script to verify all components can be imported successfully
"""

import sys
import traceback

def test_import(module_name, description):
    """Test importing a module and print results"""
    try:
        __import__(module_name)
        print(f"✅ {description}: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ {description}: FAILED - {str(e)}")
        return False

def main():
    """Run all import tests"""
    print("🔍 Testing Numerics Processor Components...\n")
    
    tests = [
        ("src.api.routers.health", "Health Router"),
        ("src.api.routers.models", "Models Router"),
        ("src.api.routers.agent", "Agent Router"),
        ("src.api.routers.data", "Data Router"),
        ("src.api.main", "Main API Application"),
        ("src.models.training", "Model Training"),
        ("src.models.testing", "Model Testing"),
        ("src.models.evaluation", "Model Evaluation"),
        ("src.agent.core", "AI Agent Core"),
        ("src.agent.tools", "Agent Tools"),
        ("src.utils.config", "Configuration"),
        ("src.evaluation.azure_evaluation", "Azure Evaluation"),
    ]
    
    passed = 0
    total = len(tests)
    
    for module, description in tests:
        if test_import(module, description):
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} components imported successfully")
    
    if passed == total:
        print("🎉 All components are working! Your project is ready to run.")
        return 0
    else:
        print("⚠️  Some components have issues. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())