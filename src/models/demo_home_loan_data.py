"""
HomeLoanData Class Usage Examples

This script demonstrates how to use the HomeLoanData class for individual tasks
or run the complete pipeline for loan default prediction analysis.
"""

from home_loan_data import HomeLoanData

def demo_individual_tasks():
    """Demonstrate individual task execution"""
    print("🎯 DEMO: Individual Task Execution")
    print("=" * 50)
    
    # Initialize the class
    loan_data = HomeLoanData('loan_data/loan_data.csv')
    
    # Task 1: Load dataset
    print("\n1️⃣ Loading Dataset...")
    loan_data.load_dataset()
    
    # Task 2: Check null values
    print("\n2️⃣ Checking Null Values...")
    null_analysis = loan_data.check_null_values()
    
    # Task 3: Analyze target distribution
    print("\n3️⃣ Analyzing Target Distribution...")
    target_analysis = loan_data.analyze_target_distribution()
    
    # Task 4: Balance dataset
    print("\n4️⃣ Balancing Dataset...")
    loan_data.balance_dataset(method='smote')
    
    # Task 5: Plot data distribution
    print("\n5️⃣ Plotting Data Distribution...")
    loan_data.plot_data_distribution()
    
    # Task 6: Encode columns
    print("\n6️⃣ Encoding Columns...")
    result = loan_data.encode_columns()
    if result:
        X, y = result
    
    # Tasks 7 & 8: Calculate metrics
    print("\n7️⃣8️⃣ Calculating Sensitivity and ROC AUC...")
    metrics = loan_data.train_model_and_calculate_metrics()
    
    if metrics:
        print(f"\n📋 Final Individual Task Results:")
        print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"   Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics

def demo_complete_pipeline():
    """Demonstrate complete pipeline execution"""
    print("🚀 DEMO: Complete Pipeline Execution")
    print("=" * 50)
    
    # Initialize and run complete pipeline
    loan_data = HomeLoanData('loan_data/loan_data.csv')
    metrics = loan_data.analyze_complete_pipeline()
    
    return metrics

def demo_different_balancing_methods():
    """Demonstrate different balancing techniques"""
    print("⚖️ DEMO: Different Balancing Methods")
    print("=" * 50)
    
    methods = ['smote', 'undersample', 'smote_tomek']
    results = {}
    
    for method in methods:
        print(f"\n🔄 Testing {method.upper()} method...")
        
        # Initialize fresh instance for each method
        loan_data = HomeLoanData('loan_data/loan_data.csv')
        loan_data.load_dataset()
        
        # Balance with different method
        loan_data.balance_dataset(method=method)
        
        # Encode and train
        loan_data.encode_columns()
        metrics = loan_data.train_model_and_calculate_metrics()
        
        results[method] = metrics
        
        if metrics:
            print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"   Sensitivity: {metrics['sensitivity']:.4f}")
        else:
            print(f"   ❌ Training failed for {method}")
    
    # Compare results
    print(f"\n📊 COMPARISON OF BALANCING METHODS:")
    print(f"{'Method':<15} {'ROC AUC':<10} {'Sensitivity':<12} {'Accuracy':<10}")
    print("-" * 50)
    
    for method, metrics in results.items():
        if metrics:
            print(f"{method.upper():<15} {metrics['roc_auc']:<10.4f} {metrics['sensitivity']:<12.4f} {metrics['accuracy']:<10.4f}")
        else:
            print(f"{method.upper():<15} {'FAILED':<10} {'FAILED':<12} {'FAILED':<10}")
    
    return results

def quick_analysis():
    """Quick analysis with minimal output"""
    print("⚡ DEMO: Quick Analysis")
    print("=" * 30)
    
    # Initialize
    loan_data = HomeLoanData('loan_data/loan_data.csv')
    
    # Load and analyze
    loan_data.load_dataset()
    loan_data.analyze_target_distribution()
    
    # Balance and train
    loan_data.balance_dataset(method='smote')
    loan_data.encode_columns()
    metrics = loan_data.train_model_and_calculate_metrics()
    
    if metrics:
        print(f"\n⚡ QUICK RESULTS:")
        print(f"   🎯 Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"   📈 ROC AUC: {metrics['roc_auc']:.3f}")
        print(f"   ✅ Accuracy: {metrics['accuracy']:.3f}")
    else:
        print(f"\n❌ Quick analysis failed")
    
    return metrics

if __name__ == "__main__":
    print("🏠 HomeLoanData Class Demonstrations")
    print("=" * 60)
    
    # Choose which demo to run
    demo_choice = input("""
Choose a demonstration:
1. Individual Tasks (step by step)
2. Complete Pipeline (all at once)
3. Different Balancing Methods
4. Quick Analysis
5. All Demos

Enter your choice (1-5): """)
    
    if demo_choice == "1":
        demo_individual_tasks()
    elif demo_choice == "2":
        demo_complete_pipeline()
    elif demo_choice == "3":
        demo_different_balancing_methods()
    elif demo_choice == "4":
        quick_analysis()
    elif demo_choice == "5":
        print("🎯 Running all demonstrations...")
        print("\n" + "="*60)
        demo_individual_tasks()
        print("\n" + "="*60)
        demo_different_balancing_methods()
        print("\n" + "="*60)
        quick_analysis()
    else:
        print("Running complete pipeline by default...")
        demo_complete_pipeline()
    
    print(f"\n🎉 Demonstration completed!")