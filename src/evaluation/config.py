"""
Evaluation Configuration for Numerics Processor Agent

This module provides configuration settings for different evaluation scenarios
and environments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import os


@dataclass
class EvaluationConfig:
    """Configuration for agent evaluation"""
    
    # Model configuration
    model_name: str = "gpt-4o-mini"
    model_base_url: str = "https://models.inference.ai.azure.com"
    api_key_env: str = "GITHUB_TOKEN"
    
    # Evaluation settings
    num_test_samples: int = 10
    evaluation_name: str = "numerics_agent_evaluation"
    
    # Output settings
    output_dir: str = "evaluation_results"
    save_detailed_results: bool = True
    
    # Performance thresholds
    response_time_thresholds: Dict[str, float] = None
    quality_score_threshold: float = 0.7
    
    # Evaluator settings
    use_builtin_evaluators: bool = True
    use_custom_evaluators: bool = True
    
    def __post_init__(self):
        if self.response_time_thresholds is None:
            self.response_time_thresholds = {
                "excellent": 2.0,
                "good": 5.0,
                "acceptable": 10.0
            }


@dataclass
class TestScenario:
    """Configuration for specific test scenarios"""
    
    name: str
    description: str
    test_queries: List[str]
    expected_tools: List[str]
    performance_criteria: Dict[str, Any]


class EvaluationScenarios:
    """Predefined evaluation scenarios for different capabilities"""
    
    @staticmethod
    def get_basic_math_scenario() -> TestScenario:
        """Basic mathematical computation scenario"""
        return TestScenario(
            name="basic_math",
            description="Basic mathematical operations and calculations",
            test_queries=[
                "Calculate the mean of [1, 2, 3, 4, 5]",
                "What is the sum of 25 + 37 + 18?",
                "Find the square root of 144",
                "Calculate 15% of 200"
            ],
            expected_tools=["numerical_computation"],
            performance_criteria={
                "accuracy": 0.95,
                "response_time": 3.0,
                "coherence": 0.8
            }
        )
    
    @staticmethod
    def get_statistical_analysis_scenario() -> TestScenario:
        """Statistical analysis scenario"""
        return TestScenario(
            name="statistical_analysis",
            description="Statistical analysis and data insights",
            test_queries=[
                "Calculate standard deviation of [10, 20, 30, 40, 50]",
                "Analyze the distribution of [1.5, 2.3, 4.1, 3.7, 2.9, 5.1, 3.2]",
                "What is the correlation between x=[1,2,3,4,5] and y=[2,4,6,8,9]?",
                "Perform t-test on samples [23, 25, 27, 24, 26] and [28, 30, 29, 31, 32]"
            ],
            expected_tools=["statistical_analysis"],
            performance_criteria={
                "accuracy": 0.90,
                "response_time": 5.0,
                "coherence": 0.85,
                "relevance": 0.80
            }
        )
    
    @staticmethod
    def get_data_visualization_scenario() -> TestScenario:
        """Data visualization scenario"""
        return TestScenario(
            name="data_visualization",
            description="Creating charts and visualizations",
            test_queries=[
                "Create a line plot for data points (1,2), (2,4), (3,6), (4,8)",
                "Generate a histogram for values [1, 2, 2, 3, 3, 3, 4, 4, 5]",
                "Plot a scatter chart for height vs weight data",
                "Create a bar chart showing monthly sales data"
            ],
            expected_tools=["data_visualization"],
            performance_criteria={
                "completeness": 0.85,
                "response_time": 7.0,
                "coherence": 0.80,
                "task_adherence": 0.85
            }
        )
    
    @staticmethod
    def get_machine_learning_scenario() -> TestScenario:
        """Machine learning and model inference scenario"""
        return TestScenario(
            name="machine_learning",
            description="Machine learning tasks and model operations",
            test_queries=[
                "Perform linear regression on x=[1,2,3,4,5] and y=[2,4,6,8,10]",
                "Predict the next value in sequence [1, 4, 9, 16, 25]",
                "Classify this data point using the trained model",
                "What's the accuracy of the model on test data?"
            ],
            expected_tools=["model_inference"],
            performance_criteria={
                "accuracy": 0.85,
                "response_time": 8.0,
                "coherence": 0.80,
                "intent_resolution": 0.85
            }
        )
    
    @staticmethod
    def get_complex_analysis_scenario() -> TestScenario:
        """Complex multi-step analysis scenario"""
        return TestScenario(
            name="complex_analysis",
            description="Complex multi-step analytical tasks",
            test_queries=[
                "Analyze the sales trend: [100, 120, 110, 140, 150, 160] and forecast next 3 months",
                "Given financial data, calculate ROI and provide investment recommendations",
                "Perform comprehensive data analysis on customer behavior dataset",
                "Create a complete statistical report with visualizations"
            ],
            expected_tools=["statistical_analysis", "data_visualization", "model_inference"],
            performance_criteria={
                "completeness": 0.80,
                "response_time": 15.0,
                "coherence": 0.85,
                "task_adherence": 0.80,
                "relevance": 0.85
            }
        )
    
    @staticmethod
    def get_all_scenarios() -> List[TestScenario]:
        """Get all predefined test scenarios"""
        return [
            EvaluationScenarios.get_basic_math_scenario(),
            EvaluationScenarios.get_statistical_analysis_scenario(),
            EvaluationScenarios.get_data_visualization_scenario(),
            EvaluationScenarios.get_machine_learning_scenario(),
            EvaluationScenarios.get_complex_analysis_scenario()
        ]


class EvaluationMetrics:
    """Standard evaluation metrics and their descriptions"""
    
    BUILTIN_METRICS = {
        "task_adherence": {
            "description": "How well the agent follows instructions and task requirements",
            "range": "[0, 1]",
            "higher_is_better": True
        },
        "intent_resolution": {
            "description": "How well the agent identifies and resolves user intent",
            "range": "[0, 1]", 
            "higher_is_better": True
        },
        "coherence": {
            "description": "How well the response flows and makes sense",
            "range": "[0, 1]",
            "higher_is_better": True
        },
        "fluency": {
            "description": "Grammatical correctness and natural language flow",
            "range": "[0, 1]",
            "higher_is_better": True
        },
        "relevance": {
            "description": "How relevant the response is to the query",
            "range": "[0, 1]",
            "higher_is_better": True
        }
    }
    
    CUSTOM_METRICS = {
        "response_quality": {
            "description": "Overall quality of agent response for numerical tasks",
            "range": "[0, 1]",
            "higher_is_better": True
        },
        "response_time": {
            "description": "Time taken to generate response",
            "range": "[0, ∞]",
            "higher_is_better": False
        },
        "numerical_accuracy": {
            "description": "Accuracy of numerical computations",
            "range": "[0, 1]",
            "higher_is_better": True
        },
        "tool_usage_efficiency": {
            "description": "How efficiently the agent uses available tools",
            "range": "[0, 1]",
            "higher_is_better": True
        }
    }
    
    @classmethod
    def get_all_metrics(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available metrics"""
        return {**cls.BUILTIN_METRICS, **cls.CUSTOM_METRICS}
    
    @classmethod
    def get_metric_info(cls, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific metric"""
        all_metrics = cls.get_all_metrics()
        return all_metrics.get(metric_name)


# Default configuration instance
DEFAULT_CONFIG = EvaluationConfig()