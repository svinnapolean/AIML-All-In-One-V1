"""
Azure Development Kit (ADK) Evaluation for Numerics Processor Agent

This module provides comprehensive evaluation capabilities for the AI agent using
Azure AI Evaluation SDK with built-in evaluators and custom evaluators.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
from datetime import datetime

# Azure AI Evaluation imports
from azure.ai.evaluation import (
    evaluate,
    TaskAdherenceEvaluator,
    IntentResolutionEvaluator,
    ToolCallAccuracyEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    RelevanceEvaluator,
    OpenAIModelConfiguration
)

# Local imports
from ..agent.core import NumericsAgent, AgentConfig
from ..config import Config


class AgentResponseEvaluator:
    """Custom code-based evaluator for agent response quality"""
    
    def __init__(self):
        self.name = "agent_response_quality"
    
    def __call__(self, *, query: str, response: str, **kwargs) -> Dict[str, Union[float, str]]:
        """Evaluate agent response quality based on objective metrics"""
        
        # Check if response contains numerical content
        has_numbers = any(char.isdigit() for char in response)
        
        # Check response length appropriateness
        response_length = len(response)
        length_score = 1.0 if 50 <= response_length <= 1000 else 0.5
        
        # Check for mathematical indicators
        math_keywords = ['calculate', 'compute', 'result', 'answer', 'solution', 'analysis']
        has_math_content = any(keyword in response.lower() for keyword in math_keywords)
        
        # Check for proper formatting
        has_structure = any(char in response for char in ['\n', '•', '-', '*'])
        
        # Calculate overall score
        factors = [
            has_numbers,
            length_score > 0.8,
            has_math_content,
            has_structure
        ]
        
        quality_score = sum(factors) / len(factors)
        
        return {
            "quality_score": quality_score,
            "has_numerical_content": has_numbers,
            "response_length": response_length,
            "has_mathematical_content": has_math_content,
            "is_well_structured": has_structure,
            "reasoning": f"Response quality: {quality_score:.2f}. Contains numbers: {has_numbers}, Math content: {has_math_content}"
        }


class ResponseTimeEvaluator:
    """Custom code-based evaluator for response time performance"""
    
    def __init__(self):
        self.name = "response_time"
    
    def __call__(self, *, response_time: float, **kwargs) -> Dict[str, Union[float, str]]:
        """Evaluate response time performance"""
        
        # Define acceptable response time thresholds (in seconds)
        excellent_threshold = 2.0
        good_threshold = 5.0
        acceptable_threshold = 10.0
        
        if response_time <= excellent_threshold:
            time_score = 1.0
            performance_rating = "excellent"
        elif response_time <= good_threshold:
            time_score = 0.8
            performance_rating = "good"
        elif response_time <= acceptable_threshold:
            time_score = 0.6
            performance_rating = "acceptable"
        else:
            time_score = 0.3
            performance_rating = "slow"
        
        return {
            "time_score": time_score,
            "response_time_seconds": response_time,
            "performance_rating": performance_rating,
            "reasoning": f"Response time: {response_time:.2f}s - {performance_rating}"
        }


class NumericsAgentEvaluator:
    """Main evaluator class for the Numerics Processing Agent"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model_config = self._setup_model_config()
        self.agent = None
        
    def _setup_model_config(self) -> OpenAIModelConfiguration:
        """Setup model configuration for evaluators"""
        
        # Use GitHub models configuration matching the agent
        return OpenAIModelConfiguration(
            type="openai",
            model="gpt-4o-mini",
            base_url="https://models.inference.ai.azure.com",
            api_key=os.getenv("GITHUB_TOKEN", "")
        )
    
    async def _initialize_agent(self) -> NumericsAgent:
        """Initialize the agent for testing"""
        if not self.agent:
            agent_config = AgentConfig(
                model_name="gpt-4o-mini",
                temperature=0.1,
                max_tokens=1000
            )
            self.agent = NumericsAgent(agent_config)
        return self.agent
    
    async def generate_test_data(self, num_samples: int = 10) -> str:
        """Generate test data for evaluation"""
        
        # Sample test queries for numerics processing
        test_queries = [
            "Calculate the mean of [1, 2, 3, 4, 5]",
            "What is the standard deviation of [10, 20, 30, 40, 50]?",
            "Analyze the dataset [1.5, 2.3, 4.1, 3.7, 2.9] and provide statistical insights",
            "Perform linear regression on x=[1,2,3,4,5] and y=[2,4,6,8,10]",
            "Create a visualization for the data points (1,2), (2,4), (3,6), (4,8)",
            "Calculate correlation between height [170,175,180,165,190] and weight [70,75,80,65,85]",
            "What is the median of [7, 3, 9, 1, 5, 8, 2]?",
            "Explain the trend in sales data: [100, 120, 110, 140, 150, 160]",
            "Calculate the moving average with window=3 for [10, 15, 20, 25, 30, 35]",
            "Perform hypothesis testing on sample data [23, 25, 27, 24, 26, 28, 22]"
        ]
        
        # Select queries based on num_samples
        selected_queries = test_queries[:min(num_samples, len(test_queries))]
        
        # Generate responses using the agent
        agent = await self._initialize_agent()
        test_data = []
        
        for query in selected_queries:
            start_time = datetime.now()
            response = await agent.chat(query)
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            
            # Extract tools used from response context if available
            tools_used = self._extract_tools_from_response(response)
            
            test_data.append({
                "query": query,
                "response": response,
                "response_time": response_time,
                "tools_used": tools_used,
                "timestamp": start_time.isoformat()
            })
        
        # Save to JSONL format
        output_path = Path("evaluation_data") / "agent_test_data.jsonl"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        return str(output_path)
    
    def _extract_tools_from_response(self, response: str) -> List[str]:
        """Extract information about tools used in the response"""
        # This would be enhanced to parse actual tool usage from agent context
        # For now, infer based on response content
        tools = []
        
        if "calculate" in response.lower() or "computation" in response.lower():
            tools.append("numerical_computation")
        if "visualization" in response.lower() or "plot" in response.lower():
            tools.append("data_visualization")
        if "analysis" in response.lower() or "statistical" in response.lower():
            tools.append("statistical_analysis")
        if "inference" in response.lower() or "prediction" in response.lower():
            tools.append("model_inference")
            
        return tools
    
    def setup_evaluators(self) -> Dict[str, Any]:
        """Setup all evaluators for comprehensive evaluation"""
        
        evaluators = {
            # Built-in Azure AI evaluators for agent capabilities
            "task_adherence": TaskAdherenceEvaluator(model_config=self.model_config),
            "intent_resolution": IntentResolutionEvaluator(model_config=self.model_config),
            "coherence": CoherenceEvaluator(model_config=self.model_config),
            "fluency": FluencyEvaluator(model_config=self.model_config),
            "relevance": RelevanceEvaluator(model_config=self.model_config),
            
            # Custom evaluators for specific agent metrics
            "response_quality": AgentResponseEvaluator(),
            "response_time": ResponseTimeEvaluator(),
        }
        
        return evaluators
    
    def setup_evaluator_config(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Setup evaluator configuration with column mappings"""
        
        return {
            "task_adherence": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}"
                }
            },
            "intent_resolution": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}"
                }
            },
            "coherence": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}"
                }
            },
            "fluency": {
                "column_mapping": {
                    "response": "${data.response}"
                }
            },
            "relevance": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}"
                }
            },
            "response_quality": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}"
                }
            },
            "response_time": {
                "column_mapping": {
                    "response_time": "${data.response_time}"
                }
            }
        }
    
    async def run_evaluation(self, data_path: Optional[str] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation of the agent"""
        
        # Generate test data if not provided
        if data_path is None:
            print("Generating test data...")
            data_path = await self.generate_test_data()
            print(f"Test data saved to: {data_path}")
        
        # Setup output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("evaluation_results")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"agent_evaluation_{timestamp}.json"
        
        # Setup evaluators
        evaluators = self.setup_evaluators()
        evaluator_config = self.setup_evaluator_config()
        
        print("Running evaluation with Azure AI Evaluation SDK...")
        print(f"Data source: {data_path}")
        print(f"Evaluators: {list(evaluators.keys())}")
        
        try:
            # Run evaluation using Azure AI Evaluation SDK
            result = evaluate(
                data=data_path,
                evaluators=evaluators,
                evaluator_config=evaluator_config,
                output_path=str(output_path),
                evaluation_name=f"numerics_agent_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            print(f"Evaluation completed successfully!")
            print(f"Results saved to: {output_path}")
            
            # Print summary metrics
            self._print_evaluation_summary(result)
            
            return result
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            raise
    
    def _print_evaluation_summary(self, result: Dict[str, Any]) -> None:
        """Print a summary of evaluation results"""
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        metrics = result.get("metrics", {})
        
        print(f"Total samples evaluated: {len(result.get('rows', []))}")
        print("\nMetrics Summary:")
        print("-" * 30)
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric_name}: {value:.3f}")
            else:
                print(f"{metric_name}: {value}")
        
        # Calculate average scores for key metrics
        key_metrics = [
            "task_adherence.task_adherence",
            "intent_resolution.intent_resolution", 
            "coherence.coherence",
            "fluency.fluency",
            "relevance.relevance",
            "response_quality.quality_score",
            "response_time.time_score"
        ]
        
        available_scores = []
        for metric in key_metrics:
            if metric in metrics and isinstance(metrics[metric], (int, float)):
                available_scores.append(metrics[metric])
        
        if available_scores:
            overall_score = sum(available_scores) / len(available_scores)
            print(f"\nOverall Agent Performance Score: {overall_score:.3f}")
            
            # Performance rating
            if overall_score >= 0.8:
                rating = "Excellent"
            elif overall_score >= 0.6:
                rating = "Good"
            elif overall_score >= 0.4:
                rating = "Fair"
            else:
                rating = "Needs Improvement"
            
            print(f"Performance Rating: {rating}")
        
        print("="*50)
    
    async def run_continuous_evaluation(self, interval_minutes: int = 60) -> None:
        """Run evaluation continuously at specified intervals"""
        
        print(f"Starting continuous evaluation with {interval_minutes} minute intervals...")
        
        while True:
            try:
                await self.run_evaluation()
                print(f"Waiting {interval_minutes} minutes until next evaluation...")
                await asyncio.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                print("Continuous evaluation stopped by user.")
                break
            except Exception as e:
                print(f"Error in continuous evaluation: {e}")
                print(f"Retrying in {interval_minutes} minutes...")
                await asyncio.sleep(interval_minutes * 60)


async def main():
    """Main function to run agent evaluation"""
    
    # Initialize evaluator
    evaluator = NumericsAgentEvaluator()
    
    # Run evaluation
    result = await evaluator.run_evaluation()
    
    return result


if __name__ == "__main__":
    asyncio.run(main())