"""
Comprehensive Evaluation Runner for Numerics Processor Agent

This module provides automated evaluation capabilities with scenario-based testing,
performance benchmarking, and detailed reporting.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from .azure_evaluation import NumericsAgentEvaluator
from .config import EvaluationConfig, EvaluationScenarios, TestScenario, DEFAULT_CONFIG
from ..agent.core import NumericsAgent, AgentConfig


class EvaluationRunner:
    """Comprehensive evaluation runner with scenario-based testing"""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.evaluator = NumericsAgentEvaluator(config)
        self.results_history = []
        
    async def run_scenario_evaluation(self, scenario: TestScenario) -> Dict[str, Any]:
        """Run evaluation for a specific scenario"""
        
        print(f"\n{'='*60}")
        print(f"RUNNING SCENARIO: {scenario.name.upper()}")
        print(f"Description: {scenario.description}")
        print(f"{'='*60}")
        
        # Generate test data for this scenario
        data_path = await self._generate_scenario_data(scenario)
        
        # Setup output path for this scenario
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / "scenarios" / scenario.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"evaluation_{timestamp}.json"
        
        # Run evaluation
        result = await self.evaluator.run_evaluation(
            data_path=data_path,
            output_path=str(output_path)
        )
        
        # Analyze results against scenario criteria
        analysis = self._analyze_scenario_results(result, scenario)
        
        # Combine results with analysis
        scenario_result = {
            "scenario": scenario.name,
            "description": scenario.description,
            "timestamp": timestamp,
            "evaluation_result": result,
            "scenario_analysis": analysis,
            "data_path": data_path,
            "output_path": str(output_path)
        }
        
        return scenario_result
    
    async def _generate_scenario_data(self, scenario: TestScenario) -> str:
        """Generate test data for a specific scenario"""
        
        # Initialize agent
        agent_config = AgentConfig(
            model_name=self.config.model_name,
            temperature=0.1,
            max_tokens=1000
        )
        agent = NumericsAgent(agent_config)
        
        # Generate responses for scenario queries
        test_data = []
        
        for query in scenario.test_queries:
            start_time = datetime.now()
            response = await agent.chat(query)
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            
            test_data.append({
                "query": query,
                "response": response,
                "response_time": response_time,
                "scenario": scenario.name,
                "expected_tools": scenario.expected_tools,
                "timestamp": start_time.isoformat()
            })
        
        # Save scenario data
        data_dir = Path("evaluation_data") / "scenarios"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        data_path = data_dir / f"{scenario.name}_test_data.jsonl"
        
        with open(data_path, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        return str(data_path)
    
    def _analyze_scenario_results(self, result: Dict[str, Any], scenario: TestScenario) -> Dict[str, Any]:
        """Analyze evaluation results against scenario performance criteria"""
        
        metrics = result.get("metrics", {})
        rows = result.get("rows", [])
        
        analysis = {
            "scenario_name": scenario.name,
            "total_queries": len(rows),
            "performance_assessment": {},
            "criteria_met": {},
            "recommendations": []
        }
        
        # Check each performance criterion
        for criterion, threshold in scenario.performance_criteria.items():
            
            # Map criterion to actual metric names
            metric_mapping = {
                "accuracy": "response_quality.quality_score",
                "response_time": "response_time.time_score",
                "coherence": "coherence.coherence",
                "relevance": "relevance.relevance",
                "task_adherence": "task_adherence.task_adherence",
                "intent_resolution": "intent_resolution.intent_resolution",
                "completeness": "response_quality.quality_score"  # Proxy metric
            }
            
            metric_name = metric_mapping.get(criterion, criterion)
            actual_value = metrics.get(metric_name)
            
            if actual_value is not None:
                # For response_time, lower is better, so we need to invert the logic
                if criterion == "response_time":
                    # Convert time score to actual time and compare
                    criteria_met = actual_value >= 0.6  # Good time score threshold
                    performance_gap = threshold - actual_value if actual_value < threshold else 0
                else:
                    criteria_met = actual_value >= threshold
                    performance_gap = threshold - actual_value if actual_value < threshold else 0
                
                analysis["performance_assessment"][criterion] = {
                    "expected": threshold,
                    "actual": actual_value,
                    "met": criteria_met,
                    "gap": performance_gap
                }
                
                analysis["criteria_met"][criterion] = criteria_met
                
                # Generate recommendations for unmet criteria
                if not criteria_met:
                    analysis["recommendations"].append(
                        f"Improve {criterion}: target {threshold}, current {actual_value:.3f}"
                    )
        
        # Calculate overall scenario score
        met_criteria = sum(analysis["criteria_met"].values())
        total_criteria = len(analysis["criteria_met"])
        analysis["scenario_score"] = met_criteria / total_criteria if total_criteria > 0 else 0
        
        # Overall assessment
        if analysis["scenario_score"] >= 0.8:
            analysis["overall_assessment"] = "Excellent"
        elif analysis["scenario_score"] >= 0.6:
            analysis["overall_assessment"] = "Good"
        elif analysis["scenario_score"] >= 0.4:
            analysis["overall_assessment"] = "Fair"
        else:
            analysis["overall_assessment"] = "Needs Improvement"
        
        return analysis
    
    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation across all scenarios"""
        
        print(f"\n{'='*80}")
        print("STARTING COMPREHENSIVE AGENT EVALUATION")
        print(f"Configuration: {self.config.evaluation_name}")
        print(f"{'='*80}")
        
        start_time = datetime.now()
        all_scenarios = EvaluationScenarios.get_all_scenarios()
        scenario_results = []
        
        # Run evaluation for each scenario
        for scenario in all_scenarios:
            try:
                scenario_result = await self.run_scenario_evaluation(scenario)
                scenario_results.append(scenario_result)
                
                # Brief summary for this scenario
                analysis = scenario_result["scenario_analysis"]
                print(f"\n✓ {scenario.name}: {analysis['overall_assessment']} "
                      f"(Score: {analysis['scenario_score']:.2f})")
                
            except Exception as e:
                print(f"\n✗ Error in scenario {scenario.name}: {str(e)}")
                scenario_results.append({
                    "scenario": scenario.name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Compile comprehensive results
        comprehensive_result = {
            "evaluation_id": f"comprehensive_{start_time.strftime('%Y%m%d_%H%M%S')}",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_seconds": total_duration,
            "configuration": {
                "model_name": self.config.model_name,
                "num_scenarios": len(all_scenarios),
                "evaluation_name": self.config.evaluation_name
            },
            "scenario_results": scenario_results,
            "summary": self._generate_comprehensive_summary(scenario_results)
        }
        
        # Save comprehensive results
        output_path = self._save_comprehensive_results(comprehensive_result)
        comprehensive_result["output_path"] = output_path
        
        # Print final summary
        self._print_comprehensive_summary(comprehensive_result)
        
        return comprehensive_result
    
    def _generate_comprehensive_summary(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary across all scenarios"""
        
        successful_scenarios = [r for r in scenario_results if "error" not in r]
        failed_scenarios = [r for r in scenario_results if "error" in r]
        
        if not successful_scenarios:
            return {
                "total_scenarios": len(scenario_results),
                "successful_scenarios": 0,
                "failed_scenarios": len(failed_scenarios),
                "overall_score": 0.0,
                "overall_rating": "Failed"
            }
        
        # Calculate averages across successful scenarios
        scenario_scores = [r["scenario_analysis"]["scenario_score"] for r in successful_scenarios]
        overall_score = sum(scenario_scores) / len(scenario_scores)
        
        # Count performance ratings
        ratings = [r["scenario_analysis"]["overall_assessment"] for r in successful_scenarios]
        rating_counts = {rating: ratings.count(rating) for rating in set(ratings)}
        
        # Determine overall rating
        if overall_score >= 0.8:
            overall_rating = "Excellent"
        elif overall_score >= 0.6:
            overall_rating = "Good"
        elif overall_score >= 0.4:
            overall_rating = "Fair"
        else:
            overall_rating = "Needs Improvement"
        
        return {
            "total_scenarios": len(scenario_results),
            "successful_scenarios": len(successful_scenarios),
            "failed_scenarios": len(failed_scenarios),
            "overall_score": overall_score,
            "overall_rating": overall_rating,
            "scenario_rating_distribution": rating_counts,
            "recommendations": self._generate_overall_recommendations(successful_scenarios)
        }
    
    def _generate_overall_recommendations(self, scenario_results: List[Dict[str, Any]]) -> List[str]:
        """Generate overall recommendations based on all scenario results"""
        
        recommendations = []
        
        # Collect all recommendations from scenarios
        all_recs = []
        for result in scenario_results:
            all_recs.extend(result["scenario_analysis"].get("recommendations", []))
        
        # Count frequency of similar recommendations
        rec_types = {}
        for rec in all_recs:
            rec_type = rec.split(":")[0]  # Get the type (before colon)
            rec_types[rec_type] = rec_types.get(rec_type, 0) + 1
        
        # Generate prioritized recommendations
        if "Improve response_time" in rec_types:
            recommendations.append("Optimize response time across multiple scenarios")
        
        if "Improve coherence" in rec_types:
            recommendations.append("Enhance response coherence and flow")
        
        if "Improve accuracy" in rec_types:
            recommendations.append("Focus on numerical accuracy improvements")
        
        if "Improve task_adherence" in rec_types:
            recommendations.append("Better instruction following and task completion")
        
        # Add general recommendations based on overall performance
        failed_scenarios = len([r for r in scenario_results if r["scenario_analysis"]["scenario_score"] < 0.6])
        if failed_scenarios > 0:
            recommendations.append(f"Address issues in {failed_scenarios} underperforming scenarios")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _save_comprehensive_results(self, result: Dict[str, Any]) -> str:
        """Save comprehensive evaluation results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / "comprehensive"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"comprehensive_evaluation_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        
        return str(output_path)
    
    def _print_comprehensive_summary(self, result: Dict[str, Any]) -> None:
        """Print comprehensive evaluation summary"""
        
        summary = result["summary"]
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        print(f"Evaluation ID: {result['evaluation_id']}")
        print(f"Total Duration: {result['total_duration_seconds']:.1f} seconds")
        print(f"Model: {result['configuration']['model_name']}")
        
        print(f"\nScenario Results:")
        print(f"  Total Scenarios: {summary['total_scenarios']}")
        print(f"  Successful: {summary['successful_scenarios']}")
        print(f"  Failed: {summary['failed_scenarios']}")
        
        print(f"\nOverall Performance:")
        print(f"  Score: {summary['overall_score']:.3f}")
        print(f"  Rating: {summary['overall_rating']}")
        
        if summary.get('scenario_rating_distribution'):
            print(f"\nRating Distribution:")
            for rating, count in summary['scenario_rating_distribution'].items():
                print(f"  {rating}: {count}")
        
        if summary.get('recommendations'):
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print(f"\nDetailed results saved to: {result['output_path']}")
        print(f"{'='*80}")
    
    async def run_quick_evaluation(self, num_queries: int = 5) -> Dict[str, Any]:
        """Run a quick evaluation with a small subset of queries"""
        
        print(f"\nRunning quick evaluation with {num_queries} queries...")
        
        # Use basic math scenario for quick evaluation
        basic_scenario = EvaluationScenarios.get_basic_math_scenario()
        basic_scenario.test_queries = basic_scenario.test_queries[:num_queries]
        
        result = await self.run_scenario_evaluation(basic_scenario)
        
        print(f"Quick evaluation completed!")
        print(f"Score: {result['scenario_analysis']['scenario_score']:.3f}")
        print(f"Rating: {result['scenario_analysis']['overall_assessment']}")
        
        return result
    
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmarking focused on speed and efficiency"""
        
        print(f"\nRunning performance benchmark...")
        
        # Create a performance-focused scenario
        perf_queries = [
            "Calculate 2 + 2",
            "What is 10 * 5?",
            "Find mean of [1,2,3]",
            "Sum of 1 to 5",
            "Square root of 25"
        ]
        
        perf_scenario = TestScenario(
            name="performance_benchmark",
            description="Quick numerical operations for performance testing",
            test_queries=perf_queries,
            expected_tools=["numerical_computation"],
            performance_criteria={
                "response_time": 2.0,  # Very strict time requirement
                "accuracy": 0.95
            }
        )
        
        result = await self.run_scenario_evaluation(perf_scenario)
        
        # Calculate additional performance metrics
        rows = result["evaluation_result"]["rows"]
        response_times = [row["inputs.response_time"] for row in rows if "inputs.response_time" in row]
        
        if response_times:
            perf_metrics = {
                "avg_response_time": sum(response_times) / len(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "queries_under_2s": sum(1 for t in response_times if t < 2.0),
                "queries_per_minute": 60 / (sum(response_times) / len(response_times)) if response_times else 0
            }
            
            result["performance_metrics"] = perf_metrics
            
            print(f"Performance Benchmark Results:")
            print(f"  Average Response Time: {perf_metrics['avg_response_time']:.2f}s")
            print(f"  Queries Under 2s: {perf_metrics['queries_under_2s']}/{len(response_times)}")
            print(f"  Estimated Queries/Minute: {perf_metrics['queries_per_minute']:.1f}")
        
        return result


async def main():
    """Main function to run comprehensive evaluation"""
    
    # Initialize evaluation runner
    runner = EvaluationRunner()
    
    # Run comprehensive evaluation
    result = await runner.run_comprehensive_evaluation()
    
    return result


if __name__ == "__main__":
    asyncio.run(main())