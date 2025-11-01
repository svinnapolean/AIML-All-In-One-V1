"""
AI Agent Learning Module - Core Implementation using Microsoft Agent Framework

This educational module demonstrates comprehensive AI agent development including:
- Agent architecture and design patterns
- Microsoft Agent Framework integration for learning
- GitHub Models integration for cost-effective AI development
- Tool creation and management for agent capabilities
- Conversation handling and context management
- Integration with ML models for intelligent responses

Learning Objectives:
- Understand agent-based AI system architecture
- Learn Microsoft Agent Framework best practices
- Implement cost-effective AI using GitHub models
- Master agent evaluation and optimization techniques
- Apply Azure ADK for comprehensive agent assessment
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

# Microsoft Agent Framework imports
# Note: Requires 'pip install agent-framework-azure-ai --pre'
try:
    from agent_framework import ChatAgent
    from agent_framework.openai import OpenAIChatClient
    from openai import AsyncOpenAI
except ImportError:
    print("Warning: Microsoft Agent Framework not installed. Run: pip install agent-framework-azure-ai --pre")
    # Mock classes for development
    class ChatAgent:
        def __init__(self, *args, **kwargs): pass
    class OpenAIChatClient:
        def __init__(self, *args, **kwargs): pass
    class AsyncOpenAI:
        def __init__(self, *args, **kwargs): pass

# Local imports
from ..models.training import ModelTrainer
from ..models.evaluation import ModelEvaluator


@dataclass
class AgentConfig:
    """Configuration for the Numerics AI Agent"""
    
    # Model configuration
    model_id: str = "openai/gpt-4.1-mini"  # GitHub model
    github_token: str = None
    
    # Agent behavior
    agent_name: str = "NumericsProcessor"
    system_instructions: str = None
    max_conversation_turns: int = 50
    
    # Tool configuration
    enable_model_inference: bool = True
    enable_data_analysis: bool = True
    enable_visualization: bool = True
    
    # Paths
    models_path: str = "models/trained_models"
    data_path: str = "data"
    results_path: str = "results"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/agent.log"
    
    def __post_init__(self):
        # Set default system instructions
        if self.system_instructions is None:
            self.system_instructions = self._get_default_instructions()
        
        # Get GitHub token from environment if not provided
        if self.github_token is None:
            self.github_token = os.getenv("GITHUB_TOKEN")
            if not self.github_token:
                raise ValueError("GitHub token is required. Set GITHUB_TOKEN environment variable or provide github_token parameter.")
        
        # Create directories
        for path in [self.models_path, self.data_path, self.results_path, os.path.dirname(self.log_file)]:
            os.makedirs(path, exist_ok=True)
    
    def _get_default_instructions(self) -> str:
        return """You are NumericsProcessor, an AI agent specialized in numerical data processing and machine learning tasks. Your capabilities include:

1. **Model Training & Evaluation**: Train, test, and evaluate machine learning models on numerical data
2. **Data Analysis**: Perform statistical analysis, identify patterns, and generate insights
3. **Predictions**: Make predictions using trained models
4. **Visualization**: Create charts and plots to visualize data and results
5. **Model Management**: Save, load, and compare different models

Guidelines:
- Always explain your analysis and reasoning
- Provide actionable insights and recommendations
- Use appropriate statistical methods and machine learning techniques
- Create visualizations to support your findings
- Be transparent about model performance and limitations
- Ask clarifying questions when requirements are unclear

Available tools:
- model_inference: Make predictions using trained models
- data_analysis: Perform statistical analysis on datasets
- visualization: Create charts and plots

Start conversations by understanding the user's data and objectives."""


class NumericsAgent:
    """
    AI Agent for numerical processing and machine learning tasks
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent = None
        self.chat_client = None
        self.current_thread = None
        self.conversation_history = []
        
        # Setup logging
        self._setup_logging()
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize agent
        asyncio.run(self._initialize_agent())
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("NumericsAgent initialized")
    
    async def _initialize_agent(self):
        """Initialize the Microsoft Agent Framework agent"""
        try:
            # Create OpenAI client for GitHub models
            openai_client = AsyncOpenAI(
                base_url="https://models.github.ai/inference",
                api_key=self.config.github_token,
            )
            
            # Create chat client
            self.chat_client = OpenAIChatClient(
                async_client=openai_client,
                model_id=self.config.model_id
            )
            
            # Create agent with tools
            self.agent = ChatAgent(
                chat_client=self.chat_client,
                name=self.config.agent_name,
                instructions=self.config.system_instructions,
                tools=self.tools
            )
            
            # Create new conversation thread
            self.current_thread = self.agent.get_new_thread()
            
            self.logger.info(f"Agent initialized with model: {self.config.model_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def _initialize_tools(self) -> List[Callable]:
        """Initialize available tools for the agent"""
        tools = []
        
        if self.config.enable_model_inference:
            tools.append(self._create_model_inference_tool())
        
        if self.config.enable_data_analysis:
            tools.append(self._create_data_analysis_tool())
        
        if self.config.enable_visualization:
            tools.append(self._create_visualization_tool())
        
        return tools
    
    def _create_model_inference_tool(self):
        """Create model inference tool"""
        def model_inference(
            model_name: str,
            input_data: str,
            data_format: str = "json"
        ) -> str:
            """
            Make predictions using a trained model.
            
            Args:
                model_name: Name of the trained model to use
                input_data: Input data in JSON or CSV format
                data_format: Format of input data ('json' or 'csv')
            
            Returns:
                Predictions as JSON string
            """
            try:
                import pandas as pd
                import json
                
                # Load the model
                trainer = ModelTrainer(config=None)  # Will use default config
                trainer.load_model(model_name)
                
                # Parse input data
                if data_format == "json":
                    data_dict = json.loads(input_data)
                    X = pd.DataFrame(data_dict)
                elif data_format == "csv":
                    from io import StringIO
                    X = pd.read_csv(StringIO(input_data))
                else:
                    return f"Error: Unsupported data format '{data_format}'"
                
                # Make predictions
                predictions = trainer.predict(X)
                
                # Return results
                results = {
                    "model_name": model_name,
                    "num_predictions": len(predictions),
                    "predictions": predictions.tolist(),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.logger.info(f"Made {len(predictions)} predictions using model {model_name}")
                return json.dumps(results, indent=2)
                
            except Exception as e:
                error_msg = f"Error in model inference: {str(e)}"
                self.logger.error(error_msg)
                return error_msg
        
        return model_inference
    
    def _create_data_analysis_tool(self):
        """Create data analysis tool"""
        def data_analysis(
            data_source: str,
            analysis_type: str = "summary",
            target_column: str = None
        ) -> str:
            """
            Perform statistical analysis on data.
            
            Args:
                data_source: Path to data file or JSON data string
                analysis_type: Type of analysis ('summary', 'correlation', 'distribution')
                target_column: Name of target column for supervised analysis
            
            Returns:
                Analysis results as JSON string
            """
            try:
                import pandas as pd
                import numpy as np
                import json
                from io import StringIO
                
                # Load data
                if data_source.startswith('{') or data_source.startswith('['):
                    # JSON data
                    data_dict = json.loads(data_source)
                    df = pd.DataFrame(data_dict)
                else:
                    # File path
                    if data_source.endswith('.csv'):
                        df = pd.read_csv(data_source)
                    elif data_source.endswith('.json'):
                        df = pd.read_json(data_source)
                    else:
                        return "Error: Unsupported file format"
                
                results = {
                    "analysis_type": analysis_type,
                    "data_shape": df.shape,
                    "timestamp": datetime.now().isoformat()
                }
                
                if analysis_type == "summary":
                    # Basic statistics
                    results["summary_statistics"] = df.describe().to_dict()
                    results["missing_values"] = df.isnull().sum().to_dict()
                    results["data_types"] = df.dtypes.astype(str).to_dict()
                
                elif analysis_type == "correlation":
                    # Correlation analysis
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr()
                        results["correlation_matrix"] = corr_matrix.to_dict()
                        
                        # Find high correlations
                        high_corr = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_val = corr_matrix.iloc[i, j]
                                if abs(corr_val) > 0.7:
                                    high_corr.append({
                                        "feature1": corr_matrix.columns[i],
                                        "feature2": corr_matrix.columns[j],
                                        "correlation": corr_val
                                    })
                        results["high_correlations"] = high_corr
                    else:
                        results["error"] = "Not enough numeric columns for correlation analysis"
                
                elif analysis_type == "distribution":
                    # Distribution analysis
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    distributions = {}
                    
                    for col in numeric_cols:
                        col_data = df[col].dropna()
                        distributions[col] = {
                            "mean": float(col_data.mean()),
                            "median": float(col_data.median()),
                            "std": float(col_data.std()),
                            "skewness": float(col_data.skew()),
                            "kurtosis": float(col_data.kurtosis()),
                            "quartiles": col_data.quantile([0.25, 0.5, 0.75]).to_dict()
                        }
                    
                    results["distributions"] = distributions
                
                self.logger.info(f"Completed {analysis_type} analysis on data with shape {df.shape}")
                return json.dumps(results, indent=2)
                
            except Exception as e:
                error_msg = f"Error in data analysis: {str(e)}"
                self.logger.error(error_msg)
                return error_msg
        
        return data_analysis
    
    def _create_visualization_tool(self):
        """Create visualization tool"""
        def visualization(
            data_source: str,
            plot_type: str = "histogram",
            x_column: str = None,
            y_column: str = None,
            save_path: str = None
        ) -> str:
            """
            Create visualizations from data.
            
            Args:
                data_source: Path to data file or JSON data string
                plot_type: Type of plot ('histogram', 'scatter', 'line', 'box', 'correlation')
                x_column: Column name for x-axis
                y_column: Column name for y-axis
                save_path: Path to save the plot (optional)
            
            Returns:
                Path to saved plot or description of the visualization
            """
            try:
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                import json
                import os
                
                # Load data
                if data_source.startswith('{') or data_source.startswith('['):
                    data_dict = json.loads(data_source)
                    df = pd.DataFrame(data_dict)
                else:
                    if data_source.endswith('.csv'):
                        df = pd.read_csv(data_source)
                    elif data_source.endswith('.json'):
                        df = pd.read_json(data_source)
                    else:
                        return "Error: Unsupported file format"
                
                # Create plot
                plt.figure(figsize=(10, 6))
                
                if plot_type == "histogram":
                    if x_column and x_column in df.columns:
                        plt.hist(df[x_column].dropna(), bins=30, alpha=0.7)
                        plt.xlabel(x_column)
                        plt.ylabel('Frequency')
                        plt.title(f'Histogram of {x_column}')
                    else:
                        return "Error: x_column required for histogram"
                
                elif plot_type == "scatter":
                    if x_column and y_column and x_column in df.columns and y_column in df.columns:
                        plt.scatter(df[x_column], df[y_column], alpha=0.6)
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                        plt.title(f'Scatter Plot: {x_column} vs {y_column}')
                    else:
                        return "Error: Both x_column and y_column required for scatter plot"
                
                elif plot_type == "line":
                    if x_column and y_column and x_column in df.columns and y_column in df.columns:
                        plt.plot(df[x_column], df[y_column])
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                        plt.title(f'Line Plot: {x_column} vs {y_column}')
                    else:
                        return "Error: Both x_column and y_column required for line plot"
                
                elif plot_type == "box":
                    if x_column and x_column in df.columns:
                        plt.boxplot(df[x_column].dropna())
                        plt.ylabel(x_column)
                        plt.title(f'Box Plot of {x_column}')
                    else:
                        return "Error: x_column required for box plot"
                
                elif plot_type == "correlation":
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr()
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                        plt.title('Correlation Matrix')
                    else:
                        return "Error: Not enough numeric columns for correlation plot"
                
                # Save plot
                if save_path is None:
                    save_path = os.path.join(self.config.results_path, f"{plot_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                result = {
                    "plot_type": plot_type,
                    "save_path": save_path,
                    "data_shape": df.shape,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.logger.info(f"Created {plot_type} plot saved to {save_path}")
                return json.dumps(result, indent=2)
                
            except Exception as e:
                error_msg = f"Error creating visualization: {str(e)}"
                self.logger.error(error_msg)
                return error_msg
        
        return visualization
    
    async def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        try:
            if self.agent is None:
                await self._initialize_agent()
            
            # Run the agent with the message
            result = await self.agent.run(message, thread=self.current_thread)
            
            # Store conversation
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "agent_response": result.text,
                "tool_calls": getattr(result, 'tool_calls', [])
            })
            
            self.logger.info(f"Processed message: {message[:50]}...")
            return result.text
            
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    async def chat_stream(self, message: str):
        """
        Send a message to the agent and get a streaming response
        
        Args:
            message: User message
            
        Yields:
            Streaming response chunks
        """
        try:
            if self.agent is None:
                await self._initialize_agent()
            
            # Stream the response
            full_response = ""
            async for chunk in self.agent.run_stream(message, thread=self.current_thread):
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text
            
            # Store conversation
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "agent_response": full_response,
                "streaming": True
            })
            
        except Exception as e:
            error_msg = f"Error in streaming chat: {str(e)}"
            self.logger.error(error_msg)
            yield error_msg
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history
    
    def reset_conversation(self):
        """Reset the conversation thread"""
        if self.agent:
            self.current_thread = self.agent.get_new_thread()
        self.conversation_history = []
        self.logger.info("Conversation reset")
    
    def save_conversation(self, filepath: str):
        """Save conversation history to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        self.logger.info(f"Conversation saved to {filepath}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available trained models"""
        model_files = []
        if os.path.exists(self.config.models_path):
            for file in os.listdir(self.config.models_path):
                if file.endswith('.joblib') and not file.endswith('_history.joblib'):
                    model_name = file.replace('.joblib', '')
                    model_files.append(model_name)
        return model_files


# Utility functions for easy agent creation
def create_numerics_agent(github_token: str = None, model_id: str = "openai/gpt-4.1-mini") -> NumericsAgent:
    """
    Create a NumericsAgent with default configuration
    
    Args:
        github_token: GitHub token for model access
        model_id: Model ID to use
        
    Returns:
        Configured NumericsAgent instance
    """
    config = AgentConfig(
        github_token=github_token,
        model_id=model_id
    )
    return NumericsAgent(config)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create agent
        agent = create_numerics_agent()
        
        # Example conversation
        response = await agent.chat("Hello! I have a dataset with numerical features and want to train a regression model. Can you help me?")
        print(f"Agent: {response}")
        
        # Example with tool usage
        response = await agent.chat("""I have this sample data in JSON format:
        {"feature_1": [1, 2, 3, 4, 5], "feature_2": [2, 4, 6, 8, 10], "target": [3, 6, 9, 12, 15]}
        
        Can you analyze this data and tell me about the correlations?""")
        print(f"Agent: {response}")
    
    # Run example
    asyncio.run(main())