# AI/ML Learning Platform - Quick Start Guide

## 🎯 Learning Objectives

This platform teaches you how to:
- Build complete ML pipelines from training to deployment
- Create AI agents using Microsoft Agent Framework
- Deploy ML models via public APIs for consumption
- Evaluate AI systems using Azure ADK
- Implement CI/CD practices for ML applications
- Containerize and scale ML applications

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 16+ (for frontend)
- GitHub Token (for AI agent)
- Docker (optional, for deployment)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/svinnapolean/AIML-All-In-One-V1.git
   cd AIML-All-In-One-V1
   ```

2. **Install Python dependencies:**
   ```bash
   # Install core dependencies
   pip install -r requirements-fixed.txt
   
   # Install AI frameworks (requires --pre flag)
   pip install agent-framework-azure-ai --pre
   pip install azure-ai-evaluation --pre
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file
   echo "GITHUB_TOKEN=your_github_token_here" > .env
   echo "MODEL_ID=openai/gpt-4.1-mini" >> .env
   echo "API_KEY=your-secure-api-key" >> .env
   ```

4. **Test the installation:**
   ```bash
   python test_imports.py
   ```

5. **Start the API server:**
   ```bash
   python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Start the frontend (optional):**
   ```bash
   cd frontend
   npm install
   npm start
   ```

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

### 🎓 Learning Path Recommendations

#### Beginner Level:
1. Start with **Model Training** (`/models/*` endpoints)
2. Learn **Data Upload** and processing (`/data/*` endpoints)  
3. Explore **API Documentation** at `http://localhost:8000/docs`

#### Intermediate Level:
1. Build **AI Agents** using the chat interface (`/agent/*` endpoints)
2. Implement **Custom Evaluators** with Azure ADK
3. Deploy using **Docker** containers

#### Advanced Level:
1. Set up **CI/CD pipelines** for model automation
2. Scale with **Load Balancing** and monitoring
3. Customize **Agent Tools** and capabilities

### Key Features - What You'll Learn
- ✅ **ML Model Training & Prediction** - Complete pipeline development
- ✅ **AI Agent Chat Interface** - Microsoft Agent Framework integration
- ✅ **Data Upload & Processing** - Production data handling
- ✅ **Azure AI Evaluation** - Comprehensive model assessment
- ✅ **Docker Deployment** - Containerized application deployment
- ✅ **React Frontend** - Modern UI development for ML applications
- ✅ **Public API Development** - RESTful APIs for ML model serving
- ✅ **CI/CD for ML** - DevOps practices for machine learning

### Troubleshooting
- Run `python test_imports.py` to verify all components
- Check the logs in the `logs/` directory
- Ensure all dependencies are installed with the correct versions