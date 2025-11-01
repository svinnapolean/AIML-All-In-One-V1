# Quick Start Guide

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 16+ (for frontend)
- GitHub Token (for AI agent)

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

### Key Features
- ✅ ML Model Training & Prediction
- ✅ AI Agent Chat Interface
- ✅ Data Upload & Processing
- ✅ Azure AI Evaluation
- ✅ Docker Deployment
- ✅ React Frontend

### Troubleshooting
- Run `python test_imports.py` to verify all components
- Check the logs in the `logs/` directory
- Ensure all dependencies are installed with the correct versions