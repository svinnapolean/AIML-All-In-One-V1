# Install the Microsoft Agent Framework (requires --pre flag)
pip install agent-framework-azure-ai --pre

# Install Azure AI Evaluation (requires --pre flag)  
pip install azure-ai-evaluation --pre

# Install all other dependencies
pip install -r requirements-fixed.txt

# Alternative: Install everything at once
pip install -r requirements-fixed.txt agent-framework-azure-ai --pre azure-ai-evaluation --pre