# AI/ML Learning Platform - User Manual

## 📚 Table of Contents
1. [Platform Overview](#platform-overview)
2. [Dashboard Page](#dashboard-page)
3. [Test Model Page](#test-model-page)
4. [Chat with AI Page](#chat-with-ai-page)
5. [Technical Glossary](#technical-glossary)
6. [API Endpoints](#api-endpoints)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Platform Overview

The AI/ML Learning Platform is a comprehensive educational tool designed to demonstrate real-world machine learning workflows, model testing, and AI agent interactions. This platform combines:

- **Model Training & Testing**: Train, evaluate, and test machine learning models
- **AI Agent Integration**: Interact with AI agents for learning and experimentation
- **Real-time Monitoring**: Track performance metrics and system health
- **Educational Focus**: Learn ML concepts through hands-on experience

### System Architecture
- **Frontend**: React TypeScript with Tailwind CSS
- **Backend**: FastAPI with Python
- **AI Framework**: Microsoft Agent Framework
- **Database**: Redis for caching
- **Deployment**: Docker containerization

---

## 🏠 Dashboard Page

**URL**: `http://localhost:3000/`

The Dashboard serves as the central hub for monitoring and controlling your AI/ML platform.

### Features Overview

#### 1. System Status Cards
- **API Health**: Shows backend service status and uptime
- **Model Status**: Displays loaded models and their performance
- **System Metrics**: Real-time performance indicators

#### 2. Quick Actions Panel
Three primary action buttons for different workflows:

**🧪 Test Model Button**
- **Purpose**: Navigate to model testing interface
- **Action**: Opens dedicated model testing page
- **Use Case**: When you want to test trained models with custom data

**📊 View Performance Button**
- **Purpose**: Fetch and display model performance metrics
- **Action**: Retrieves metrics from all available models
- **Use Case**: Monitor model accuracy and performance over time

**💬 Chat with AI Button**
- **Purpose**: Launch AI agent interaction interface
- **Action**: Opens ChatGPT-style conversation interface
- **Use Case**: Interact with AI agents for learning and experimentation

#### 3. Navigation Tabs

**🏠 Overview Tab**
- System status overview
- Quick action buttons
- Health monitoring widgets

**📋 Action Logs Tab**
- Real-time activity tracking
- Timestamp and duration logging
- Success/error status indicators
- Detailed action descriptions

**💭 Chat Evaluations Tab**
- AI conversation quality metrics
- Response time analysis
- User satisfaction scores
- Conversation history summaries

### Key Metrics Explained

**System Health Indicators:**
- ✅ **Green**: System operating normally
- ⚠️ **Yellow**: Minor issues detected
- ❌ **Red**: Critical errors requiring attention

**Performance Metrics:**
- **Uptime**: How long the system has been running
- **Response Time**: API response latency in milliseconds
- **Memory Usage**: Current RAM consumption
- **Active Models**: Number of loaded ML models

---

## 🧪 Test Model Page

**URL**: `http://localhost:3000/test-model`

The Test Model page provides comprehensive model testing and evaluation capabilities.

### Model Selection Interface

#### Available Models Grid
Each model card displays:
- **Model Name**: Unique identifier for the model
- **Model Type**: Classification, Regression, etc.
- **Status**: Active/Inactive indicator
- **Performance Preview**: Key metrics (ROC AUC, Accuracy)
- **Creation Date**: When the model was trained
- **Description**: Brief model information

#### Selection Process
1. Click on any model card to select it
2. Selected model highlighted with blue border
3. Model details update in the testing interface

### Testing Interface

#### Test Mode Selection
**Single Test Mode**
- Test individual data points
- Input comma-separated values
- Instant prediction results
- Confidence scores included

**Batch Test Mode**
- Upload CSV files for bulk testing
- Process multiple predictions simultaneously
- Statistical summary of results
- Performance metrics calculation

#### Input Methods

**Manual Input (Single Test)**
```
Format: 1,2,3,4 or 0.5,1.2,3.7,2.1
Example: Enter feature values separated by commas
```

**File Upload (Batch Test)**
- Supported formats: CSV
- Automatic feature detection
- Progress tracking during processing

### Model Evaluation Metrics

#### Core Regression Metrics

**MSE (Mean Squared Error)**
- **Definition**: Average of squared differences between actual and predicted values
- **Formula**: MSE = (1/n) × Σ(actual - predicted)²
- **Interpretation**: Lower values indicate better model performance
- **Range**: 0 to ∞ (0 is perfect)

**RMSE (Root Mean Squared Error)**
- **Definition**: Square root of MSE
- **Formula**: RMSE = √MSE
- **Interpretation**: Same units as target variable, easier to interpret
- **Range**: 0 to ∞ (0 is perfect)

**R² (R-Squared Score)**
- **Definition**: Proportion of variance in target explained by features
- **Formula**: R² = 1 - (SS_res / SS_tot)
- **Interpretation**: Higher values indicate better model fit
- **Range**: -∞ to 1 (1 is perfect, 0 means no better than mean)

**MAE (Mean Absolute Error)**
- **Definition**: Average of absolute differences between actual and predicted
- **Formula**: MAE = (1/n) × Σ|actual - predicted|
- **Interpretation**: Average prediction error in original units
- **Range**: 0 to ∞ (0 is perfect)

#### Classification Metrics (when applicable)

**Accuracy**
- **Definition**: Percentage of correct predictions
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Range**: 0 to 1 (1 is perfect)

**Precision**
- **Definition**: True positives out of all positive predictions
- **Formula**: TP / (TP + FP)
- **Range**: 0 to 1 (1 is perfect)

**Recall (Sensitivity)**
- **Definition**: True positives out of all actual positives
- **Formula**: TP / (TP + FN)
- **Range**: 0 to 1 (1 is perfect)

**F1-Score**
- **Definition**: Harmonic mean of precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0 to 1 (1 is perfect)

### Evaluation Results Display

#### Metrics Dashboard
- **Color-coded cards**: Blue (MSE), Green (RMSE), Purple (R²), Orange (MAE)
- **Large numeric display**: Primary metric value
- **Descriptive labels**: Full metric names and abbreviations
- **Tooltips**: Brief explanations of each metric

#### Performance Summary
- **Sample Count**: Number of data points evaluated
- **Prediction Accuracy**: Overall model performance percentage
- **Variance Explained**: How much of the data pattern the model captures

#### Recommendations Panel
- **Performance Assessment**: Automatic model quality evaluation
- **Optimization Suggestions**: Specific improvement recommendations
- **Outlier Detection**: Identification of unusual predictions

### Test Results History

#### Individual Test Records
Each test result includes:
- **Timestamp**: When the test was performed
- **Model Used**: Which model was tested
- **Input Data**: Features provided for prediction
- **Prediction**: Model output
- **Processing Time**: How long the prediction took
- **Confidence Score**: Model's certainty in the prediction

#### Batch Test Summaries
- **File Information**: Uploaded file details
- **Prediction Count**: Number of predictions made
- **Aggregate Metrics**: Overall performance statistics
- **Sample Predictions**: Preview of results

---

## 💬 Chat with AI Page

**URL**: `http://localhost:3000/chat`

The Chat interface provides a ChatGPT-style conversation experience with AI agents.

### Interface Layout

#### Left Sidebar
**Chat History**
- **Session List**: All previous conversations
- **Session Names**: Auto-generated or custom titles
- **Timestamps**: When each conversation started
- **Active Indicator**: Currently selected conversation

**New Chat Button**
- **Purpose**: Start fresh conversation
- **Action**: Creates new session with clean history
- **Shortcut**: Prominent placement for easy access

#### Main Chat Area

**Header Controls**
- **Model Selection Dropdown**: Choose AI model for responses
- **Session Title**: Current conversation name
- **Settings**: Configuration options

**Message Display**
- **User Messages**: Right-aligned with blue background
- **AI Responses**: Left-aligned with white background
- **Timestamps**: When each message was sent
- **Message Status**: Delivery and processing indicators

### Input Methods

#### Text Input Mode
- **Purpose**: Standard text-based conversation
- **Features**: Multi-line support, emoji, formatting
- **Use Cases**: General questions, explanations, help requests

#### Test Data Mode
- **Purpose**: Send structured data for AI analysis
- **Format**: JSON, CSV, or formatted text
- **Use Cases**: Data analysis requests, model inputs

#### File Upload Mode
- **Purpose**: Send documents or datasets to AI
- **Supported Formats**: CSV, TXT, JSON
- **Use Cases**: Document analysis, data processing

### AI Model Selection

#### Available Models
- **GPT-4**: Most capable model for complex reasoning
- **GPT-3.5**: Faster responses for simpler queries
- **Claude**: Alternative model with different strengths
- **Custom Models**: Domain-specific trained models

#### Model Characteristics
- **Response Quality**: Accuracy and helpfulness
- **Processing Speed**: Time to generate responses
- **Token Limits**: Maximum conversation length
- **Specialized Capabilities**: Code, math, analysis

### Conversation Features

#### Message Actions
- **Copy**: Copy AI responses to clipboard
- **Regenerate**: Request new response to same query
- **Edit**: Modify your messages
- **Delete**: Remove messages from history

#### Conversation Management
- **Save**: Preserve important conversations
- **Export**: Download conversation history
- **Share**: Generate shareable links
- **Archive**: Store old conversations

---

## 📖 Technical Glossary

### Machine Learning Terms

**Algorithm**: Mathematical procedure for finding patterns in data
**Feature**: Individual measurable property of observed phenomena
**Target Variable**: The outcome you're trying to predict
**Training Data**: Dataset used to teach the model
**Test Data**: Dataset used to evaluate model performance
**Overfitting**: Model learns training data too specifically, poor generalization
**Underfitting**: Model is too simple to capture underlying patterns
**Cross-Validation**: Technique to assess model generalization
**Hyperparameter**: Configuration setting for learning algorithm
**Bias**: Error from overly simplistic assumptions
**Variance**: Error from sensitivity to small fluctuations

### Evaluation Metrics

**True Positive (TP)**: Correctly predicted positive cases
**True Negative (TN)**: Correctly predicted negative cases
**False Positive (FP)**: Incorrectly predicted positive (Type I error)
**False Negative (FN)**: Incorrectly predicted negative (Type II error)
**Confusion Matrix**: Table showing prediction accuracy breakdown
**ROC Curve**: Plot of true positive rate vs false positive rate
**AUC**: Area Under Curve, measure of classification performance

### Technical Architecture

**API**: Application Programming Interface for system communication
**REST**: Representational State Transfer, web service architecture
**JSON**: JavaScript Object Notation, data interchange format
**WebSocket**: Protocol for real-time bidirectional communication
**Microservices**: Architecture pattern with small, independent services
**Container**: Lightweight, portable software package
**Load Balancer**: Distributes incoming requests across multiple servers

---

## 🔌 API Endpoints

### Model Management
- `GET /models` - List all available models
- `POST /models/train` - Train new model
- `POST /models/{model_name}/predict` - Single prediction
- `POST /models/{model_name}/predict-batch` - Batch predictions
- `POST /models/evaluate/{model_name}` - Evaluate model performance
- `GET /models/{model_name}/metrics` - Get cached model metrics

### Health & Status
- `GET /health` - System health check
- `GET /info` - Platform information

### Agent Integration
- `POST /agent/chat` - Send message to AI agent
- `GET /agent/sessions` - List chat sessions
- `POST /agent/sessions` - Create new chat session

### Data Management
- `POST /data/upload` - Upload training data
- `GET /data/status` - Data processing status
- `POST /data/preprocess` - Data preprocessing

---

## 🛠 Troubleshooting

### Common Issues

#### Model Loading Errors
**Symptom**: "Model not found" error
**Solution**: 
1. Check if model file exists in `models/trained_models/`
2. Verify model name matches exactly
3. Ensure model was trained successfully

#### API Connection Issues
**Symptom**: "Failed to fetch" errors
**Solution**:
1. Verify Docker containers are running
2. Check if backend is accessible at `localhost:8000`
3. Examine browser network tab for specific errors

#### File Upload Problems
**Symptom**: Upload fails or hangs
**Solution**:
1. Check file format (CSV, Excel supported)
2. Verify file size < 10MB
3. Ensure proper column structure

#### Chat Interface Issues
**Symptom**: Messages not sending or AI not responding
**Solution**:
1. Check AI agent service status
2. Verify model selection
3. Clear browser cache and reload

### Performance Optimization

#### Slow Predictions
- Use smaller datasets for batch predictions
- Consider model complexity vs speed tradeoffs
- Monitor system resource usage

#### High Memory Usage
- Limit concurrent model loading
- Clear old conversation histories
- Restart containers if needed

#### Network Timeouts
- Increase timeout settings for large files
- Use chunked uploads for big datasets
- Check network connectivity

### Getting Help

#### Log Files
- Frontend: Browser developer console
- Backend: `logs/api.log`
- Containers: `docker logs [container_name]`

#### Debug Mode
Enable debug logging by setting environment variables:
```bash
LOG_LEVEL=DEBUG
PYTHONPATH=/app/src
```

#### Support Resources
- Platform documentation: `/docs`
- API documentation: `/docs` (interactive)
- GitHub Issues: Repository issue tracker
- Community Forum: Platform discussions

---

## 📊 Usage Examples

### Example 1: Testing a Regression Model
1. Navigate to Test Model page
2. Select "sample-regressor" model
3. Enter test data: `5.2,3.1,4.8,2.7`
4. Click "Test Model"
5. Review prediction and confidence score
6. Click "Run Evaluation" for detailed metrics

### Example 2: Batch Testing with CSV
1. Prepare CSV file with features in columns
2. Select "Batch Test" mode
3. Upload your CSV file
4. Click "Batch Test"
5. Review aggregated results and metrics

### Example 3: AI Agent Conversation
1. Go to Chat page
2. Select appropriate AI model
3. Type question: "Explain what R-squared means"
4. Review AI explanation
5. Follow up with specific questions

---

## 🔄 Platform Updates

### Version History
- **v1.0**: Initial release with basic model testing
- **v1.1**: Added chat interface and AI agent integration
- **v1.2**: Enhanced evaluation metrics and batch processing
- **v1.3**: Improved UI/UX and documentation

### Upcoming Features
- Advanced model comparison tools
- Automated hyperparameter tuning
- Extended AI model support
- Real-time collaboration features

---

*This manual is generated automatically and updated with each platform release. For the latest version, check the repository documentation.*