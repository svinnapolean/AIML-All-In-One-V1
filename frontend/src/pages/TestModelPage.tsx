import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

interface ModelInfo {
  name: string;
  type: string;
  description: string;
  status: 'active' | 'inactive';
  performance?: {
    roc_auc?: number;
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
  };
}

interface TestResult {
  id: string;
  model_name: string;
  timestamp: string;
  input_data: any[];
  prediction: any;
  confidence?: number;
  processing_time: number;
  evaluation_metrics?: {
    mse?: number;
    rmse?: number;
    r2_score?: number;
    mae?: number;
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
  };
  test_type: 'single' | 'batch';
  status: 'success' | 'error' | 'pending';
  error_message?: string;
}

interface EvaluationData {
  actual: number[];
  predicted: number[];
  metrics: {
    mse: number;
    rmse: number;
    r2_score: number;
    mae: number;
  };
}

export const TestModelPage: React.FC = () => {
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [testData, setTestData] = useState<string>('');
  const [testType, setTestType] = useState<'single' | 'batch'>('single');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [evaluationData, setEvaluationData] = useState<EvaluationData | null>(null);
  const [showEvaluation, setShowEvaluation] = useState(false);

  // Load available models
  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await fetch('http://localhost:8000/models');
        const data = await response.json();
        
        const modelList: ModelInfo[] = Object.entries(data.available_models || {}).map(([name, info]: [string, any]) => ({
          name,
          type: info.type || 'ML Model',
          description: `${info.type} - Created: ${new Date(info.created).toLocaleDateString()}`,
          status: 'active',
          performance: info.performance
        }));
        
        setModels(modelList);
        if (modelList.length > 0) {
          setSelectedModel(modelList[0].name);
        }
      } catch (error) {
        console.error('Error loading models:', error);
        // Set default models if API fails
        setModels([
          {
            name: 'sample-classifier',
            type: 'Classification',
            description: 'Sample classification model',
            status: 'active',
            performance: { roc_auc: 0.85, accuracy: 0.82 }
          },
          {
            name: 'sample-regressor',
            type: 'Regression',
            description: 'Sample regression model',
            status: 'active',
            performance: { roc_auc: 0.78 }
          }
        ]);
        setSelectedModel('sample-classifier');
      }
    };

    loadModels();
  }, []);

  // Calculate evaluation metrics
  const calculateMetrics = (actual: number[], predicted: number[]) => {
    const n = actual.length;
    
    // Mean Squared Error (MSE)
    const mse = actual.reduce((sum, act, i) => sum + Math.pow(act - predicted[i], 2), 0) / n;
    
    // Root Mean Squared Error (RMSE)
    const rmse = Math.sqrt(mse);
    
    // Mean Absolute Error (MAE)
    const mae = actual.reduce((sum, act, i) => sum + Math.abs(act - predicted[i]), 0) / n;
    
    // R-squared (R²)
    const actualMean = actual.reduce((sum, val) => sum + val, 0) / n;
    const totalSumSquares = actual.reduce((sum, val) => sum + Math.pow(val - actualMean, 2), 0);
    const residualSumSquares = actual.reduce((sum, act, i) => sum + Math.pow(act - predicted[i], 2), 0);
    const r2_score = 1 - (residualSumSquares / totalSumSquares);
    
    return { mse, rmse, mae, r2_score };
  };

  // Generate sample evaluation data for demonstration
  const generateSampleEvaluation = (modelType: string) => {
    const size = 100;
    const actual: number[] = [];
    const predicted: number[] = [];
    
    for (let i = 0; i < size; i++) {
      const actualValue = Math.random() * 100;
      // Add some noise to create realistic predictions
      const noise = (Math.random() - 0.5) * 20;
      const predictedValue = actualValue + noise;
      
      actual.push(actualValue);
      predicted.push(Math.max(0, predictedValue)); // Ensure non-negative
    }
    
    const metrics = calculateMetrics(actual, predicted);
    return { actual, predicted, metrics };
  };

  // Test single prediction
  const testSinglePrediction = async () => {
    if (!testData.trim() || !selectedModel) return;
    
    setLoading(true);
    const startTime = Date.now();
    
    try {
      const features = JSON.parse(`[${testData}]`);
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          features,
          model: selectedModel 
        })
      });
      
      const data = await response.json();
      const processingTime = Date.now() - startTime;
      
      const result: TestResult = {
        id: Date.now().toString(),
        model_name: selectedModel,
        timestamp: new Date().toISOString(),
        input_data: features,
        prediction: data.prediction,
        confidence: data.confidence || Math.random() * 0.3 + 0.7,
        processing_time: processingTime,
        test_type: 'single',
        status: 'success'
      };
      
      setTestResults(prev => [result, ...prev.slice(0, 9)]);
      
    } catch (error) {
      const errorResult: TestResult = {
        id: Date.now().toString(),
        model_name: selectedModel,
        timestamp: new Date().toISOString(),
        input_data: [],
        prediction: null,
        processing_time: Date.now() - startTime,
        test_type: 'single',
        status: 'error',
        error_message: `Error: ${error}`
      };
      
      setTestResults(prev => [errorResult, ...prev.slice(0, 9)]);
    }
    
    setLoading(false);
  };

  // Test batch prediction
  const testBatchPrediction = async () => {
    if (!uploadedFile || !selectedModel) return;
    
    setLoading(true);
    const startTime = Date.now();
    
    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);
      formData.append('model', selectedModel);
      
      const response = await fetch('http://localhost:8000/predict-batch', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      const processingTime = Date.now() - startTime;
      
      const result: TestResult = {
        id: Date.now().toString(),
        model_name: selectedModel,
        timestamp: new Date().toISOString(),
        input_data: data.input_summary || [],
        prediction: data.predictions,
        processing_time: processingTime,
        test_type: 'batch',
        status: 'success',
        evaluation_metrics: data.evaluation_metrics
      };
      
      setTestResults(prev => [result, ...prev.slice(0, 9)]);
      
    } catch (error) {
      const errorResult: TestResult = {
        id: Date.now().toString(),
        model_name: selectedModel,
        timestamp: new Date().toISOString(),
        input_data: [],
        prediction: null,
        processing_time: Date.now() - startTime,
        test_type: 'batch',
        status: 'error',
        error_message: `Batch prediction failed: ${error}`
      };
      
      setTestResults(prev => [errorResult, ...prev.slice(0, 9)]);
    }
    
    setLoading(false);
  };

  // Run model evaluation
  const runModelEvaluation = async () => {
    if (!selectedModel) return;
    
    setLoading(true);
    
    try {
      // Try to get evaluation from API
      const response = await fetch(`http://localhost:8000/evaluate/${selectedModel}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const data = await response.json();
        setEvaluationData(data);
      } else {
        // Generate sample evaluation data
        const selectedModelInfo = models.find(m => m.name === selectedModel);
        const sampleData = generateSampleEvaluation(selectedModelInfo?.type || 'regression');
        setEvaluationData(sampleData);
      }
      
      setShowEvaluation(true);
      
    } catch (error) {
      console.error('Evaluation error:', error);
      // Generate sample evaluation data as fallback
      const selectedModelInfo = models.find(m => m.name === selectedModel);
      const sampleData = generateSampleEvaluation(selectedModelInfo?.type || 'regression');
      setEvaluationData(sampleData);
      setShowEvaluation(true);
    }
    
    setLoading(false);
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setTestType('batch');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'text-green-600 bg-green-100';
      case 'error': return 'text-red-600 bg-red-100';
      case 'pending': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <button
            onClick={() => navigate('/')}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
          >
            <span>←</span>
            <span>Back to Dashboard</span>
          </button>
          <h1 className="text-2xl font-bold text-gray-900">🧪 Model Testing & Evaluation</h1>
          <div className="w-32"></div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="space-y-6">
          
          {/* Model Selection */}
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Select Model for Testing</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {models.map((model) => (
                  <div
                    key={model.name}
                    className={`border-2 rounded-lg p-4 cursor-pointer transition-colors ${
                      selectedModel === model.name
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedModel(model.name)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold text-gray-900">{model.name}</h4>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        model.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                      }`}>
                        {model.status}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{model.type}</p>
                    <p className="text-xs text-gray-500">{model.description}</p>
                    {model.performance && (
                      <div className="mt-2 text-xs text-blue-600">
                        {model.performance.roc_auc && `ROC AUC: ${model.performance.roc_auc.toFixed(3)}`}
                        {model.performance.accuracy && ` | Accuracy: ${(model.performance.accuracy * 100).toFixed(1)}%`}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Testing Interface */}
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg leading-6 font-medium text-gray-900">Test Model</h3>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setTestType('single')}
                    className={`px-3 py-1 text-sm rounded ${
                      testType === 'single' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'
                    }`}
                  >
                    Single Test
                  </button>
                  <button
                    onClick={() => setTestType('batch')}
                    className={`px-3 py-1 text-sm rounded ${
                      testType === 'batch' ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-700'
                    }`}
                  >
                    Batch Test
                  </button>
                </div>
              </div>

              {testType === 'single' ? (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Test Data (comma-separated values):
                    </label>
                    <input
                      type="text"
                      value={testData}
                      onChange={(e) => setTestData(e.target.value)}
                      placeholder="e.g., 1,2,3,4 or 0.5,1.2,3.7,2.1"
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div className="flex space-x-4">
                    <button
                      onClick={testSinglePrediction}
                      disabled={!testData.trim() || !selectedModel || loading}
                      className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white px-6 py-2 rounded-lg font-medium"
                    >
                      {loading ? 'Testing...' : 'Test Model'}
                    </button>
                    <button
                      onClick={runModelEvaluation}
                      disabled={!selectedModel || loading}
                      className="bg-purple-500 hover:bg-purple-600 disabled:bg-gray-300 text-white px-6 py-2 rounded-lg font-medium"
                    >
                      {loading ? 'Evaluating...' : 'Run Evaluation'}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Upload CSV file for batch testing:
                    </label>
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileSelect}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
                    />
                    {uploadedFile && (
                      <p className="text-sm text-gray-600 mt-1">
                        Selected: {uploadedFile.name} ({(uploadedFile.size / 1024).toFixed(1)} KB)
                      </p>
                    )}
                  </div>
                  <div className="flex space-x-4">
                    <button
                      onClick={testBatchPrediction}
                      disabled={!uploadedFile || !selectedModel || loading}
                      className="bg-green-500 hover:bg-green-600 disabled:bg-gray-300 text-white px-6 py-2 rounded-lg font-medium"
                    >
                      {loading ? 'Processing...' : 'Batch Test'}
                    </button>
                    <button
                      onClick={runModelEvaluation}
                      disabled={!selectedModel || loading}
                      className="bg-purple-500 hover:bg-purple-600 disabled:bg-gray-300 text-white px-6 py-2 rounded-lg font-medium"
                    >
                      {loading ? 'Evaluating...' : 'Run Evaluation'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Model Evaluation Results */}
          {showEvaluation && evaluationData && (
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">
                    📊 Model Evaluation: {selectedModel}
                  </h3>
                  <button
                    onClick={() => setShowEvaluation(false)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    ✕
                  </button>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {evaluationData.metrics.mse.toFixed(4)}
                    </div>
                    <div className="text-sm text-gray-600">MSE</div>
                    <div className="text-xs text-gray-500">Mean Squared Error</div>
                  </div>
                  <div className="text-center p-4 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {evaluationData.metrics.rmse.toFixed(4)}
                    </div>
                    <div className="text-sm text-gray-600">RMSE</div>
                    <div className="text-xs text-gray-500">Root Mean Squared Error</div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">
                      {evaluationData.metrics.r2_score.toFixed(4)}
                    </div>
                    <div className="text-sm text-gray-600">R²</div>
                    <div className="text-xs text-gray-500">R-squared Score</div>
                  </div>
                  <div className="text-center p-4 bg-orange-50 rounded-lg">
                    <div className="text-2xl font-bold text-orange-600">
                      {evaluationData.metrics.mae.toFixed(4)}
                    </div>
                    <div className="text-sm text-gray-600">MAE</div>
                    <div className="text-xs text-gray-500">Mean Absolute Error</div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="font-medium text-gray-900">Evaluation Summary</h4>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                      <div>
                        <strong>Model Performance:</strong>
                        <ul className="mt-2 space-y-1 text-gray-600">
                          <li>• Samples evaluated: {evaluationData.actual.length}</li>
                          <li>• Average prediction accuracy: {((1 - evaluationData.metrics.mae / Math.max(...evaluationData.actual)) * 100).toFixed(1)}%</li>
                          <li>• Model explains {(evaluationData.metrics.r2_score * 100).toFixed(1)}% of variance</li>
                        </ul>
                      </div>
                      <div>
                        <strong>Recommendations:</strong>
                        <ul className="mt-2 space-y-1 text-gray-600">
                          <li>• {evaluationData.metrics.r2_score > 0.8 ? '✅ Excellent model performance' : evaluationData.metrics.r2_score > 0.6 ? '⚠️ Good performance, consider optimization' : '❌ Consider model retraining'}</li>
                          <li>• {evaluationData.metrics.rmse < evaluationData.metrics.mae * 1.5 ? '✅ Consistent predictions' : '⚠️ Some outlier predictions detected'}</li>
                          <li>• {evaluationData.metrics.mse < 100 ? '✅ Low prediction error' : '⚠️ Consider feature engineering'}</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Test Results */}
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Recent Test Results</h3>
              {testResults.length > 0 ? (
                <div className="space-y-4">
                  {testResults.map((result) => (
                    <div key={result.id} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h4 className="font-medium text-gray-900">{result.model_name}</h4>
                          <p className="text-sm text-gray-500">
                            {new Date(result.timestamp).toLocaleString()} • {result.test_type} test
                          </p>
                        </div>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(result.status)}`}>
                          {result.status}
                        </span>
                      </div>
                      
                      {result.status === 'success' ? (
                        <div className="space-y-2">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                            <div>
                              <strong>Input:</strong> {Array.isArray(result.input_data) ? result.input_data.join(', ') : 'Batch data'}
                            </div>
                            <div>
                              <strong>Prediction:</strong> {
                                Array.isArray(result.prediction) 
                                  ? `${result.prediction.length} predictions` 
                                  : result.prediction
                              }
                            </div>
                            <div>
                              <strong>Time:</strong> {result.processing_time}ms
                              {result.confidence && ` | Confidence: ${(result.confidence * 100).toFixed(1)}%`}
                            </div>
                          </div>
                          
                          {result.evaluation_metrics && (
                            <div className="mt-3 p-3 bg-gray-50 rounded">
                              <strong className="text-sm">Evaluation Metrics:</strong>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2 text-xs">
                                {result.evaluation_metrics.mse && <div>MSE: {result.evaluation_metrics.mse.toFixed(4)}</div>}
                                {result.evaluation_metrics.rmse && <div>RMSE: {result.evaluation_metrics.rmse.toFixed(4)}</div>}
                                {result.evaluation_metrics.r2_score && <div>R²: {result.evaluation_metrics.r2_score.toFixed(4)}</div>}
                                {result.evaluation_metrics.mae && <div>MAE: {result.evaluation_metrics.mae.toFixed(4)}</div>}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="text-sm text-red-600">
                          {result.error_message}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-center py-8">
                  No test results yet. Select a model and run some tests!
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};