import React, { useState } from 'react';

interface PredictionRequest {
  model_name: string;
  features: Record<string, number>;
  simulation_method: string;
}

interface PredictionResponse {
  model_name: string;
  prediction: number;
  prediction_proba: number;
  simulation_report: {
    original_features: number;
    simulated_features: number;
  };
  processing_time: number;
  features_used: number;
  features_simulated: number;
}

export const Chat: React.FC = () => {
  const [input, setInput] = useState('');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!input.trim()) return;

    setLoading(true);
    try {
      const requestData: PredictionRequest = {
        model_name: "demo_local_model_1",
        features: {
          income_ratio: 0.35,
          credit_score: 750,
          employment_years: 6.5,
          debt_to_income: 0.2
        },
        simulation_method: "statistical"
      };

      const response = await fetch('/models/advanced/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error('Error making prediction:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">AI Agent Chat</h1>
        <p className="mt-1 text-sm text-gray-600">
          Interact with AI models for loan risk prediction with intelligent feature simulation
        </p>
      </div>

      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="space-y-4">
            <div>
              <label htmlFor="chat-input" className="block text-sm font-medium text-gray-700">
                Ask the AI about loan risk prediction
              </label>
              <div className="mt-1">
                <textarea
                  id="chat-input"
                  rows={3}
                  className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 block w-full sm:text-sm border-gray-300 rounded-md"
                  placeholder="Ask about loan risk, credit scoring, or request a prediction..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                />
              </div>
            </div>
            <div className="flex space-x-3">
              <button
                onClick={handlePredict}
                disabled={loading}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
              >
                {loading ? '🔄 Processing...' : '🧠 Get Prediction'}
              </button>
            </div>
          </div>

          {prediction && (
            <div className="mt-6 border-t pt-6">
              <h3 className="text-lg font-medium text-gray-900">AI Response</h3>
              <div className="mt-4 bg-gray-50 rounded-lg p-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-700">Model Used</p>
                    <p className="text-lg text-gray-900">{prediction.model_name}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-700">Risk Prediction</p>
                    <p className="text-lg text-gray-900">{(prediction.prediction_proba * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-700">Processing Time</p>
                    <p className="text-lg text-gray-900">{(prediction.processing_time * 1000).toFixed(1)}ms</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-700">Features Simulated</p>
                    <p className="text-lg text-gray-900">{prediction.features_simulated}/{prediction.features_used}</p>
                  </div>
                </div>
                <div className="mt-4">
                  <p className="text-sm text-gray-600">
                    🧠 The AI automatically simulated {prediction.features_simulated} missing features 
                    using statistical methods to provide accurate predictions.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};