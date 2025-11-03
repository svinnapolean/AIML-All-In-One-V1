import { getApiBaseUrl } from '../config/apiConfig';

class APIClient {
  private baseUrl: string;
  
  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl || getApiBaseUrl();
  }

  async healthCheck() {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return await response.json();
    } catch (error) {
      throw new Error(`Health check failed: ${error}`);
    }
  }

  async getModels() {
    try {
      const response = await fetch(`${this.baseUrl}/models`);
      return await response.json();
    } catch (error) {
      throw new Error(`Failed to fetch models: ${error}`);
    }
  }

  async predict(features: number[], modelName?: string) {
    try {
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          features,
          model: modelName
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      throw new Error(`Prediction request failed: ${error}`);
    }
  }

  async predictBatch(file: File, modelName?: string) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (modelName) {
        formData.append('model', modelName);
      }

      const response = await fetch(`${this.baseUrl}/predict-batch`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Batch prediction failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      throw new Error(`Batch prediction request failed: ${error}`);
    }
  }

  async evaluateModel(modelName: string, testData?: any) {
    try {
      const response = await fetch(`${this.baseUrl}/evaluate/${modelName}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          test_data: testData
        }),
      });

      if (!response.ok) {
        throw new Error(`Model evaluation failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      throw new Error(`Model evaluation request failed: ${error}`);
    }
  }

  async getModelMetrics(modelName: string) {
    try {
      const response = await fetch(`${this.baseUrl}/models/${modelName}/metrics`);
      
      if (!response.ok) {
        throw new Error(`Failed to get model metrics: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      throw new Error(`Model metrics request failed: ${error}`);
    }
  }

  async trainModel(modelConfig: any) {
    try {
      const response = await fetch(`${this.baseUrl}/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(modelConfig),
      });

      if (!response.ok) {
        throw new Error(`Model training failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      throw new Error(`Model training request failed: ${error}`);
    }
  }

  // Utility method to calculate evaluation metrics locally
  static calculateMetrics(actual: number[], predicted: number[]) {
    const n = actual.length;
    
    if (n !== predicted.length) {
      throw new Error('Actual and predicted arrays must have the same length');
    }

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
    
    // Mean Absolute Percentage Error (MAPE)
    const mape = actual.reduce((sum, act, i) => {
      if (act !== 0) {
        return sum + Math.abs((act - predicted[i]) / act);
      }
      return sum;
    }, 0) / n * 100;

    // For classification metrics (if values are binary/categorical)
    const isClassification = actual.every(val => val === 0 || val === 1) && 
                             predicted.every(val => val >= 0 && val <= 1);
    
    let classificationMetrics = {};
    
    if (isClassification) {
      // Convert predictions to binary (threshold at 0.5)
      const binaryPredictions = predicted.map(p => p >= 0.5 ? 1 : 0);
      
      let tp = 0, tn = 0, fp = 0, fn = 0;
      
      for (let i = 0; i < n; i++) {
        if (actual[i] === 1 && binaryPredictions[i] === 1) tp++;
        else if (actual[i] === 0 && binaryPredictions[i] === 0) tn++;
        else if (actual[i] === 0 && binaryPredictions[i] === 1) fp++;
        else if (actual[i] === 1 && binaryPredictions[i] === 0) fn++;
      }
      
      const accuracy = (tp + tn) / n;
      const precision = tp / (tp + fp) || 0;
      const recall = tp / (tp + fn) || 0;
      const f1_score = 2 * (precision * recall) / (precision + recall) || 0;
      
      classificationMetrics = {
        accuracy,
        precision,
        recall,
        f1_score,
        confusion_matrix: { tp, tn, fp, fn }
      };
    }

    return {
      mse: Number(mse.toFixed(6)),
      rmse: Number(rmse.toFixed(6)),
      mae: Number(mae.toFixed(6)),
      r2_score: Number(r2_score.toFixed(6)),
      mape: Number(mape.toFixed(2)),
      ...classificationMetrics
    };
  }

  // Generate sample data for testing
  static generateSampleData(size: number = 100, type: 'regression' | 'classification' = 'regression') {
    const data = [];
    
    for (let i = 0; i < size; i++) {
      if (type === 'regression') {
        // Generate regression data with some pattern
        const x = Math.random() * 100;
        const y = 2 * x + 10 + (Math.random() - 0.5) * 20; // Linear with noise
        data.push({ features: [x], target: y });
      } else {
        // Generate classification data
        const x1 = Math.random() * 10;
        const x2 = Math.random() * 10;
        const y = x1 + x2 > 10 ? 1 : 0; // Simple decision boundary
        data.push({ features: [x1, x2], target: y });
      }
    }
    
    return data;
  }
}

export default APIClient;