import React from 'react';

export const Models: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">ML Models</h1>
        <p className="mt-1 text-sm text-gray-600">
          Manage and explore your machine learning models
        </p>
      </div>

      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Available Models</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border rounded-lg p-4">
            <h3 className="font-medium">🧠 Advanced Autoencoder+Classifier</h3>
            <p className="text-sm text-gray-600">Deep learning model with feature reconstruction</p>
            <p className="text-sm font-medium text-green-600">AUC: 0.855</p>
          </div>
          <div className="border rounded-lg p-4">
            <h3 className="font-medium">🚀 LightGBM Turbo</h3>
            <p className="text-sm text-gray-600">Optimized gradient boosting model</p>
            <p className="text-sm font-medium text-green-600">AUC: 0.936</p>
          </div>
          <div className="border rounded-lg p-4">
            <h3 className="font-medium">⚡ XGBoost Ultra-Fast</h3>
            <p className="text-sm text-gray-600">0.067s training time</p>
            <p className="text-sm font-medium text-green-600">AUC: 0.966</p>
          </div>
          <div className="border rounded-lg p-4">
            <h3 className="font-medium">📊 Demo Models</h3>
            <p className="text-sm text-gray-600">Pre-trained demonstration models</p>
            <p className="text-sm font-medium text-green-600">AUC: 0.950-0.970</p>
          </div>
        </div>
      </div>
    </div>
  );
};