import React from 'react';

export const Settings: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="mt-1 text-sm text-gray-600">
          Configure your AI/ML platform settings
        </p>
      </div>

      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">API Configuration</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">API Endpoint</label>
            <input
              type="text"
              value="http://localhost:8000"
              readOnly
              className="mt-1 block w-full border-gray-300 rounded-md shadow-sm bg-gray-50"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Default Model</label>
            <select className="mt-1 block w-full border-gray-300 rounded-md shadow-sm">
              <option>demo_local_model_1</option>
              <option>demo_local_model_2</option>
              <option>advanced_autoencoder_classifier</option>
              <option>advanced_lightgbm_classifier</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
};