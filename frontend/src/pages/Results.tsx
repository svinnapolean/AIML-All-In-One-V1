import React from 'react';

export const Results: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Results & Analysis</h1>
        <p className="mt-1 text-sm text-gray-600">
          View model performance and prediction results
        </p>
      </div>

      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Performance Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">26.7</div>
            <div className="text-sm text-blue-600">req/s</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">37ms</div>
            <div className="text-sm text-green-600">avg response</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">100%</div>
            <div className="text-sm text-purple-600">success rate</div>
          </div>
          <div className="text-center p-4 bg-orange-50 rounded-lg">
            <div className="text-2xl font-bold text-orange-600">0.966</div>
            <div className="text-sm text-orange-600">best AUC</div>
          </div>
        </div>
      </div>
    </div>
  );
};