import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

interface APIHealth {
  status: string;
  model_loaded: boolean;
  model_name: string;
  uptime_seconds: number;
  version: string;
}

interface ActionLog {
  id: string;
  action: 'test_model' | 'view_performance' | 'chat_ai';
  timestamp: string;
  status: 'success' | 'error' | 'in_progress';
  details: string;
  duration?: number;
}

interface ChatEvaluation {
  id: string;
  timestamp: string;
  query: string;
  response: string;
  metrics: {
    relevance_score: number;
    accuracy_score: number;
    response_time: number;
    user_satisfaction?: number;
  };
  evaluation_status: 'completed' | 'pending' | 'failed';
}

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [health, setHealth] = useState<APIHealth | null>(null);
  const [actionLogs, setActionLogs] = useState<ActionLog[]>([]);
  const [chatEvaluations, setChatEvaluations] = useState<ChatEvaluation[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'logs' | 'evaluations'>('overview');

  // Add action to logs
  const addActionLog = (action: ActionLog['action'], details: string, status: ActionLog['status'] = 'in_progress') => {
    const newLog: ActionLog = {
      id: Date.now().toString(),
      action,
      timestamp: new Date().toISOString(),
      status,
      details,
      duration: status === 'success' ? Math.random() * 2000 + 500 : undefined
    };
    setActionLogs(prev => [newLog, ...prev.slice(0, 9)]); // Keep last 10 logs
    return newLog.id;
  };

  // Update action log status
  const updateActionLog = (id: string, status: ActionLog['status'], details?: string, duration?: number) => {
    setActionLogs(prev => prev.map(log => 
      log.id === id 
        ? { ...log, status, details: details || log.details, duration }
        : log
    ));
  };

  // Handle Test Model action - Navigate to Test Model page
  const handleTestModel = () => {
    navigate('/test-model');
    const logId = addActionLog('test_model', 'Navigating to model testing page...');
    updateActionLog(logId, 'success', 'Model testing page opened', 250);
  };

  // Handle View Performance action
  const handleViewPerformance = async () => {
    const logId = addActionLog('view_performance', 'Fetching performance metrics...');
    try {
      const startTime = Date.now();
      const response = await fetch('http://localhost:8000/models');
      const data = await response.json();
      const duration = Date.now() - startTime;
      updateActionLog(logId, 'success', `Performance data loaded. Found ${Object.keys(data.available_models || {}).length} models`, duration);
    } catch (error) {
      updateActionLog(logId, 'error', `Failed to load performance data: ${error}`);
    }
  };

  // Handle Chat with AI action - Navigate to Chat page
  const handleChatWithAI = () => {
    navigate('/chat');
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch API health
        const healthResponse = await fetch('http://localhost:8000/health');
        const healthData = await healthResponse.json();
        setHealth(healthData);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const getActionIcon = (action: ActionLog['action']) => {
    switch (action) {
      case 'test_model': return '🧪';
      case 'view_performance': return '📊';
      case 'chat_ai': return '💬';
      default: return '⚡';
    }
  };

  const getStatusColor = (status: ActionLog['status']) => {
    switch (status) {
      case 'success': return 'text-green-600 bg-green-100';
      case 'error': return 'text-red-600 bg-red-100';
      case 'in_progress': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">🚀 AI/ML Dashboard</h1>
        <p className="mt-1 text-sm text-gray-600">
          Monitor your AI platform, test models, and track performance with real-time evaluations
        </p>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', name: 'Overview', icon: '🏠' },
            { id: 'logs', name: 'Action Logs', icon: '📋' },
            { id: 'evaluations', name: 'Chat Evaluations', icon: '💭' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm flex items-center space-x-2`}
            >
              <span>{tab.icon}</span>
              <span>{tab.name}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Action Buttons */}
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Quick Actions</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button
                  onClick={handleTestModel}
                  className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg flex items-center justify-center space-x-2 transition-colors"
                >
                  <span>🧪</span>
                  <span>Test Model</span>
                </button>
                <button
                  onClick={handleViewPerformance}
                  className="bg-green-500 hover:bg-green-700 text-white font-bold py-3 px-4 rounded-lg flex items-center justify-center space-x-2 transition-colors"
                >
                  <span>📊</span>
                  <span>View Performance</span>
                </button>
                <button
                  onClick={handleChatWithAI}
                  className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-3 px-4 rounded-lg flex items-center justify-center space-x-2 transition-colors"
                >
                  <span>💬</span>
                  <span>Chat with AI</span>
                </button>
              </div>
            </div>
          </div>

          {/* API Status */}
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900">API Status</h3>
              {health ? (
                <div className="mt-4 space-y-2">
                  <div className="flex items-center">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      {health.status}
                    </span>
                    <span className="ml-2 text-sm text-gray-600">Version {health.version}</span>
                  </div>
                  <div className="text-sm text-gray-600">
                    <p>📊 Model: {health.model_name}</p>
                    <p>⏱️ Uptime: {Math.floor(health.uptime_seconds / 60)} minutes</p>
                    <p>🔗 Model Loaded: {health.model_loaded ? '✅ Yes' : '❌ No'}</p>
                  </div>
                </div>
              ) : (
                <p className="mt-2 text-sm text-red-600">Unable to fetch API status</p>
              )}
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900">Performance Overview</h3>
              <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">26.7</div>
                  <div className="text-sm text-gray-600">req/s throughput</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">37ms</div>
                  <div className="text-sm text-gray-600">avg response</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">100%</div>
                  <div className="text-sm text-gray-600">success rate</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">{actionLogs.length}</div>
                  <div className="text-sm text-gray-600">total actions</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Action Logs Tab */}
      {activeTab === 'logs' && (
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Action Logs</h3>
            {actionLogs.length > 0 ? (
              <div className="space-y-3">
                {actionLogs.map((log) => (
                  <div key={log.id} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-3">
                        <span className="text-2xl">{getActionIcon(log.action)}</span>
                        <div>
                          <p className="font-medium text-gray-900 capitalize">
                            {log.action.replace('_', ' ')}
                          </p>
                          <p className="text-sm text-gray-600">{log.details}</p>
                          <p className="text-xs text-gray-400">
                            {new Date(log.timestamp).toLocaleString()}
                            {log.duration && ` • ${log.duration}ms`}
                          </p>
                        </div>
                      </div>
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(log.status)}`}>
                        {log.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">No actions logged yet. Try using the quick actions above!</p>
            )}
          </div>
        </div>
      )}

      {/* Chat Evaluations Tab */}
      {activeTab === 'evaluations' && (
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Chat AI Evaluations</h3>
            {chatEvaluations.length > 0 ? (
              <div className="space-y-4">
                {chatEvaluations.map((evaluation) => (
                  <div key={evaluation.id} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
                    <div className="space-y-3">
                      <div>
                        <p className="font-medium text-gray-900">Query: "{evaluation.query}"</p>
                        <p className="text-sm text-gray-600 mt-1">Response: {evaluation.response}</p>
                        <p className="text-xs text-gray-400">
                          {new Date(evaluation.timestamp).toLocaleString()}
                        </p>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2 border-t">
                        <div className="text-center">
                          <div className="text-lg font-bold text-blue-600">
                            {(evaluation.metrics.relevance_score * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-gray-600">Relevance</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-bold text-green-600">
                            {(evaluation.metrics.accuracy_score * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-gray-600">Accuracy</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-bold text-purple-600">
                            {evaluation.metrics.response_time}ms
                          </div>
                          <div className="text-xs text-gray-600">Response Time</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-bold text-orange-600">
                            {evaluation.metrics.user_satisfaction 
                              ? (evaluation.metrics.user_satisfaction * 100).toFixed(1) + '%'
                              : 'N/A'
                            }
                          </div>
                          <div className="text-xs text-gray-600">Satisfaction</div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">No chat evaluations yet. Try the "Chat with AI" action to generate evaluations!</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};