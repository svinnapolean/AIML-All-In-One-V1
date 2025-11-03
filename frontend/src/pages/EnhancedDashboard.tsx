import React, { useState, useEffect, useMemo } from 'react';
import { getApiBaseUrl, getHealthCheckUrl } from '../config/apiConfig';
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
    };
    setActionLogs(prev => [newLog, ...prev]);
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

  // Handle Test Model action
  const handleTestModel = async () => {
    const logId = addActionLog('test_model', 'Navigating to model testing interface...');
    
    try {
      // Simulate loading time
      await new Promise(resolve => setTimeout(resolve, 1000));
      navigate('/test-model');
      updateActionLog(logId, 'success', 'Navigated to model testing interface');
    } catch (error) {
      updateActionLog(logId, 'error', `Navigation failed: ${error}`);
    }
  };

  // Handle View Performance action
  const handleViewPerformance = async () => {
    const logId = addActionLog('view_performance', 'Loading performance data...');
    const startTime = Date.now();
    
    try {
      // Simulate API call to get performance data
      const response = await fetch(getHealthCheckUrl());
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
        const healthResponse = await fetch(getHealthCheckUrl());
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
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6 space-y-4 sm:space-y-6">
        {/* Header - Responsive */}
        <div className="bg-white rounded-lg shadow-sm p-4 sm:p-6">
          <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-900">🚀 AI/ML Dashboard</h1>
          <p className="mt-1 text-xs sm:text-sm text-gray-600">
            Monitor your AI platform, test models, and track performance with real-time evaluations
          </p>
        </div>

        {/* Navigation Tabs - Mobile Friendly */}
        <div className="bg-white rounded-lg shadow-sm">
          <div className="border-b border-gray-200">
            <nav className="px-4 sm:px-6">
              <div className="flex space-x-2 sm:space-x-8 overflow-x-auto scrollbar-hide">
                {[
                  { id: 'overview', label: 'Overview', icon: '📊' },
                  { id: 'logs', label: 'Action Logs', icon: '📝' },
                  { id: 'evaluations', label: 'AI Evaluations', icon: '🧠' }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`flex items-center space-x-1 sm:space-x-2 py-3 sm:py-4 px-2 sm:px-1 border-b-2 font-medium text-xs sm:text-sm whitespace-nowrap ${
                      activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <span className="text-sm sm:text-base">{tab.icon}</span>
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>
            </nav>
          </div>
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-4 sm:space-y-6">
            {/* Quick Actions - Responsive Grid */}
            <div className="bg-white rounded-lg shadow-sm p-4 sm:p-6">
              <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-3 sm:mb-4">🚀 Quick Actions</h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
                {/* Test Model Button */}
                <button 
                  onClick={handleTestModel}
                  className="group bg-gradient-to-r from-blue-500 to-blue-600 text-white p-4 sm:p-6 rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all duration-200 transform hover:scale-105 shadow-md hover:shadow-lg"
                >
                  <div className="flex flex-col items-center text-center space-y-2">
                    <div className="text-2xl sm:text-3xl group-hover:animate-pulse">🧪</div>
                    <div className="font-semibold text-sm sm:text-base">Test Model</div>
                    <div className="text-xs sm:text-sm opacity-90">Evaluate AI models with custom datasets</div>
                  </div>
                </button>

                {/* View Performance Button */}
                <button 
                  onClick={handleViewPerformance}
                  className="group bg-gradient-to-r from-green-500 to-green-600 text-white p-4 sm:p-6 rounded-lg hover:from-green-600 hover:to-green-700 transition-all duration-200 transform hover:scale-105 shadow-md hover:shadow-lg"
                >
                  <div className="flex flex-col items-center text-center space-y-2">
                    <div className="text-2xl sm:text-3xl group-hover:animate-pulse">📊</div>
                    <div className="font-semibold text-sm sm:text-base">View Performance</div>
                    <div className="text-xs sm:text-sm opacity-90">Analyze model metrics and insights</div>
                  </div>
                </button>

                {/* Chat with AI Button */}
                <button 
                  onClick={handleChatWithAI}
                  className="group bg-gradient-to-r from-purple-500 to-purple-600 text-white p-4 sm:p-6 rounded-lg hover:from-purple-600 hover:to-purple-700 transition-all duration-200 transform hover:scale-105 shadow-md hover:shadow-lg sm:col-span-2 lg:col-span-1"
                >
                  <div className="flex flex-col items-center text-center space-y-2">
                    <div className="text-2xl sm:text-3xl group-hover:animate-pulse">💬</div>
                    <div className="font-semibold text-sm sm:text-base">Chat with AI</div>
                    <div className="text-xs sm:text-sm opacity-90">Interactive AI conversation interface</div>
                  </div>
                </button>
              </div>
            </div>

            {/* API Status */}
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900">🔗 API Status</h3>
                {health ? (
                  <div className="mt-2">
                    <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                      health.status === 'healthy' ? 'text-green-600 bg-green-100' : 'text-red-600 bg-red-100'
                    }`}>
                      {health.status === 'healthy' ? '✅' : '❌'} {health.status.toUpperCase()}
                    </div>
                    <div className="text-sm text-gray-600 mt-2">
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
    </div>
  );
};

export default Dashboard;