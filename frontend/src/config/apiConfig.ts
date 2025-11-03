// API Configuration for different environments
export const getApiBaseUrl = (): string => {
  // In production (Docker), use relative paths handled by nginx proxy
  if (process.env.NODE_ENV === 'production') {
    return '/api';
  }
  
  // In development, use direct backend URL
  return process.env.REACT_APP_API_URL || 'http://localhost:8000';
};

export const getWebSocketUrl = (): string => {
  // In production (Docker), use relative paths handled by nginx proxy
  if (process.env.NODE_ENV === 'production') {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/api/agent/ws`;
  }
  
  // In development, use direct backend URL
  const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/agent/ws';
  return wsUrl;
};

// Health check configuration
export const getHealthCheckUrl = (): string => {
  if (process.env.NODE_ENV === 'production') {
    return '/api/health';
  }
  return process.env.REACT_APP_API_URL || 'http://localhost:8000/health';
};