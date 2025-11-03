import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Dashboard } from './pages/EnhancedDashboard';
import { ChatPage } from './pages/ChatPage';
import { TestModelPage } from './pages/TestModelPage';

function App() {
  return (
    <Router>
      <div className="App min-h-screen bg-gray-50">
        <div className="w-full">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/test-model" element={<TestModelPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
