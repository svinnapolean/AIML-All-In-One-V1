import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
  model?: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  created_at: string;
  model: string;
}

export const ChatPage: React.FC = () => {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('chat-assistant');
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [testData, setTestData] = useState('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [inputMode, setInputMode] = useState<'text' | 'data' | 'file'>('text');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const models = [
    { name: 'chat-assistant', type: 'Chat AI', description: 'General purpose AI assistant' },
    { name: 'ml-predictor', type: 'ML Model', description: 'Machine learning prediction model' }
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentSession?.messages]);

  useEffect(() => {
    const handleResize = () => {
      if (typeof window !== 'undefined' && window.innerWidth >= 1024) {
        setIsSidebarOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const createNewChat = () => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      created_at: new Date().toISOString(),
      model: selectedModel
    };
    setSessions(prev => [newSession, ...prev]);
    setCurrentSession(newSession);
    if (typeof window !== 'undefined' && window.innerWidth < 1024) {
      setIsSidebarOpen(false);
    }
  };

  const handleSessionSelect = (session: ChatSession) => {
    setCurrentSession(session);
    if (typeof window !== 'undefined' && window.innerWidth < 1024) {
      setIsSidebarOpen(false);
    }
  };

  const sendMessage = async () => {
    if ((!message.trim() && !testData.trim() && !uploadedFile) || isLoading) return;

    let currentSessionToUse = currentSession;
    if (!currentSessionToUse) {
      createNewChat();
      currentSessionToUse = {
        id: Date.now().toString(),
        title: 'New Chat',
        messages: [],
        created_at: new Date().toISOString(),
        model: selectedModel
      };
      setSessions(prev => [currentSessionToUse!, ...prev]);
      setCurrentSession(currentSessionToUse);
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMode === 'text' ? message : inputMode === 'data' ? `Test Data: ${testData}` : `File Upload: ${uploadedFile?.name}`,
      timestamp: new Date().toISOString(),
      model: selectedModel
    };

    const updatedSession = {
      ...currentSessionToUse,
      messages: [...currentSessionToUse.messages, userMessage]
    };
    
    setCurrentSession(updatedSession);
    setSessions(prev => prev.map(s => s.id === updatedSession.id ? updatedSession : s));

    if (currentSessionToUse.messages.length === 0) {
      const title = userMessage.content.substring(0, 50) + (userMessage.content.length > 50 ? '...' : '');
      setSessions(prev => prev.map(session => 
        session.id === currentSessionToUse!.id ? { ...session, title } : session
      ));
    }

    setIsLoading(true);
    
    try {
      let assistantContent = '';
      
      if (inputMode === 'text') {
        assistantContent = `AI Response: I understand you're asking about "${message}". This is a simulated response from the ${selectedModel} model.`;
      } else if (inputMode === 'data') {
        assistantContent = `Prediction Result: Based on test data [${testData}], the model predicts: [simulated result]. This would connect to your ML API at /api/predict.`;
      } else if (inputMode === 'file' && uploadedFile) {
        assistantContent = `Batch Processing: File "${uploadedFile.name}" (${(uploadedFile.size / 1024).toFixed(1)} KB) would be processed through /api/predict-batch. Results would be displayed here.`;
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: assistantContent,
        timestamp: new Date().toISOString(),
        model: selectedModel
      };

      const finalSession = {
        ...updatedSession,
        messages: [...updatedSession.messages, assistantMessage]
      };
      
      setCurrentSession(finalSession);
      setSessions(prev => prev.map(s => s.id === finalSession.id ? finalSession : s));

    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `Error: ${error}. Please ensure your backend is running and accessible.`,
        timestamp: new Date().toISOString(),
        model: selectedModel
      };

      const errorSession = {
        ...updatedSession,
        messages: [...updatedSession.messages, errorMessage]
      };
      
      setCurrentSession(errorSession);
      setSessions(prev => prev.map(s => s.id === errorSession.id ? errorSession : s));
    }

    setIsLoading(false);
    setMessage('');
    setTestData('');
    setUploadedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setInputMode('file');
    }
  };

  const deleteSession = (sessionId: string) => {
    setSessions(prev => prev.filter(s => s.id !== sessionId));
    if (currentSession?.id === sessionId) {
      setCurrentSession(null);
    }
    if (typeof window !== 'undefined' && window.innerWidth < 1024) {
      setIsSidebarOpen(false);
    }
  };

  return (
    <div className="h-screen bg-gray-50 flex flex-col">
      {/* Top Navigation */}
      <div className="bg-white border-b border-gray-200 px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <button
            onClick={() => navigate('/')}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
          >
            <span>←</span>
            <span className="text-sm sm:text-base">Back to Dashboard</span>
          </button>
          <h1 className="flex-1 text-center text-base sm:text-lg font-semibold text-gray-900">
            🤖 AI Chat Interface
          </h1>
          <div className="flex items-center justify-end flex-1 sm:flex-none gap-3">
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="lg:hidden inline-flex items-center space-x-2 rounded-lg border border-gray-200 px-3 py-2 text-sm text-gray-700 shadow-sm hover:bg-gray-50"
            >
              <span>📜</span>
              <span>History</span>
            </button>
            <div className="hidden lg:block w-32" aria-hidden="true"></div>
          </div>
        </div>
      </div>

      {/* Mobile Overlay */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/40 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        ></div>
      )}

      {/* Main Content */}
      <div className="flex-1 overflow-hidden flex flex-col lg:flex-row">
        {/* Sidebar */}
        <div
          className={`z-40 bg-white border-r border-gray-200 flex flex-col transition-transform duration-200 ease-in-out lg:static lg:z-auto lg:w-80 lg:flex lg:translate-x-0 ${
            isSidebarOpen
              ? 'fixed inset-y-0 left-0 flex w-full max-w-sm translate-x-0 shadow-xl'
              : 'hidden lg:flex'
          }`}
        >
          {/* New Chat Button */}
          <div className="p-4 border-b border-gray-200">
            <button
              onClick={createNewChat}
              className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded-lg flex items-center justify-center space-x-2"
            >
              <span>➕</span>
              <span>New Chat</span>
            </button>
          </div>

          {/* Mobile Close */}
          <div className="lg:hidden flex items-center justify-between px-4 py-3 border-b border-gray-200">
            <h2 className="text-sm font-semibold text-gray-900">Conversations</h2>
            <button
              onClick={() => setIsSidebarOpen(false)}
              className="text-gray-500 hover:text-gray-900 text-sm"
            >
              Close ✕
            </button>
          </div>

          {/* Model Selection */}
          <div className="p-4 border-b border-gray-200">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {models.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.name} ({model.type})
                </option>
              ))}
            </select>
          </div>

          {/* Chat History */}
          <div className="flex-1 overflow-y-auto p-4">
            <h3 className="text-sm font-medium text-gray-700 mb-3">Chat History</h3>
            {sessions.length === 0 ? (
              <p className="text-xs text-gray-500">No chat history yet</p>
            ) : (
              <div className="space-y-2">
                {sessions.map((session) => (
                  <div
                    key={session.id}
                    className={`p-3 rounded-lg cursor-pointer hover:bg-gray-50 ${
                      currentSession?.id === session.id ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50'
                    }`}
                    onClick={() => handleSessionSelect(session)}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {session.title}
                        </p>
                        <p className="text-xs text-gray-500">
                          {new Date(session.created_at).toLocaleDateString()}
                        </p>
                        <p className="text-xs text-blue-600">
                          {session.model}
                        </p>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteSession(session.id);
                        }}
                        className="text-gray-400 hover:text-red-500 text-xs ml-2"
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Chat Header */}
          <div className="bg-white border-b border-gray-200 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">
                  {currentSession ? currentSession.title : 'Select or Create a Chat'}
                </h2>
                {currentSession && (
                  <p className="text-sm text-gray-500">
                    Model: {currentSession.model} • {currentSession.messages.length} messages
                  </p>
                )}
              </div>
              
              {/* Input Mode Selector */}
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => setInputMode('text')}
                  className={`px-3 py-1 text-xs sm:text-sm rounded ${
                    inputMode === 'text' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'
                  }`}
                >
                  💬 Chat
                </button>
                <button
                  onClick={() => setInputMode('data')}
                  className={`px-3 py-1 text-xs sm:text-sm rounded ${
                    inputMode === 'data' ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-700'
                  }`}
                >
                  🧪 Test Data
                </button>
                <button
                  onClick={() => setInputMode('file')}
                  className={`px-3 py-1 text-xs sm:text-sm rounded ${
                    inputMode === 'file' ? 'bg-purple-500 text-white' : 'bg-gray-200 text-gray-700'
                  }`}
                >
                  📁 Upload CSV
                </button>
              </div>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-4">
            {currentSession ? (
              <>
                {currentSession.messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-full sm:max-w-2xl lg:max-w-3xl p-4 rounded-lg ${
                        msg.type === 'user'
                          ? 'bg-blue-500 text-white'
                          : 'bg-white border border-gray-200 text-gray-900'
                      }`}
                    >
                      <div className="whitespace-pre-wrap break-words text-sm sm:text-base">{msg.content}</div>
                      <div className="text-xs opacity-75 mt-2">
                        {new Date(msg.timestamp).toLocaleTimeString()} • {msg.model}
                      </div>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="max-w-full sm:max-w-md bg-white border border-gray-200 text-gray-900 p-4 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                        <span>Processing...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </>
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="text-4xl sm:text-6xl mb-4">💬</div>
                  <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-2">
                    Welcome to AI Chat
                  </h2>
                  <p className="text-gray-600 mb-4">
                    Create a new chat or select from your chat history
                  </p>
                  <button
                    onClick={createNewChat}
                    className="bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded-lg"
                  >
                    Start New Chat
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="bg-white border-t border-gray-200 p-4">
            {inputMode === 'text' && (
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-4">
                <input
                  type="text"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                  placeholder="Type your message..."
                  className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isLoading}
                />
                <button
                  onClick={sendMessage}
                  disabled={!message.trim() || isLoading}
                  className="w-full sm:w-auto bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white px-6 py-2 rounded-lg font-medium"
                >
                  Send
                </button>
              </div>
            )}

            {inputMode === 'data' && (
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Enter test data (comma-separated):
                </label>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-4">
                  <input
                    type="text"
                    value={testData}
                    onChange={(e) => setTestData(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="e.g., 1,2,3,4 or 0.5,1.2,3.7,2.1"
                    className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
                    disabled={isLoading}
                  />
                  <button
                    onClick={sendMessage}
                    disabled={!testData.trim() || isLoading}
                    className="w-full sm:w-auto bg-green-500 hover:bg-green-600 disabled:bg-gray-300 text-white px-6 py-2 rounded-lg font-medium"
                  >
                    Predict
                  </button>
                </div>
              </div>
            )}

            {inputMode === 'file' && (
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Upload CSV file for batch prediction:
                </label>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-4">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleFileSelect}
                    className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    disabled={isLoading}
                  />
                  <button
                    onClick={sendMessage}
                    disabled={!uploadedFile || isLoading}
                    className="w-full sm:w-auto bg-purple-500 hover:bg-purple-600 disabled:bg-gray-300 text-white px-6 py-2 rounded-lg font-medium"
                  >
                    Process
                  </button>
                </div>
                {uploadedFile && (
                  <p className="text-sm text-gray-600">
                    Selected: {uploadedFile.name} ({(uploadedFile.size / 1024).toFixed(1)} KB)
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
