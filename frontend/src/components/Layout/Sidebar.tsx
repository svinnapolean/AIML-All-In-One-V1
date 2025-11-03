import React from 'react';
import { Link, useLocation } from 'react-router-dom';

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: '📊' },
  { name: 'AI Agent Chat', href: '/chat', icon: '🤖' },
  { name: 'ML Models', href: '/models', icon: '🧠' },
  { name: 'Data Management', href: '/data', icon: '📁' },
  { name: 'Results & Analysis', href: '/results', icon: '📈' },
  { name: 'Settings', href: '/settings', icon: '⚙️' },
];

export const Sidebar: React.FC<SidebarProps> = ({ open, onClose }) => {
  const location = useLocation();

  return (
    <>
      {/* Mobile sidebar */}
      {open && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div className="fixed inset-0 bg-gray-600 bg-opacity-75" onClick={onClose} />
          <div className="relative flex w-full max-w-xs flex-1 flex-col bg-gray-800">
            <div className="absolute top-0 right-0 -mr-12 pt-2">
              <button
                type="button"
                className="ml-1 flex h-10 w-10 items-center justify-center rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
                onClick={onClose}
              >
                <span className="sr-only">Close sidebar</span>
                <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <SidebarContent currentPath={location.pathname} />
          </div>
        </div>
      )}

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-64 lg:flex-col">
        <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-gray-800 px-6">
          <div className="flex h-16 shrink-0 items-center">
            <h2 className="text-white text-lg font-semibold">AI/ML Platform</h2>
          </div>
          <SidebarContent currentPath={location.pathname} />
        </div>
      </div>
    </>
  );
};

const SidebarContent: React.FC<{ currentPath: string }> = ({ currentPath }) => {
  return (
    <nav className="flex flex-1 flex-col">
      <ul className="flex flex-1 flex-col gap-y-7">
        <li>
          <ul className="-mx-2 space-y-1">
            {navigation.map((item) => {
              const isActive = currentPath === item.href;
              return (
                <li key={item.name}>
                  <Link
                    to={item.href}
                    className={`group flex gap-x-3 rounded-md p-2 text-sm leading-6 font-semibold ${
                      isActive
                        ? 'bg-gray-700 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-700'
                    }`}
                  >
                    <span className="text-lg">{item.icon}</span>
                    {item.name}
                  </Link>
                </li>
              );
            })}
          </ul>
        </li>
      </ul>
    </nav>
  );
};