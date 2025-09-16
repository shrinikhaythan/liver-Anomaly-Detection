import React from 'react';
import { Activity, Settings, User, Bell } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';

export function Navbar() {
  return (
    <nav className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <Activity className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-blue-900">LiverAI Diagnostics</h1>
              <p className="text-blue-600 text-sm">CT Anomaly Detection System</p>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <Button variant="ghost" size="sm" className="relative">
            <Bell className="h-5 w-5" />
            <Badge className="absolute -top-1 -right-1 h-5 w-5 rounded-full p-0 flex items-center justify-center text-xs">
              3
            </Badge>
          </Button>
          
          <Button variant="ghost" size="sm">
            <Settings className="h-5 w-5" />
          </Button>
          
          <Button variant="ghost" size="sm" className="flex items-center space-x-2">
            <User className="h-5 w-5" />
            <span>Dr. Sarah Chen</span>
          </Button>
        </div>
      </div>
    </nav>
  );
}