import React from 'react';
import { AlertTriangle, CheckCircle, AlertCircle, Brain, FileText } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';

interface TrafficLightAlertProps {
  flagStatus: 'green' | 'yellow' | 'red';
  anomalyScore: number;
  summary: string;
  sliceId?: string;
  aiAnalysis?: string;
}

export function TrafficLightAlert({ flagStatus, anomalyScore, summary, sliceId, aiAnalysis }: TrafficLightAlertProps) {
  const getAlertConfig = () => {
    switch (flagStatus) {
      case 'green':
        return {
          icon: <CheckCircle className="h-8 w-8" />,
          color: 'text-green-600',
          bgColor: 'bg-green-50',
          borderColor: 'border-green-200',
          badgeVariant: 'default' as const,
          title: 'No Anomaly Detected',
          description: 'Normal liver parenchyma'
        };
      case 'yellow':
        return {
          icon: <AlertCircle className="h-8 w-8" />,
          color: 'text-yellow-600',
          bgColor: 'bg-yellow-50',
          borderColor: 'border-yellow-200',
          badgeVariant: 'secondary' as const,
          title: 'Mild Anomaly',
          description: 'Requires clinical correlation'
        };
      case 'red':
        return {
          icon: <AlertTriangle className="h-8 w-8" />,
          color: 'text-red-600',
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200',
          badgeVariant: 'destructive' as const,
          title: 'Severe Anomaly',
          description: 'Immediate attention required'
        };
      default:
        return {
          icon: <AlertCircle className="h-8 w-8" />,
          color: 'text-gray-600',
          bgColor: 'bg-gray-50',
          borderColor: 'border-gray-200',
          badgeVariant: 'outline' as const,
          title: 'Processing',
          description: 'Analysis in progress'
        };
    }
  };

  const config = getAlertConfig();

  return (
    <Card className={`w-full ${config.bgColor} ${config.borderColor}`}>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <div className={config.color}>
            {config.icon}
          </div>
          <div>
            <span>Alert Status</span>
            {sliceId && (
              <span className="ml-2 text-gray-600">- {sliceId}</span>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className={`${config.color}`}>{config.title}</h3>
            <p className="text-gray-600">{config.description}</p>
          </div>
          <Badge variant={config.badgeVariant} className="text-lg px-3 py-1">
            {anomalyScore}%
          </Badge>
        </div>

        {/* Traditional Summary */}
        {summary && (
          <div className="pt-4 border-t border-gray-200">
            <h4 className="flex items-center space-x-2 text-gray-700 mb-2">
              <FileText className="h-4 w-4" />
              <span>Clinical Summary</span>
            </h4>
            <p className="text-gray-600">{summary}</p>
          </div>
        )}
        
        {/* AI Analysis Section */}
        {aiAnalysis && (
          <div className="pt-4 border-t border-gray-200">
            <h4 className="flex items-center space-x-2 text-blue-700 mb-3">
              <Brain className="h-4 w-4" />
              <span>AI Medical Analysis</span>
              <Badge variant="outline" className="text-xs">
                Generated
              </Badge>
            </h4>
            <div className="bg-blue-50 rounded-lg p-3">
              <p className="text-gray-700 text-sm leading-relaxed">{aiAnalysis}</p>
              <div className="mt-2 pt-2 border-t border-blue-200">
                <p className="text-xs text-gray-500 italic">
                  * AI-generated analysis. Please review with qualified medical professional.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Traffic Light Visual Indicator */}
        <div className="flex justify-center pt-4">
          <div className="flex space-x-2">
            <div className={`w-4 h-4 rounded-full ${flagStatus === 'red' ? 'bg-red-500' : 'bg-gray-300'}`}></div>
            <div className={`w-4 h-4 rounded-full ${flagStatus === 'yellow' ? 'bg-yellow-500' : 'bg-gray-300'}`}></div>
            <div className={`w-4 h-4 rounded-full ${flagStatus === 'green' ? 'bg-green-500' : 'bg-gray-300'}`}></div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}