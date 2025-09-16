import React from 'react';
import { Check, Clock, Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

interface ProcessingStep {
  id: number;
  name: string;
  status: 'pending' | 'processing' | 'completed';
  duration?: string;
}

interface ProcessingStepperProps {
  steps: ProcessingStep[];
  currentStep: number;
}

export function ProcessingStepper({ steps, currentStep }: ProcessingStepperProps) {
  const getStepIcon = (step: ProcessingStep, index: number) => {
    if (step.status === 'completed') {
      return <Check className="h-5 w-5 text-white" />;
    } else if (step.status === 'processing' || index === currentStep) {
      return <Loader2 className="h-5 w-5 text-white animate-spin" />;
    } else {
      return <span className="text-white">{step.id}</span>;
    }
  };

  const getStepColor = (step: ProcessingStep, index: number) => {
    if (step.status === 'completed') {
      return 'bg-green-600';
    } else if (step.status === 'processing' || index === currentStep) {
      return 'bg-blue-600';
    } else {
      return 'bg-gray-400';
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Clock className="h-5 w-5 text-blue-600" />
          <span>Processing Workflow</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {steps.map((step, index) => (
            <div key={step.id} className="flex items-center space-x-4">
              <div 
                className={`flex items-center justify-center w-10 h-10 rounded-full ${getStepColor(step, index)}`}
              >
                {getStepIcon(step, index)}
              </div>
              
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <p className={`${step.status === 'completed' ? 'text-gray-900' : 'text-gray-600'}`}>
                    {step.name}
                  </p>
                  {step.duration && step.status === 'completed' && (
                    <span className="text-gray-500">{step.duration}</span>
                  )}
                </div>
                
                {step.status === 'processing' || index === currentStep ? (
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
                  </div>
                ) : null}
              </div>
              
              {index < steps.length - 1 && (
                <div className="absolute left-5 mt-12 w-0.5 h-8 bg-gray-300"></div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}