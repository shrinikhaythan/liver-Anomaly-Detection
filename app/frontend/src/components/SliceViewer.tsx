import React, { useState } from 'react';
import { ChevronLeft, ChevronRight, Eye, AlertTriangle, CheckCircle, AlertCircle, Brain, FileText } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { ImageWithFallback } from './figma/ImageWithFallback';
import { Badge } from './ui/badge';

interface CtSlice {
  sliceId: string;
  originalImage: string;
  reconstructedImage: string;
  heatmapImage: string;
  anomalyScore: number;
  flagStatus: string;
  aiAnalysis?: string;
  findings?: string;
}

interface SliceViewerProps {
  slices: CtSlice[];
  currentSliceIndex: number;
  onSliceChange: (index: number) => void;
}

export function SliceViewer({ slices, currentSliceIndex, onSliceChange }: SliceViewerProps) {
  const currentSlice = slices[currentSliceIndex];

  const handlePrevious = () => {
    if (currentSliceIndex > 0) {
      onSliceChange(currentSliceIndex - 1);
    }
  };

  const handleNext = () => {
    if (currentSliceIndex < slices.length - 1) {
      onSliceChange(currentSliceIndex + 1);
    }
  };

  const handleSliderChange = (value: number[]) => {
    onSliceChange(value[0]);
  };

  if (!currentSlice) return null;

  const getTrafficLightIcon = (flag: string) => {
    const flagLower = flag.toLowerCase();
    if (flagLower === 'red') {
      return <AlertCircle className="h-5 w-5 text-red-500" />;
    } else if (flagLower === 'yellow') {
      return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
    } else {
      return <CheckCircle className="h-5 w-5 text-green-500" />;
    }
  };

  const getTrafficLightBorder = (flag: string) => {
    const flagLower = flag.toLowerCase();
    if (flagLower === 'red') {
      return 'border-red-500 bg-red-50';
    } else if (flagLower === 'yellow') {
      return 'border-yellow-500 bg-yellow-50';
    } else {
      return 'border-green-500 bg-green-50';
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Eye className="h-5 w-5 text-blue-600" />
            <span>CT Slice Viewer</span>
            {getTrafficLightIcon(currentSlice.flagStatus)}
          </div>
          <div className="flex items-center space-x-4">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getTrafficLightBorder(currentSlice.flagStatus)}`}>
              {currentSlice.flagStatus.toUpperCase()} - {currentSlice.anomalyScore.toFixed(1)}%
            </div>
            <span className="text-gray-600">
              Slice {currentSlice.sliceId} ({currentSliceIndex + 1} of {slices.length})
            </span>
          </div>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Navigation Controls */}
        <div className="flex items-center space-x-4">
          <Button 
            variant="outline" 
            size="sm"
            onClick={handlePrevious}
            disabled={currentSliceIndex === 0}
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </Button>
          
          <div className="flex-1 px-4">
            <Slider
              value={[currentSliceIndex]}
              onValueChange={handleSliderChange}
              max={slices.length - 1}
              min={0}
              step={1}
              className="w-full"
            />
          </div>
          
          <Button 
            variant="outline" 
            size="sm"
            onClick={handleNext}
            disabled={currentSliceIndex === slices.length - 1}
          >
            Next
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        {/* Three-Panel Image Viewer */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Panel A: Original CT */}
          <div className="space-y-2">
            <h4 className="text-center text-gray-700">Original CT Slice</h4>
            <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden border">
              <ImageWithFallback
                src={currentSlice.originalImage}
                alt={`Original CT slice ${currentSlice.sliceId}`}
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          {/* Panel B: Healthy Reconstruction */}
          <div className="space-y-2">
            <h4 className="text-center text-gray-700">Healthy Reconstruction</h4>
            <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden border">
              <ImageWithFallback
                src={currentSlice.reconstructedImage}
                alt={`Reconstructed slice ${currentSlice.sliceId}`}
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          {/* Panel C: Heatmap Overlay */}
          <div className="space-y-2">
            <div className="flex items-center justify-center space-x-2">
              <h4 className="text-center text-gray-700">Anomaly Heatmap</h4>
              {getTrafficLightIcon(currentSlice.flagStatus)}
            </div>
            <div className={`aspect-square rounded-lg overflow-hidden border-2 ${getTrafficLightBorder(currentSlice.flagStatus)} border-2`}>
              <ImageWithFallback
                src={currentSlice.heatmapImage}
                alt={`Heatmap for slice ${currentSlice.sliceId}`}
                className="w-full h-full object-cover"
                fallback={
                  <div className="w-full h-full flex items-center justify-center text-gray-400">
                    <div className="text-center">
                      <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
                      <p>Heatmap Generating...</p>
                    </div>
                  </div>
                }
              />
            </div>
            <div className="text-center text-xs text-gray-500">
              Score: {currentSlice.anomalyScore.toFixed(1)}% | Status: {currentSlice.flagStatus.toUpperCase()}
            </div>
          </div>
        </div>

        {/* Slice Info */}
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-gray-600">Slice ID:</span>
              <span className="ml-2">{currentSlice.sliceId}</span>
            </div>
            <div>
              <span className="text-gray-600">Anomaly Score:</span>
              <span className="ml-2">{currentSlice.anomalyScore}%</span>
            </div>
          </div>
        </div>

        {/* AI Analysis Section */}
        {currentSlice.aiAnalysis && (
          <div className="mt-6">
            <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center space-x-2 text-blue-800">
                  <Brain className="h-5 w-5" />
                  <span>AI Medical Analysis</span>
                  <Badge variant="outline" className="ml-2 text-xs">
                    Generated Report
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <div className="flex items-start space-x-2 mb-2">
                    <FileText className="h-4 w-4 text-blue-600 mt-1 flex-shrink-0" />
                    <div className="text-sm text-gray-700 leading-relaxed">
                      {currentSlice.aiAnalysis}
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <p className="text-xs text-gray-500 italic">
                      * This analysis is AI-generated and should be reviewed by a qualified medical professional.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
        
        {/* Fallback for when no AI analysis is available */}
        {!currentSlice.aiAnalysis && currentSlice.findings && (
          <div className="mt-6">
            <Card className="bg-gray-50">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center space-x-2 text-gray-700">
                  <FileText className="h-5 w-5" />
                  <span>Analysis Summary</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700">{currentSlice.findings}</p>
              </CardContent>
            </Card>
          </div>
        )}
      </CardContent>
    </Card>
  );
}