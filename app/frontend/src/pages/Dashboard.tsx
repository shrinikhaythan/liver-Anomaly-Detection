import React, { useState, useEffect } from 'react';
import { Navbar } from '../components/Navbar';
import { UploadBox } from '../components/UploadBox';
import { ProcessingStepper } from '../components/ProcessingStepper';
import { SliceViewer } from '../components/SliceViewer';
import { TrafficLightAlert } from '../components/TrafficLightAlert';
import { ReportPanel } from '../components/ReportPanel';
import apiService from '../services/api';

// Real processing steps
const processingSteps = [
  { id: 1, name: "Preprocessing", status: "pending", duration: "2-3s" },
  { id: 2, name: "CNN Liver Segmentation", status: "pending", duration: "5-8s" },
  { id: 3, name: "Diffusion Reconstruction", status: "pending", duration: "10-15s" },
  { id: 4, name: "Anomaly Detection", status: "pending", duration: "3-5s" },
  { id: 5, name: "Report Generation", status: "pending", duration: "1-2s" }
];

export function Dashboard() {
  const [isUploaded, setIsUploaded] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [currentSliceIndex, setCurrentSliceIndex] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [scanId, setScanId] = useState('');
  const [processedResults, setProcessedResults] = useState(null);
  const [uploadStats, setUploadStats] = useState({ totalSlices: 0, processedSlices: 0, anomalousSlices: 0, processingTime: "0s" });
  const [ctSlices, setCtSlices] = useState([]);
  const [patientData, setPatientData] = useState({ patientName: "Unknown Patient", patientId: "", studyDate: new Date().toISOString().split('T')[0], scanType: "CT Scan" });
  const [overallAssessment, setOverallAssessment] = useState({ maxAnomalyScore: 0, flagStatus: "green", summary: "Processing...", aiSummary: "" });
  const [error, setError] = useState('');

  const handleUpload = async (files: FileList) => {
    try {
      setError('');
      setIsUploaded(false);
      setIsProcessing(false);
      setIsComplete(false);
      
      const file = files[0];
      const uploadResult = await apiService.uploadFile(file);
      
      setScanId(uploadResult.scan_id);
      setUploadStats(prev => ({ ...prev, totalSlices: uploadResult.slice_count }));
      setPatientData(prev => ({ ...prev, patientId: uploadResult.scan_id }));
      setIsUploaded(true);
      
      // Start processing
      setTimeout(() => {
        processRealData(uploadResult.scan_id);
      }, 1000);
      
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
    }
  };

  const processRealData = async (scanId: string) => {
    try {
      setIsProcessing(true);
      setCurrentStep(0);
      
      // Start processing with real backend
      const processingResult = await apiService.processScan(scanId);
      
      // Simulate step progression
      for (let step = 0; step < processingSteps.length; step++) {
        setCurrentStep(step);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      
      // Set real results
      setProcessedResults(processingResult.results);
      setCtSlices(processingResult.results.results || []);
      setUploadStats({
        totalSlices: processingResult.results.totalSlices || 0,
        processedSlices: processingResult.results.totalSlices || 0,
        anomalousSlices: processingResult.results.results?.filter(r => r.flag === 'Red').length || 0,
        processingTime: "25s"
      });
      
      // Calculate overall assessment
      const results = processingResult.results.results || [];
      const maxScore = Math.max(...results.map(r => r.anomalyScore), 0);
      const highRisk = results.filter(r => r.flag === 'Red').length;
      const mediumRisk = results.filter(r => r.flag === 'Yellow').length;
      
      let flagStatus = 'green';
      let summary = 'All scans appear normal';
      
      if (highRisk > 0) {
        flagStatus = 'red';
        summary = `${highRisk} slice${highRisk > 1 ? 's' : ''} show significant anomalies requiring immediate attention`;
      } else if (mediumRisk > 0) {
        flagStatus = 'yellow';
        summary = `${mediumRisk} slice${mediumRisk > 1 ? 's' : ''} show mild anomalies requiring review`;
      }
      
      // Extract AI summary from the results
      const aiSummary = processingResult.results.ai_summary || '';
      
      setOverallAssessment({
        maxAnomalyScore: maxScore,
        flagStatus: flagStatus,
        summary: summary,
        aiSummary: aiSummary
      });
      
      setIsProcessing(false);
      setIsComplete(true);
      
    } catch (err) {
      setError(`Processing failed: ${err.message}`);
      setIsProcessing(false);
    }
  };

  const handleSliceChange = (index: number) => {
    setCurrentSliceIndex(index);
  };

  const currentSlice = ctSlices[currentSliceIndex] || { 
    sliceId: 'No data', 
    anomalyScore: 0, 
    flag: 'green', 
    findings: 'No analysis available',
    ai_analysis: ''
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      
      <main className="p-6 space-y-6">
        {/* Error Display */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <strong className="font-bold">Error: </strong>
            <span className="block sm:inline">{error}</span>
          </div>
        )}
        
        {/* Upload Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <UploadBox
              onUpload={handleUpload}
              isUploaded={isUploaded}
              sliceCount={uploadStats.totalSlices}
            />
          </div>
          <div>
            {(isProcessing || isComplete) && (
              <ProcessingStepper
                steps={processingSteps.map((step) => ({
                  ...step,
                  status: isComplete 
                    ? 'completed' 
                    : step.id <= currentStep + 1 
                    ? step.id === currentStep + 1 
                      ? 'processing' 
                      : 'completed'
                    : 'pending'
                }))}
                currentStep={currentStep}
              />
            )}
          </div>
        </div>

        {/* Results Section */}
        {isComplete && (
          <>
            {/* Main Results Grid */}
            <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
              {/* Slice Viewer - Takes up 3 columns */}
              <div className="xl:col-span-3">
                <SliceViewer
                  slices={ctSlices.map((s: any) => ({
                    sliceId: s.sliceId,
                    anomalyScore: s.anomalyScore,
                    flagStatus: (s.flag || 'green').toLowerCase(),
                    findings: s.findings || `Anomaly Score: ${s.anomalyScore}`,
                    aiAnalysis: s.ai_analysis || '',
                    originalImage: s.originalImage ? apiService.getResultImageUrl(s.originalImage) : undefined,
                    reconstructedImage: s.reconstructedImage ? apiService.getResultImageUrl(s.reconstructedImage) : undefined,
                    heatmapImage: s.heatmapPath ? apiService.getResultImageUrl(s.heatmapPath) : undefined,
                  }))}
                  currentSliceIndex={currentSliceIndex}
                  onSliceChange={handleSliceChange}
                />
              </div>

              {/* Alert Panel - Takes up 1 column */}
              <div>
                <TrafficLightAlert
                  flagStatus={((currentSlice.flag || 'green').toLowerCase()) as 'green' | 'yellow' | 'red'}
                  anomalyScore={currentSlice.anomalyScore || 0}
                  summary={currentSlice.findings || `Anomaly Score: ${currentSlice.anomalyScore || 0}`}
                  sliceId={currentSlice.sliceId || 'Unknown Slice'}
                  aiAnalysis={currentSlice.ai_analysis || ''}
                />
              </div>
            </div>

            {/* Report Panel */}
            <ReportPanel
              patientData={patientData}
              slices={ctSlices}
              overallAssessment={overallAssessment}
            />
          </>
        )}

        {/* Processing Message */}
        {isProcessing && (
          <div className="text-center py-12">
            <div className="inline-flex items-center space-x-2 text-blue-600">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <span>Processing CT slices... This may take a few minutes.</span>
            </div>
          </div>
        )}

        {/* Instructions */}
        {!isUploaded && (
          <div className="text-center py-12">
            <div className="max-w-md mx-auto space-y-4">
              <h2 className="text-gray-900">Welcome to LiverAI Diagnostics</h2>
              <p className="text-gray-600">
                Upload a folder of CT slices to begin automated liver anomaly detection. 
                Our AI system will analyze each slice and provide detailed diagnostic insights.
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}