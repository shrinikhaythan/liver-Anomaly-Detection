import React, { useState } from 'react';
import { FileText, ChevronDown, ChevronUp, Download, Printer, Brain, Sparkles } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { Badge } from './ui/badge';
import { Textarea } from './ui/textarea';

interface CtSlice {
  sliceId: string;
  anomalyScore: number;
  flagStatus: 'green' | 'yellow' | 'red';
  findings: string;
  aiAnalysis?: string;
}

interface PatientData {
  patientName: string;
  patientId: string;
  studyDate: string;
  scanType: string;
}

interface ReportPanelProps {
  patientData: PatientData;
  slices: CtSlice[];
  overallAssessment: {
    maxAnomalyScore: number;
    flagStatus: 'green' | 'yellow' | 'red';
    summary: string;
    aiSummary?: string;
  };
}

export function ReportPanel({ patientData, slices, overallAssessment }: ReportPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [additionalNotes, setAdditionalNotes] = useState('');

  const getFlagColor = (status: string) => {
    switch (status) {
      case 'green': return 'bg-green-100 text-green-800';
      case 'yellow': return 'bg-yellow-100 text-yellow-800';  
      case 'red': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getFlagText = (status: string) => {
    switch (status) {
      case 'green': return 'Normal';
      case 'yellow': return 'Attention';
      case 'red': return 'Critical';
      default: return 'Unknown';
    }
  };

  const anomalousSlices = slices.filter(slice => slice.flagStatus !== 'green');
  const criticalSlices = slices.filter(slice => slice.flagStatus === 'red');

  return (
    <Card className="w-full">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <FileText className="h-5 w-5 text-blue-600" />
                <span>Doctor-Friendly Report</span>
                <Badge className={getFlagColor(overallAssessment.flagStatus)}>
                  {getFlagText(overallAssessment.flagStatus)}
                </Badge>
              </div>
              {isOpen ? (
                <ChevronUp className="h-5 w-5 text-gray-500" />
              ) : (
                <ChevronDown className="h-5 w-5 text-gray-500" />
              )}
            </CardTitle>
          </CardHeader>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="space-y-6">
            {/* Patient Information */}
            <div className="bg-blue-50 rounded-lg p-4">
              <h3 className="text-blue-900 mb-3">Patient Information</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-blue-700">Patient Name:</span>
                  <span className="ml-2 text-blue-900">{patientData.patientName}</span>
                </div>
                <div>
                  <span className="text-blue-700">Patient ID:</span>
                  <span className="ml-2 text-blue-900">{patientData.patientId}</span>
                </div>
                <div>
                  <span className="text-blue-700">Study Date:</span>
                  <span className="ml-2 text-blue-900">{patientData.studyDate}</span>
                </div>
                <div>
                  <span className="text-blue-700">Scan Type:</span>
                  <span className="ml-2 text-blue-900">{patientData.scanType}</span>
                </div>
              </div>
            </div>

            {/* Overall Assessment */}
            <div>
              <h3 className="mb-3">Overall Assessment</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span>Maximum Anomaly Score:</span>
                  <Badge variant="outline" className="text-lg">
                    {overallAssessment.maxAnomalyScore}%
                  </Badge>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg">
                  <span className="block text-gray-700 mb-2">Summary:</span>
                  <p className="text-gray-900">{overallAssessment.summary}</p>
                </div>
                
                {/* AI Summary Section */}
                {overallAssessment.aiSummary && (
                  <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200">
                    <div className="flex items-center space-x-2 mb-3">
                      <Brain className="h-5 w-5 text-blue-600" />
                      <span className="block text-blue-800 font-medium">AI Summary Report</span>
                      <Badge variant="outline" className="text-xs">
                        <Sparkles className="h-3 w-3 mr-1" />
                        Generated
                      </Badge>
                    </div>
                    <div className="bg-white rounded-lg p-3 shadow-sm">
                      <p className="text-gray-900 leading-relaxed">{overallAssessment.aiSummary}</p>
                      <div className="mt-3 pt-2 border-t border-gray-200">
                        <p className="text-xs text-gray-500 italic">
                          * This analysis is AI-generated and should be reviewed by a qualified medical professional.
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Summary Statistics */}
            <div>
              <h3 className="mb-3">Summary Statistics</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <div className="text-green-600">Normal Slices</div>
                  <div className="text-green-900">{slices.length - anomalousSlices.length}</div>
                </div>
                <div className="text-center p-3 bg-yellow-50 rounded-lg">
                  <div className="text-yellow-600">Attention Required</div>
                  <div className="text-yellow-900">{anomalousSlices.length - criticalSlices.length}</div>
                </div>
                <div className="text-center p-3 bg-red-50 rounded-lg">
                  <div className="text-red-600">Critical Findings</div>
                  <div className="text-red-900">{criticalSlices.length}</div>
                </div>
              </div>
            </div>

            {/* Detailed Findings */}
            {anomalousSlices.length > 0 && (
              <div>
                <h3 className="mb-3">Detailed Findings</h3>
                <div className="space-y-3">
                  {anomalousSlices.map((slice) => (
                    <div key={slice.sliceId} className="border rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-gray-900">{slice.sliceId}</span>
                        <div className="flex items-center space-x-2">
                          <Badge className={getFlagColor(slice.flagStatus)}>
                            {getFlagText(slice.flagStatus)}
                          </Badge>
                          <Badge variant="outline">{slice.anomalyScore}%</Badge>
                        </div>
                      </div>
                      <p className="text-gray-700">{slice.findings}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Additional Notes */}
            <div>
              <h3 className="mb-3">Additional Clinical Notes</h3>
              <Textarea
                placeholder="Enter additional observations or clinical notes..."
                value={additionalNotes}
                onChange={(e) => setAdditionalNotes(e.target.value)}
                className="min-h-[100px]"
              />
            </div>

            {/* Action Buttons */}
            <div className="flex space-x-3 pt-4 border-t">
              <Button variant="outline" className="flex items-center space-x-2">
                <Download className="h-4 w-4" />
                <span>Export PDF</span>
              </Button>
              <Button variant="outline" className="flex items-center space-x-2">
                <Printer className="h-4 w-4" />
                <span>Print Report</span>
              </Button>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}