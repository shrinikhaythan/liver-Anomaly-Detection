import React, { useState, useCallback } from 'react';
import { Upload, FileText, CheckCircle } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';

interface UploadBoxProps {
  onUpload: (files: FileList) => void;
  isUploaded: boolean;
  sliceCount: number;
}

export function UploadBox({ onUpload, isUploaded, sliceCount }: UploadBoxProps) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onUpload(files);
    }
  }, [onUpload]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onUpload(files);
    }
  }, [onUpload]);

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Upload className="h-5 w-5 text-blue-600" />
            <h3>Upload CT Slices</h3>
          </div>
          
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              isDragOver 
                ? 'border-blue-500 bg-blue-50' 
                : isUploaded 
                ? 'border-green-500 bg-green-50'
                : 'border-gray-300 hover:border-blue-400'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {isUploaded ? (
              <div className="space-y-3">
                <CheckCircle className="mx-auto h-12 w-12 text-green-600" />
                <div>
                  <p className="text-green-700">Upload Complete</p>
                  <p className="text-green-600">{sliceCount} CT slices detected</p>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <FileText className="mx-auto h-12 w-12 text-gray-400" />
                <div>
                  <p className="text-gray-600">
                    Drag and drop CT slice folder here, or{' '}
                    <label className="text-blue-600 hover:text-blue-700 cursor-pointer underline">
                      browse files
                      <input
                        type="file"
                        multiple
                        accept=".dcm,.jpg,.png"
                        onChange={handleFileSelect}
                        className="hidden"
                        webkitdirectory=""
                      />
                    </label>
                  </p>
                  <p className="text-gray-500">Supports DICOM, JPG, PNG formats</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}