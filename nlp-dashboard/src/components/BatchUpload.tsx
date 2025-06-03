"use client";
import { useState, useRef } from 'react';
import { classifyBatchService } from '@/services/api';
import { BatchClassificationResult, BatchProcessingOptions } from '@/types';
import AnimatedContainer from './AnimatedContainer';
import CSVTemplateDownload from './CSVTemplateDownload';

interface BatchUploadProps {
  onResult: (result: BatchClassificationResult) => void;
}

export default function BatchUpload({ onResult }: BatchUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Processing options
  const [expectedLabels, setExpectedLabels] = useState(false);
  const [hasHeader, setHasHeader] = useState(true);
  const [saveReport, setSaveReport] = useState(false);
  const [exportResults, setExportResults] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.toLowerCase().endsWith('.csv')) {
        setFile(droppedFile);
      } else {
        alert('Please upload a CSV file');
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.toLowerCase().endsWith('.csv')) {
        setFile(selectedFile);
      } else {
        alert('Please upload a CSV file');
        e.target.value = '';
      }
    }
  };

  const handleProcess = async () => {
    if (!file) return;
    
    setIsSubmitting(true);
    setIsLoading(true);
    
    try {
      const options: BatchProcessingOptions = {
        expectedLabels,
        hasHeader,
        saveReport,
        exportResults
      };
      
      const result = await classifyBatchService(file, options);
      onResult(result);
    } catch (error) {
      console.error('Error processing batch:', error);
      alert('Error processing batch: ' + (error instanceof Error ? error.message : 'Unknown error'));
    } finally {
      setIsLoading(false);
      setTimeout(() => setIsSubmitting(false), 300);
    }
  };

  const removeFile = () => {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  return (
    <AnimatedContainer className="w-full">
      <div className="space-y-6">
        {/* CSV Template Download */}
        <CSVTemplateDownload />
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
        <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">
          Batch Processing
        </h2>
        
        <div className={`flex flex-col space-y-4 transition-transform duration-300 ${isSubmitting ? 'scale-98 opacity-90' : 'scale-100 opacity-100'}`}>
          {/* File Upload Area */}
          <div
            className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
              dragActive
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                : file
                ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            
            {file ? (
              <div className="space-y-2">
                <div className="text-green-600 dark:text-green-400">
                  <svg className="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-gray-800 dark:text-gray-200">{file.name}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {(file.size / 1024).toFixed(1)} KB
                </p>
                <button
                  onClick={removeFile}
                  className="text-xs text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300"
                >
                  Remove file
                </button>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="text-gray-400">
                  <svg className="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-gray-800 dark:text-gray-200">
                  Drop your CSV file here, or click to browse
                </p>                <p className="text-xs text-gray-500 dark:text-gray-400">
                  CSV must contain a &apos;content&apos; column with text to analyze
                </p>
              </div>
            )}
          </div>

          {/* Processing Options */}
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg border border-gray-200 dark:border-gray-600">
            <h3 className="text-lg font-medium mb-3 text-gray-800 dark:text-gray-200">
              Processing Options
            </h3>
            
            <div className="space-y-3">
              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  id="hasHeader"
                  checked={hasHeader}
                  onChange={(e) => setHasHeader(e.target.checked)}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="hasHeader" className="text-sm text-gray-700 dark:text-gray-300">
                  CSV has header row
                </label>
              </div>
              
              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  id="expectedLabels"
                  checked={expectedLabels}
                  onChange={(e) => setExpectedLabels(e.target.checked)}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="expectedLabels" className="text-sm text-gray-700 dark:text-gray-300">
                  CSV includes expected labels (for evaluation)
                </label>
              </div>
              
              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  id="saveReport"
                  checked={saveReport}
                  onChange={(e) => setSaveReport(e.target.checked)}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="saveReport" className="text-sm text-gray-700 dark:text-gray-300">
                  Generate HTML report
                </label>
              </div>
              
              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  id="exportResults"
                  checked={exportResults}
                  onChange={(e) => setExportResults(e.target.checked)}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="exportResults" className="text-sm text-gray-700 dark:text-gray-300">
                  Export results as CSV
                </label>
              </div>
            </div>
              <div className="mt-3 text-xs text-gray-500 dark:text-gray-400 space-y-1">
              <p>💡 Expected labels should be in &apos;label&apos; column (0=Republican, 1=Democrat, 2=Neutral, 3=Others)</p>
              <p>📊 Evaluation metrics will be calculated if expected labels are provided</p>
            </div>
          </div>

          {/* Process Button */}
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {file ? `Ready to process ${file.name}` : 'Select a CSV file to begin'}
            </span>
            <button
              className={`px-6 py-3 rounded-lg font-semibold transition-all duration-300
                ${isLoading || !file
                  ? 'bg-gray-400 dark:bg-gray-600 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 text-white'
                }
                shadow-md hover:shadow-lg transform hover:-translate-y-0.5 disabled:transform-none`}
              onClick={handleProcess}
              disabled={isLoading || !file}
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="w-5 h-5 border-2 border-t-transparent border-white rounded-full animate-spin mr-2"></div>
                  <span>Processing Batch...</span>
                </div>
              ) : (
                'Process Batch'
              )}            </button>
          </div>
        </div>
        </div>
      </div>
    </AnimatedContainer>
  );
}
