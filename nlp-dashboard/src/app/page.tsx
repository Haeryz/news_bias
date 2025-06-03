"use client";
import { useState } from "react";
import ClassificationForm from "@/components/ClassificationForm";
import ResultDisplay from "@/components/ResultDisplay";
import BatchUpload from "@/components/BatchUpload";
import BatchResultDisplay from "@/components/BatchResultDisplay";
import { ClassificationResult, BatchClassificationResult } from "@/types";

export default function Home() {
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [batchResult, setBatchResult] = useState<BatchClassificationResult | null>(null);
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single');

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800 dark:text-white mb-2">
            News Bias Classification
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Analyze political bias, sentiment, and key phrases in news articles
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="flex space-x-1 bg-white dark:bg-gray-800 p-1 rounded-lg shadow-md">
            <button
              onClick={() => setActiveTab('single')}
              className={`py-3 px-6 rounded-md font-medium transition-all ${
                activeTab === 'single'
                  ? 'bg-blue-500 text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              Single Article
            </button>
            <button
              onClick={() => setActiveTab('batch')}
              className={`py-3 px-6 rounded-md font-medium transition-all ${
                activeTab === 'batch'
                  ? 'bg-blue-500 text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              Batch Processing
            </button>
          </div>
        </div>
        
        <div className="flex flex-col items-center space-y-8">
          {/* Single Article Tab */}
          {activeTab === 'single' && (
            <>
              <div className="w-full max-w-2xl">
                <ClassificationForm onResult={setResult} />
              </div>
              
              {result && (
                <div className="w-full">
                  <ResultDisplay result={result} />
                </div>
              )}
            </>
          )}

          {/* Batch Processing Tab */}
          {activeTab === 'batch' && (
            <>
              {!batchResult ? (
                <div className="w-full max-w-2xl">
                  <BatchUpload onResult={setBatchResult} />
                </div>
              ) : (
                <div className="w-full">
                  <BatchResultDisplay 
                    result={batchResult} 
                    onClear={() => setBatchResult(null)} 
                  />
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}