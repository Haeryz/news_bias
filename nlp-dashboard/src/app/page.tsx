"use client";
import { useState } from "react";
import ClassificationForm from "@/components/ClassificationForm";
import ResultDisplay from "@/components/ResultDisplay";
import { ClassificationResult } from "@/types";

export default function Home() {
  const [result, setResult] = useState<ClassificationResult | null>(null);

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
        
        <div className="flex flex-col items-center space-y-8">
          <div className="w-full max-w-2xl">
            <ClassificationForm onResult={setResult} />
          </div>
          
          {result && (
            <div className="w-full">
              <ResultDisplay result={result} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}