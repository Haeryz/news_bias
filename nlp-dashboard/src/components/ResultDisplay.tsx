"use client";
import { useState, useEffect } from 'react';
import { ClassificationResult } from '@/types';
import AnimatedContainer from './AnimatedContainer';

interface ResultDisplayProps {
  result: ClassificationResult | null;
}

export default function ResultDisplay({ result }: ResultDisplayProps) {
  const [animate, setAnimate] = useState(false);
  
  useEffect(() => {
    if (result) {
      setAnimate(false);
      setTimeout(() => setAnimate(true), 50);
    }
  }, [result]);

  if (!result) return null;

  const getColorClass = () => {
    switch (result.label.toLowerCase()) {
      case 'republik':
        return 'border-red-500 bg-red-50 text-red-800 dark:bg-red-900/30 dark:text-red-200';
      case 'demokrat':
        return 'border-blue-500 bg-blue-50 text-blue-800 dark:bg-blue-900/30 dark:text-blue-200';
      case 'netral':
        return 'border-green-500 bg-green-50 text-green-800 dark:bg-green-900/30 dark:text-green-200';
      case 'others':
        return 'border-purple-500 bg-purple-50 text-purple-800 dark:bg-purple-900/30 dark:text-purple-200';
      default:
        return 'border-gray-500 bg-gray-50 text-gray-800 dark:bg-gray-900/30 dark:text-gray-200';
    }
  };

  return (
    <AnimatedContainer className="w-full max-w-md">
      <div 
        className={`p-6 border-l-4 rounded-lg shadow-lg w-full
          transition-all duration-500 ease-in-out ${getColorClass()} 
          ${animate ? 'opacity-100 translate-y-0 scale-100' : 'opacity-0 translate-y-6 scale-95'}`}
      >
        <p className="text-xl font-semibold">{result.label}</p>
        {result.confidence !== undefined && (
          <p className="text-sm mt-2 opacity-80">
            Confidence: {(result.confidence * 100).toFixed(0)}%
          </p>
        )}
      </div>
    </AnimatedContainer>
  );
}