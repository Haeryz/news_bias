"use client";
import { useState } from 'react';
import { classifyTextService } from '@/services/api';
import { ClassificationResult } from '@/types';
import AnimatedContainer from './AnimatedContainer';

interface ClassificationFormProps {
  onResult: (result: ClassificationResult) => void;
}

export default function ClassificationForm({ onResult }: ClassificationFormProps) {
  const [text, setText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleClassify = async () => {
    if (!text.trim()) return;
    
    setIsSubmitting(true);
    setIsLoading(true);
    try {
      const result = await classifyTextService(text);
      onResult(result);
    } catch (error) {
      console.error('Error classifying text:', error);
    } finally {
      setIsLoading(false);
      setTimeout(() => setIsSubmitting(false), 300); // Reset animation state
    }
  };
  return (
    <AnimatedContainer className="w-full">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
        <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">
          Enter Text for Analysis
        </h2>
        <div className={`flex flex-col space-y-4 transition-transform duration-300 ${isSubmitting ? 'scale-98 opacity-90' : 'scale-100 opacity-100'}`}>
          <textarea
            className="w-full p-4 border border-gray-200 dark:border-gray-700 rounded-lg 
              bg-gray-50 dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 
              focus:border-blue-500 focus:bg-white dark:focus:bg-gray-600 outline-none 
              min-h-[200px] resize-none transition-all duration-300 
              text-gray-800 dark:text-gray-200"
            placeholder="Paste your news article or text here for bias analysis...

Example: 'The new immigration policy has sparked controversy among lawmakers..'"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {text.length} characters
            </span>
            <button
              className={`px-6 py-3 rounded-lg font-semibold transition-all duration-300
                ${isLoading 
                  ? 'bg-gray-400 dark:bg-gray-600 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white'
                }
                shadow-md hover:shadow-lg transform hover:-translate-y-0.5 disabled:transform-none`}
              onClick={handleClassify}
              disabled={isLoading || !text.trim()}
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="w-5 h-5 border-2 border-t-transparent border-white rounded-full animate-spin mr-2"></div>
                  <span>Analyzing...</span>
                </div>
              ) : (
                'Analyze Bias & Sentiment'
              )}
            </button>
          </div>
        </div>
      </div>
    </AnimatedContainer>
  );
}