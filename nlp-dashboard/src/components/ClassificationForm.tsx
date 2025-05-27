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
  
  // LIME options
  const [includeLime, setIncludeLime] = useState(false);
  const [fastMode, setFastMode] = useState(true);
  const [numFeatures, setNumFeatures] = useState(5);

  const handleClassify = async () => {
    if (!text.trim()) return;
    
    setIsSubmitting(true);
    setIsLoading(true);
    try {
      const result = await classifyTextService(text, {
        includeLime,
        fastMode,
        numFeatures
      });
      onResult(result);
    } catch (error) {
      console.error('Error classifying text:', error);
    } finally {
      setIsLoading(false);
      setTimeout(() => setIsSubmitting(false), 300); // Reset animation state
    }
  };  return (
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
          
          {/* LIME Options Panel */}
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg border border-gray-200 dark:border-gray-600">
            <h3 className="text-lg font-medium mb-3 text-gray-800 dark:text-gray-200">
              Analysis Options
            </h3>
            
            {/* LIME Toggle */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  id="includeLime"
                  checked={includeLime}
                  onChange={(e) => setIncludeLime(e.target.checked)}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                />
                <label htmlFor="includeLime" className="text-sm font-medium text-gray-800 dark:text-gray-200">
                  Enable LIME Explanations
                </label>
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {includeLime ? '⚠️ Slower (~30-60s)' : '⚡ Fast (~2-5s)'}
              </div>
            </div>
            
            {/* LIME Advanced Options */}
            {includeLime && (
              <div className="space-y-3 pl-4 border-l-2 border-blue-200 dark:border-blue-600">
                <div className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    id="fastMode"
                    checked={fastMode}
                    onChange={(e) => setFastMode(e.target.checked)}
                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                  />
                  <label htmlFor="fastMode" className="text-sm text-gray-700 dark:text-gray-300">
                    Fast Mode (20 samples vs 50)
                  </label>
                </div>
                
                <div className="flex items-center space-x-3">
                  <label htmlFor="numFeatures" className="text-sm text-gray-700 dark:text-gray-300 min-w-fit">
                    Features to analyze:
                  </label>
                  <select
                    id="numFeatures"
                    value={numFeatures}
                    onChange={(e) => setNumFeatures(parseInt(e.target.value))}
                    className="px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-600 text-gray-800 dark:text-gray-200"
                  >
                    <option value={3}>3 (Fastest)</option>
                    <option value={5}>5 (Balanced)</option>
                    <option value={8}>8 (Detailed)</option>
                    <option value={10}>10 (Most Detailed)</option>
                  </select>
                </div>
              </div>
            )}
            
            <div className="mt-3 text-xs text-gray-500 dark:text-gray-400">
              💡 LIME explanations show which words most influenced the bias classification
            </div>
          </div>
          
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
                  <span>{includeLime ? 'Generating LIME...' : 'Analyzing...'}</span>
                </div>
              ) : (
                `Analyze ${includeLime ? 'with LIME' : 'Bias & Sentiment'}`
              )}
            </button>
          </div>
        </div>
      </div>
    </AnimatedContainer>
  );
}