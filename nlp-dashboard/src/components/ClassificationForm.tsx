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
    <AnimatedContainer className="w-full max-w-md">
      <div className={`flex flex-col items-center space-y-4 transition-transform duration-300 ${isSubmitting ? 'scale-95 opacity-90' : 'scale-100 opacity-100'}`}>
        <textarea
          className="w-full p-4 border border-gray-200 dark:border-gray-700 rounded-lg 
            bg-white dark:bg-gray-800 shadow-lg focus:ring-2 focus:ring-blue-500 
            focus:border-blue-500 outline-none min-h-[150px] resize-none 
            transition-all duration-300 text-gray-800 dark:text-gray-200"
          placeholder="Enter text to analyze..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button
          className={`px-8 py-3 rounded-lg font-semibold transition-all duration-300
            ${isLoading ? 'bg-gray-400 dark:bg-gray-600 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white'}
            shadow-md hover:shadow-lg transform hover:-translate-y-0.5`}
          onClick={handleClassify}
          disabled={isLoading || !text.trim()}
        >
          {isLoading ? (
            <div className="flex items-center justify-center">
              <div className="w-5 h-5 border-2 border-t-transparent border-white rounded-full animate-spin mr-2"></div>
              <span>Processing...</span>
            </div>
          ) : (
            'Analyze Text'
          )}
        </button>
      </div>
    </AnimatedContainer>
  );
}