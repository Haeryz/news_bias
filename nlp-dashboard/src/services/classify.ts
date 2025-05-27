import { ClassificationResult } from '@/types';

export interface ClassifyOptions {
  includeLime?: boolean;
  fastMode?: boolean;
  numFeatures?: number;
}

export const classifyTextProxy = async (
  text: string, 
  options: ClassifyOptions = {}
): Promise<ClassificationResult> => {
    try {
      const { includeLime = false, fastMode = true, numFeatures = 5 } = options;
      
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          text, 
          includeLime, 
          fastMode, 
          numFeatures 
        }),
      });
  
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
  
      return await response.json();
    } catch (error) {
      console.error('Proxy classification error:', error);
      throw new Error('Failed to classify text via proxy');
    }
  };