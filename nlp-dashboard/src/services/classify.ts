import { ClassificationResult } from '@/types';

export const classifyTextProxy = async (text: string): Promise<ClassificationResult> => {
    try {
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
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