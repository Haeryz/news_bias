import { ClassificationResult } from '@/types';
import { classifyTextProxy } from './classify';

export const classifyTextService = async (text: string): Promise<ClassificationResult> => {
  try {
    const data = await classifyTextProxy(text);

    // Return the full data structure as received from the backend
    // The backend should now return the complete analysis
    return data;
  } catch (error) {
    console.error('Classification error:', error);
    throw new Error('Failed to classify text');
  }
};