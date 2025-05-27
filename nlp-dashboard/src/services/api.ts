import { ClassificationResult } from '@/types';
import { classifyTextProxy, ClassifyOptions } from './classify';

export const classifyTextService = async (
  text: string, 
  options: ClassifyOptions = {}
): Promise<ClassificationResult> => {
  try {
    const data = await classifyTextProxy(text, options);

    // Return the full data structure as received from the backend
    // The backend should now return the complete analysis
    return data;
  } catch (error) {
    console.error('Classification error:', error);
    throw new Error('Failed to classify text');
  }
};