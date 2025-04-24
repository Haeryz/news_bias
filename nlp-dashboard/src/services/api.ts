import { ClassificationResult } from '@/types';
import { classifyTextProxy } from './classify';

export const classifyTextService = async (text: string): Promise<ClassificationResult> => {
  try {
    const data = await classifyTextProxy(text);

    const labelMap: { [key: number]: string } = {
      0: 'Republik',
      1: 'Demokrat',
      2: 'Netral',
      3: 'Others',
    };

    return {
      label: labelMap[data.label] || 'Unknown',
      confidence: data.confidence || undefined,
    };
  } catch (error) {
    console.error('Classification error:', error);
    throw new Error('Failed to classify text');
  }
};