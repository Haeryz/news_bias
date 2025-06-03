import { ClassificationResult, BatchClassificationResult, BatchProcessingOptions } from '@/types';
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

export const classifyBatchService = async (
  file: File,
  options: BatchProcessingOptions = {}
): Promise<BatchClassificationResult> => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('expectedLabels', String(options.expectedLabels || false));
    formData.append('hasHeader', String(options.hasHeader !== false)); // default true
    formData.append('saveReport', String(options.saveReport || false));
    formData.append('exportResults', String(options.exportResults || false));

    const response = await fetch('/api/classify-batch', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to process batch');
    }

    return await response.json();
  } catch (error) {
    console.error('Batch classification error:', error);
    throw new Error('Failed to classify batch');
  }
};