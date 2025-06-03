export interface ConfidenceScores {
  label_0: number;
  label_1: number;
  label_2: number;
  label_3: number;
}

export interface SentimentScores {
  positive: number;
  neutral: number;
  negative: number;
}

export interface Sentiment {
  label: string;
  scores: SentimentScores;
}

export interface HighlightedPhrase {
  phrase: string;
  explanation: string;
  sentiment: string;
  confidence_scores: SentimentScores;
}

// LIME Explanation interfaces
export interface HighlightedWord {
  word: string;
  importance: number;
  highlight_level: string;
  clean_word: string;
}

export interface KeyInfluences {
  supporting: Array<[string, number]>;
  opposing: Array<[string, number]>;
}

export interface LimeExplanation {
  original_text: string;
  predicted_class: number;
  predicted_label: string;
  confidence: number;
  simple_explanation: string;
  highlighted_text: HighlightedWord[];
  key_influences: KeyInfluences;
  all_probabilities: Record<string, number>;
  explanation_quality: string;
}

export interface ClassificationResult {
  label: number;
  label_meaning: string;
  confidence: number;
  confidence_scores: ConfidenceScores;
  bias_score: number;
  sentiment?: Sentiment;
  highlighted_phrases?: HighlightedPhrase[];
  key_phrases?: string[];
  // LIME-specific fields
  has_explanation?: boolean;
  explanation?: LimeExplanation;
  sentiment_analysis_error?: string;
}

// Batch processing types
export interface BatchProcessingOptions {
  expectedLabels?: boolean;
  hasHeader?: boolean;
  saveReport?: boolean;
  exportResults?: boolean;
}

export interface BatchResultRecord {
  content: string;
  label?: number;
  predicted_label: number;
  predicted_meaning: string;
  expected_meaning?: string;
  bias_score: number;
  processing_time_ms: number;
  prediction_status?: 'correct' | 'incorrect' | 'missing';
  [key: string]: string | number | boolean | undefined; // For probability columns
}

export interface ConfusionMatrix {
  matrix: number[];
  labels: string[];
}

export interface ClassMetric {
  label: number;
  label_name: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
}

export interface ClassificationReport {
  precision: Record<string, number>;
  recall: Record<string, number>;
  f1_score: Record<string, number>;
  support: Record<string, number>;
  accuracy?: number;
  macro_avg?: Record<string, number>;
  weighted_avg?: Record<string, number>;
}

export interface EvaluationMetrics {
  accuracy: number;
  classification_report: ClassificationReport;
  confusion_matrix: ConfusionMatrix;
  class_metrics: ClassMetric[];
  total_samples: number;
  correct_predictions: number;
  incorrect_predictions: number;
  incorrect_examples: BatchResultRecord[];
  avg_precision: number;
  avg_recall: number;
  avg_f1: number;
  confusion_matrix_img?: string;
  class_distribution_img?: string;
}

export interface ProcessingStats {
  avg_processing_time_ms: number;
  min_processing_time_ms: number;
  max_processing_time_ms: number;
  total_batch_time_seconds: number;
}

export interface BatchClassificationResult {
  total_processed: number;
  successful_predictions: number;
  failed_predictions: number;
  processing_stats: ProcessingStats;
  results: BatchResultRecord[];
  evaluation?: EvaluationMetrics;
  report_path?: string;
  exported_csv?: string;
  report_error?: string;
}