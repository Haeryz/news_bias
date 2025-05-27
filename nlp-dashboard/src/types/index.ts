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