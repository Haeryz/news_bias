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

export interface ClassificationResult {
  label: number;
  label_meaning: string;
  confidence: number;
  confidence_scores: ConfidenceScores;
  bias_score: number;
  sentiment: Sentiment;
  highlighted_phrases: HighlightedPhrase[];
  key_phrases: string[];
}