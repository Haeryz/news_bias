# lime_explainer.py
"""
User-friendly LIME explanations for news bias classification.
Provides text highlighting and simple explanations instead of complex graphs.
"""

import numpy as np
import tensorflow as tf
from lime.lime_text import LimeTextExplainer
import re
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BiasExplainer:
    """
    Provides user-friendly explanations for bias classification using LIME.
    Designed for web applications with non-technical users.
    """
    
    def __init__(self, model, tokenizer, label_meanings: Dict[int, str]):
        """
        Initialize the explainer.
        
        Args:
            model: The trained bias classification model
            tokenizer: The tokenizer for the model
            label_meanings: Dictionary mapping label indices to human-readable names
        """
        self.model = model
        self.tokenizer = tokenizer
        self.label_meanings = label_meanings        # Initialize LIME explainer with performance optimizations
        self.explainer = LimeTextExplainer(
            class_names=list(label_meanings.values()),
            bow=False,  # Don't use bag of words (faster)
            split_expression=r'\W+',  # Simple word splitting
            random_state=42  # For consistent results
        )
        
        logger.info("✅ BiasExplainer initialized successfully")
    
    def clean_text(self, text: str) -> str:
        """Clean text for model input"""
        text = re.sub(r'[\'""]+', '"', text)
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict_probabilities(self, texts: List[str]) -> np.ndarray:
        """
        Predict bias probabilities for a list of texts.
        Required by LIME explainer.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            numpy array of shape (n_samples, n_classes) with probabilities
        """
        predictions = []
        
        for text in texts:
            try:
                cleaned_text = self.clean_text(text)
                inputs = self.tokenizer(
                    cleaned_text, 
                    return_tensors="tf", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=512
                )
                outputs = self.model(inputs)
                probabilities = tf.nn.softmax(outputs.logits).numpy()[0]
                predictions.append(probabilities)
            except Exception as e:
                logger.error(f"Error predicting text: {e}")
                # Return uniform probabilities as fallback
                n_classes = len(self.label_meanings)
                predictions.append(np.ones(n_classes) / n_classes)
        
        return np.array(predictions)
    
    def explain_prediction(self, text: str, num_features: int = 8, 
                          num_samples: int = 50) -> Dict:
        """
        Generate user-friendly explanation for a text classification.
        
        Args:
            text: Text to explain
            num_features: Number of most important features to analyze
            num_samples: Number of samples for LIME (fewer = faster)
            
        Returns:
            Dictionary with explanation data suitable for web display
        """
        try:
            # Get LIME explanation
            explanation = self.explainer.explain_instance(
                text, 
                self.predict_probabilities, 
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Get model prediction
            prediction = self.predict_probabilities([text])[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class] * 100
            
            # Extract feature importance for the predicted class
            feature_importance = explanation.as_list()
            
            # Separate positive and negative influences
            positive_words, negative_words = self._categorize_influences(feature_importance)
            
            # Create highlighted text data
            highlighted_text = self._create_highlighted_text(text, feature_importance)
              # Generate simple explanation
            simple_explanation = self._generate_simple_explanation(
                predicted_class, positive_words, negative_words
            )
            
            return {
                'original_text': text,
                'predicted_class': int(predicted_class),  # Convert numpy.int64 to int
                'predicted_label': self.label_meanings[predicted_class],
                'confidence': round(float(confidence), 1),  # Ensure float
                'simple_explanation': simple_explanation,
                'highlighted_text': highlighted_text,
                'key_influences': {
                    'supporting': positive_words[:5],  # Top 5 supporting words
                    'opposing': negative_words[:5]     # Top 5 opposing words
                },
                'all_probabilities': {
                    self.label_meanings[i]: round(float(prob * 100), 2) 
                    for i, prob in enumerate(prediction)
                },
                'explanation_quality': self._assess_explanation_quality(feature_importance)
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._fallback_explanation(text)
    
    def _categorize_influences(self, feature_importance: List[Tuple[str, float]]) -> Tuple[List, List]:
        """Separate words into positive and negative influences"""
        positive_words = []
        negative_words = []
        for word, importance in feature_importance:
            if importance > 0:
                positive_words.append((word, round(float(importance), 4)))
            else:
                negative_words.append((word, round(float(abs(importance)), 4)))
        
        # Sort by importance (descending)
        positive_words.sort(key=lambda x: x[1], reverse=True)
        negative_words.sort(key=lambda x: x[1], reverse=True)
        
        return positive_words, negative_words
    
    def _create_highlighted_text(self, text: str, feature_importance: List[Tuple[str, float]]) -> List[Dict]:
        """
        Create highlighted text data for frontend display.
        
        Returns list of dictionaries with word and highlight information.
        """
        # Create importance lookup
        importance_dict = {word.lower(): importance for word, importance in feature_importance}
        
        # Split text into words while preserving spacing
        words = text.split()
        highlighted = []
        
        for word in words:
            # Clean word for lookup (remove punctuation)
            clean_word = re.sub(r'[^\w]', '', word.lower())
            importance = importance_dict.get(clean_word, 0)
            
            # Determine highlight level based on importance thresholds
            highlight_level = self._get_highlight_level(importance)
            highlighted.append({
                'word': word,
                'importance': round(float(importance), 4),
                'highlight_level': highlight_level,
                'clean_word': clean_word
            })
        
        return highlighted
    
    def _get_highlight_level(self, importance: float) -> str:
        """Determine highlight level based on importance score"""
        if importance > 0.08:
            return 'high_positive'
        elif importance > 0.04:
            return 'medium_positive'
        elif importance > 0.01:
            return 'low_positive'
        elif importance < -0.08:
            return 'high_negative'
        elif importance < -0.04:
            return 'medium_negative'
        elif importance < -0.01:
            return 'low_negative'
        else:
            return 'neutral'
    
    def _generate_simple_explanation(self, predicted_class: int, 
                                   positive_words: List, negative_words: List) -> str:
        """Generate a simple, user-friendly explanation"""
        label = self.label_meanings[predicted_class]
        
        explanation_parts = []
        explanation_parts.append(f"The AI classified this text as '{label}'")
        
        if positive_words:
            top_positive = [word for word, _ in positive_words[:3]]
            if len(top_positive) == 1:
                explanation_parts.append(f"mainly because of the word '{top_positive[0]}'")
            else:
                explanation_parts.append(f"mainly because of words like '{', '.join(top_positive)}'")
        
        if negative_words and len(negative_words) > 0:
            top_negative = [word for word, _ in negative_words[:2]]
            if top_negative:
                explanation_parts.append(f"However, words like '{', '.join(top_negative)}' suggest otherwise")
        
        return ". ".join(explanation_parts) + "."
    
    def _assess_explanation_quality(self, feature_importance: List[Tuple[str, float]]) -> str:
        """Assess the quality/reliability of the explanation"""
        if not feature_importance:
            return "low"
        
        # Check if there are significant features
        max_importance = max(abs(importance) for _, importance in feature_importance)
        
        if max_importance > 0.1:
            return "high"
        elif max_importance > 0.05:
            return "medium"
        else:
            return "low"
    
    def _fallback_explanation(self, text: str) -> Dict:
        """Provide fallback explanation when LIME fails"""
        # Get basic prediction
        try:
            prediction = self.predict_probabilities([text])[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class] * 100
        except:
            predicted_class = 2  # Default to neutral
            confidence = 33.3
            prediction = np.ones(len(self.label_meanings)) / len(self.label_meanings)
        return {
            'original_text': text,
            'predicted_class': int(predicted_class),  # Convert numpy.int64 to int
            'predicted_label': self.label_meanings.get(predicted_class, "Unknown"),
            'confidence': round(float(confidence), 1),  # Ensure float
            'simple_explanation': "Unable to generate detailed explanation. The model made a prediction but explanation analysis failed.",
            'highlighted_text': [{'word': word, 'importance': 0, 'highlight_level': 'neutral'} 
                               for word in text.split()],
            'key_influences': {'supporting': [], 'opposing': []},
            'all_probabilities': {
                self.label_meanings.get(i, f"Class_{i}"): round(float(prob * 100), 2) 
                for i, prob in enumerate(prediction)
            },
            'explanation_quality': 'unavailable',
            'error': 'Explanation generation failed'
        }

def create_bias_explainer(model, tokenizer, label_meanings: Dict[int, str]) -> Optional[BiasExplainer]:
    """
    Factory function to create a BiasExplainer instance.
    
    Args:
        model: Trained bias classification model
        tokenizer: Model tokenizer
        label_meanings: Dictionary mapping class indices to labels
        
    Returns:
        BiasExplainer instance or None if creation fails
    """
    try:
        return BiasExplainer(model, tokenizer, label_meanings)
    except Exception as e:
        logger.error(f"Failed to create BiasExplainer: {e}")
        return None
