import os
import time
import requests
from typing import Dict, Any, List, Optional, Tuple
from fastapi import HTTPException
import re

class AzureLanguageService:
    """
    Class to handle Azure AI Language service for sentiment analysis and key phrase extraction
    """
    
    def __init__(self, api_key: str, endpoint: str):
        """
        Initialize the Azure Language Service client
        
        Args:
            api_key (str): Azure AI Language API key
            endpoint (str): Azure AI Language endpoint
        """
        self.api_key = api_key
        self.endpoint = endpoint.rstrip('/')  # Remove trailing slash if present
        self.headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/json"
        }
        self.api_version = "2023-04-01"
        
        # List of emotionally charged or biased terms to flag
        self.biased_terms = {
            # Political terms
            "radical": "Political label that can be dismissive",
            "extremist": "Label that dismisses opposing views",
            "socialist": "Often used as a politically charged label",
            "fascist": "Strong political accusation",
            "communist": "Often used as a politically charged label",
            "leftist": "Political label that can be dismissive",
            "right-wing": "Political label that can be dismissive",
            
            # Crisis/emergency terms
            "crisis": "Emotionally charged term suggesting emergency",
            "disaster": "Emotionally charged term suggesting catastrophe",
            "catastrophe": "Emotionally charged term suggesting extreme failure",
            "emergency": "Suggests urgent, critical situation",
            "chaos": "Suggests complete disorder and confusion",
            
            # Loaded adjectives
            "shocking": "Emotionally charged term suggesting surprise/outrage",
            "outrageous": "Strongly emotional term expressing indignation",
            "scandalous": "Implies moral wrongdoing",
            "disastrous": "Implies severe negative outcome",
            "horrific": "Extremely emotionally charged",
            "terrifying": "Appeals to fear",
            "devastating": "Implies severe destruction",
            
            # Absolutist terms
            "always": "Absolutist term that rarely applies",
            "never": "Absolutist term that rarely applies",
            "every": "Generalization that may not be accurate",
            "all": "Generalization that may not be accurate",
            "none": "Absolutist term that rarely applies",
            
            # Dehumanizing terms
            "thug": "Dehumanizing term with racial connotations",
            "illegals": "Dehumanizing term for undocumented immigrants",
            "animals": "Dehumanizing term when referring to people",
            
            # Politically charged phrases
            "fake news": "Dismissive term for information someone disagrees with",
            "mainstream media": "Often used to discredit reporting",
            "liberal agenda": "Suggests conspiracy rather than legitimate political positions",
            "conservative agenda": "Suggests conspiracy rather than legitimate political positions",
            "radical left": "Political label that can be dismissive",
            "alt-right": "Political label with extremist connotations",
        }
    
    def submit_for_analysis(self, text: str) -> str:
        """
        Submit text to Azure AI Language for sentiment analysis and key phrase extraction (Step 1)
        
        Args:
            text (str): The text to analyze
            
        Returns:
            str: The operation ID to be used for retrieving results
        """
        # Create the URL for the analyze operation
        url = f"{self.endpoint}/language/analyze-text/jobs?api-version={self.api_version}"
        
        # Create the request body
        request_body = {
            "displayName": "News Bias Analysis",
            "analysisInput": {
                "documents": [
                    {
                        "id": "doc1",
                        "language": "en",
                        "text": text
                    }
                ]
            },
            "tasks": [
                {
                    "kind": "SentimentAnalysis",
                    "parameters": {
                        "modelVersion": "latest",
                        "opinionMining": True
                    }
                },
                {
                    "kind": "KeyPhraseExtraction",
                    "parameters": {
                        "modelVersion": "latest"
                    }
                }
            ]
        }
        
        try:
            # Send the POST request
            response = requests.post(url, headers=self.headers, json=request_body)
            response.raise_for_status()  # Raise exception for 4XX/5XX status codes
            
            # Get the operation ID from the Operation-Location header
            operation_location = response.headers.get("Operation-Location")
            if not operation_location:
                raise HTTPException(status_code=500, detail="Failed to get operation location from Azure")
            
            # Extract operation ID from the URL
            operation_id = operation_location.split('/')[-1].split('?')[0]
            return operation_id
            
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error submitting text to Azure Language: {str(e)}")
    
    def get_analysis_results(self, operation_id: str, max_retries: int = 10, retry_delay: int = 1) -> Dict[str, Any]:
        """
        Retrieve the text analysis results from Azure (Step 2)
        
        Args:
            operation_id (str): The operation ID from the submission step
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Delay between retries in seconds
            
        Returns:
            Dict[str, Any]: The analysis results with sentiment and key phrases
        """
        # Create the URL for getting results
        url = f"{self.endpoint}/language/analyze-text/jobs/{operation_id}?api-version={self.api_version}"
        
        # Initialize retry counter
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Send the GET request
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()  # Raise exception for 4XX/5XX status codes
                
                # Parse the response
                result = response.json()
                
                # Check if processing is complete
                status = result.get("status")
                
                if status == "succeeded":
                    return result
                
                elif status == "failed":
                    error_details = result.get("errors", [])
                    error_message = error_details[0].get("message") if error_details else "Unknown error"
                    raise HTTPException(status_code=500, detail=f"Text analysis failed: {error_message}")
                    
                # If still running, wait and try again
                retry_count += 1
                time.sleep(retry_delay)
                
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=500, detail=f"Error retrieving analysis results from Azure: {str(e)}")
                
        # If we've exhausted retries
        raise HTTPException(status_code=408, detail="Text analysis timed out")
    
    def extract_highlighted_phrases(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and highlight potentially biased phrases from the text analysis results
        
        Args:
            result (Dict[str, Any]): The full result from Azure Language
            
        Returns:
            List[Dict[str, Any]]: List of highlighted phrases with explanations
        """
        highlighted_phrases = []
        
        try:
            # Extract key phrases
            key_phrase_results = next((task for task in result.get("tasks", {}).get("items", []) 
                                      if task.get("kind") == "KeyPhraseExtractionLROResults"), None)
            
            key_phrases = []
            if key_phrase_results and key_phrase_results.get("status") == "succeeded":
                doc_results = key_phrase_results.get("results", {}).get("documents", [])
                if doc_results:
                    key_phrases = doc_results[0].get("keyPhrases", [])
            
            # Extract sentiment analysis
            sentiment_results = next((task for task in result.get("tasks", {}).get("items", []) 
                                     if task.get("kind") == "SentimentAnalysisLROResults"), None)
            
            sentences = []
            if sentiment_results and sentiment_results.get("status") == "succeeded":
                doc_results = sentiment_results.get("results", {}).get("documents", [])
                if doc_results:
                    sentences = doc_results[0].get("sentences", [])
            
            # Check for biased terms in key phrases
            for phrase in key_phrases:
                lower_phrase = phrase.lower()
                
                # Check if the phrase contains any biased terms
                for term, explanation in self.biased_terms.items():
                    if re.search(r'\b' + re.escape(term) + r'\b', lower_phrase):
                        # Find the sentiment for this phrase if possible
                        sentiment_info = "unknown"
                        confidence = {}
                        
                        # Try to find a matching sentence containing this phrase
                        for sentence in sentences:
                            if phrase.lower() in sentence.get("text", "").lower():
                                sentiment_info = sentence.get("sentiment", "unknown")
                                confidence = sentence.get("confidenceScores", {})
                                break
                        
                        highlighted_phrases.append({
                            "phrase": phrase,
                            "explanation": f"Loaded term: '{term}' - {explanation}",
                            "sentiment": sentiment_info,
                            "confidence_scores": confidence
                        })
                        break
            
            # Check for negative/positive sentiment sentences
            for sentence in sentences:
                text = sentence.get("text", "")
                sentiment = sentence.get("sentiment", "")
                scores = sentence.get("confidenceScores", {})
                
                # Flag strongly negative or positive sentences
                if (sentiment == "negative" and scores.get("negative", 0) > 0.75) or \
                   (sentiment == "positive" and scores.get("positive", 0) > 0.75):
                    
                    # Check if we already added this phrase
                    if not any(p.get("phrase") == text for p in highlighted_phrases):
                        highlighted_phrases.append({
                            "phrase": text,
                            "explanation": f"Strong {sentiment} sentiment detected",
                            "sentiment": sentiment,
                            "confidence_scores": scores
                        })
            
            return highlighted_phrases
            
        except Exception as e:
            print(f"Error extracting highlighted phrases: {str(e)}")
            return []
            
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text through the complete pipeline: submit and get results
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict[str, Any]: Analysis results with sentiment, key phrases, and highlighted phrases
        """
        # Step 1: Submit the text for analysis
        operation_id = self.submit_for_analysis(text)
        
        # Step 2: Wait for processing and get results
        result = self.get_analysis_results(operation_id)
        
        # Step 3: Extract highlighted phrases
        highlighted_phrases = self.extract_highlighted_phrases(result)
        
        # Step 4: Package the results
        final_result = {
            "sentiment": {},
            "key_phrases": [],
            "highlighted_phrases": highlighted_phrases,
            "raw_result": result  # Include raw result for debugging
        }
        
        # Extract overall sentiment
        sentiment_results = next((task for task in result.get("tasks", {}).get("items", []) 
                                 if task.get("kind") == "SentimentAnalysisLROResults"), None)
        
        if sentiment_results and sentiment_results.get("status") == "succeeded":
            doc_results = sentiment_results.get("results", {}).get("documents", [])
            if doc_results:
                final_result["sentiment"] = {
                    "label": doc_results[0].get("sentiment", ""),
                    "scores": doc_results[0].get("confidenceScores", {})
                }
        
        # Extract key phrases
        key_phrase_results = next((task for task in result.get("tasks", {}).get("items", []) 
                                  if task.get("kind") == "KeyPhraseExtractionLROResults"), None)
        
        if key_phrase_results and key_phrase_results.get("status") == "succeeded":
            doc_results = key_phrase_results.get("results", {}).get("documents", [])
            if doc_results:
                final_result["key_phrases"] = doc_results[0].get("keyPhrases", [])
        
        return final_result


# Helper function to create language service with credentials from environment variables
def create_language_service() -> AzureLanguageService:
    """
    Create an Azure Language Service instance using credentials from environment variables
    
    Returns:
        AzureLanguageService: Configured language service instance
    """
    api_key = os.environ.get("AZURE_LANGUAGE_API_KEY")
    endpoint = os.environ.get("AZURE_LANGUAGE_ENDPOINT", 
                             "https://aingmacanauuuu.cognitiveservices.azure.com")
    
    if not api_key:
        raise ValueError("AZURE_LANGUAGE_API_KEY environment variable is required")
        
    return AzureLanguageService(api_key=api_key, endpoint=endpoint)