import os
import time
import requests
from typing import Dict, Any, Optional, Tuple
import json
import base64
from fastapi import UploadFile, HTTPException

class AzurePDFParser:
    """
    Class to handle PDF parsing using Azure Cognitive Services Document Intelligence API
    """
    
    def __init__(self, api_key: str, endpoint: str):
        """
        Initialize the PDF parser with Azure credentials
        
        Args:
            api_key (str): Azure Cognitive Services API key
            endpoint (str): Azure Cognitive Services endpoint (e.g., https://your-resource-name.cognitiveservices.azure.com)
        """
        self.api_key = api_key
        self.endpoint = endpoint.rstrip('/')  # Remove trailing slash if present
        self.headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/json"
        }
        self.api_version = "2024-11-30"
    
    async def submit_document(self, file: UploadFile) -> str:
        """
        Submit a PDF document to Azure for processing (Step 1)
        
        Args:
            file (UploadFile): The uploaded PDF file
            
        Returns:
            str: The operation ID to be used for retrieving results
        """
        # Read the file content
        file_content = await file.read()
        
        # Convert to base64
        encoded_content = base64.b64encode(file_content).decode('utf-8')
        
        # Create the request body
        request_body = {
            "base64Source": encoded_content
        }
        
        # Create the URL for the analyze operation
        url = f"{self.endpoint}/documentintelligence/documentModels/prebuilt-read:analyze?api-version={self.api_version}"
        
        try:
            # Send the POST request
            # Note: Use appropriate async library (like httpx) or run sync requests in executor for FastAPI
            # Using requests here for simplicity, assuming it's run appropriately in an async context
            response = requests.post(url, headers=self.headers, json=request_body)
            response.raise_for_status()  # Raise exception for 4XX/5XX status codes
            
            # Get the operation ID from the Operation-Location header
            operation_location = response.headers.get("Operation-Location")
            if not operation_location:
                raise HTTPException(status_code=500, detail="Failed to get operation location from Azure")
            
            # --- FIX IS HERE ---
            # Extract operation ID from the URL, removing the query string
            operation_id_with_query = operation_location.split('/')[-1]
            operation_id = operation_id_with_query.split('?')[0] # Get only the part before the '?'
            # --- END FIX ---

            return operation_id
            
        except requests.exceptions.RequestException as e:
            # Consider more specific error handling/logging if needed
            error_detail = f"Error submitting document to Azure: {response.status_code} {response.text}" if 'response' in locals() else str(e)
            raise HTTPException(status_code=500, detail=error_detail)
    
    def get_document_results(self, operation_id: str, max_retries: int = 10, retry_delay: int = 1) -> Dict[str, Any]:
        """
        Retrieve the document processing results from Azure (Step 2)
        
        Args:
            operation_id (str): The operation ID from the submission step
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Delay between retries in seconds
            
        Returns:
            Dict[str, Any]: The processing results with extracted content
        """
        # Create the URL for getting results
        url = f"{self.endpoint}/documentintelligence/documentModels/prebuilt-read/analyzeResults/{operation_id}?api-version={self.api_version}"
        
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
                    error_details = result.get("analyzeResult", {}).get("errors", [])
                    error_message = error_details[0].get("message") if error_details else "Unknown error"
                    raise HTTPException(status_code=500, detail=f"Document processing failed: {error_message}")
                    
                # If still running, wait and try again
                retry_count += 1
                time.sleep(retry_delay)
                
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=500, detail=f"Error retrieving document results from Azure: {str(e)}")
                
        # If we've exhausted retries
        raise HTTPException(status_code=408, detail="Document processing timed out")
    
    def extract_content(self, result: Dict[str, Any]) -> str:
        """
        Extract just the content from the Azure Document Intelligence results
        
        Args:
            result (Dict[str, Any]): The full result from Azure
            
        Returns:
            str: The extracted text content
        """
        try:
            content = result.get("analyzeResult", {}).get("content", "")
            return content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting content from results: {str(e)}")
            
    async def process_pdf(self, file: UploadFile) -> str:
        """
        Process a PDF file through the complete pipeline: submit and get results
        
        Args:
            file (UploadFile): The uploaded PDF file
            
        Returns:
            str: The extracted text content
        """
        # Step 1: Submit the document
        operation_id = await self.submit_document(file)
        
        # Step 2: Wait for processing and get results
        result = self.get_document_results(operation_id)
        
        # Step 3: Extract just the content
        content = self.extract_content(result)
        
        return content


# Helper function to create parser with credentials from environment variables
def create_pdf_parser() -> AzurePDFParser:
    """
    Create a PDF parser instance using credentials from environment variables
    
    Returns:
        AzurePDFParser: Configured PDF parser instance
    """
    api_key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", 
                             "https://mamamnk-smalekom.cognitiveservices.azure.com")
    
    if not api_key:
        raise ValueError("AZURE_DOCUMENT_INTELLIGENCE_API_KEY environment variable is required")
        
    return AzurePDFParser(api_key=api_key, endpoint=endpoint)