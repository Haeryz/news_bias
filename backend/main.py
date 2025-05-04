# backend/main.py
# --- CORRECTED VERSION USING build_wine_model() and load_weights() ---
import os
import re
import numpy as np
import tensorflow as tf
import keras # Import Keras separately
from keras import layers # Import layers from Keras
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from PDF_parser import create_pdf_parser  # Import the PDF parser
from sentiment import create_language_service  # Import the language service
from dotenv import load_dotenv  # Import for loading environment variables

# Load environment variables from .env file
load_dotenv()

# --- App Initialization ---
app = FastAPI(
    title="Multi-Model API (News Bias & Wine Quality)",
    description="Classifies news articles for bias and predicts wine quality.",
    version="1.1.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev, adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize PDF Parser and Language Service ---
try:
    pdf_parser = create_pdf_parser()
    print("✅ PDF parser initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize PDF parser: {e}")
    pdf_parser = None

try:
    language_service = create_language_service()
    print("✅ Azure Language service initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize Azure Language service: {e}")
    language_service = None

# --- News Bias Classification Model Loading ---
BIAS_MODEL_DIR = "bias_classification_model"
try:
    bias_model = TFDistilBertForSequenceClassification.from_pretrained(BIAS_MODEL_DIR)
    bias_tokenizer = DistilBertTokenizer.from_pretrained(BIAS_MODEL_DIR)
    print("✅ News Bias model and tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load News Bias model or tokenizer: {e}")
    bias_model = None
    bias_tokenizer = None

# --- Wine Quality Prediction Model - Architecture Definition & Weights Loading ---
WINE_WEIGHTS_FILENAME = 'wine_quality_tf_model.weights.h5' # Correct weights filename
WINE_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), WINE_WEIGHTS_FILENAME)

# Define the expected order of features for the wine model
WINE_FEATURE_ORDER = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'chlorides',
    'free_sulfur_dioxide', 'density', 'alcohol', 'type_white'
]
N_WINE_FEATURES = len(WINE_FEATURE_ORDER) # Should be 8

def build_wine_model(n_features: int):
    """Builds the Keras model architecture (must match the trained model)."""
    # This architecture MUST EXACTLY MATCH Cell 5 of your notebook
    model = keras.Sequential(
        [
            layers.InputLayer(shape=(n_features,), name='input_layer'),
            layers.Dense(16, activation='relu', name='hidden_layer_1'),
            layers.Dropout(0.3, name='dropout_1'),
            # Add any other layers exactly as they were in Cell 5
            layers.Dense(1, name='output_layer')
        ],
        name="wine_quality_regressor_rebuilt"
    )
    print("-> Wine model architecture defined.")
    # Optionally build the model explicitly (helps catch shape errors early)
    # model.build(input_shape=(None, n_features))
    # model.summary() # Optional: Print summary to verify architecture
    return model

wine_model = None
try:
    if os.path.exists(WINE_WEIGHTS_PATH):
        print(f"-> Attempting to build wine model architecture...")
        # 1. Create the architecture first
        wine_model = build_wine_model(N_WINE_FEATURES)

        print(f"-> Attempting to load weights from: {WINE_WEIGHTS_PATH}")
        # 2. Load the weights into the architecture
        wine_model.load_weights(WINE_WEIGHTS_PATH) # Use load_weights() !

        print(f"✅ Wine Quality model architecture built and weights loaded successfully!")
    else:
        print(f"⚠️ Wine Quality model weights file not found at {WINE_WEIGHTS_PATH}. Prediction endpoint will be unavailable.")
        wine_model = None
except Exception as e:
    print(f"❌ Failed to build Wine Quality model or load weights: {e}")
    import traceback
    traceback.print_exc() # Print full error details
    wine_model = None

# --- Define bias label meanings dictionary ---
BIAS_LABEL_MEANINGS = {
    0: "Republican",
    1: "Liberal",
    2: "Neutral",
    3: "Other"  # Adjust these meanings based on your actual labels
}

# --- Helper Functions (clean_text) ---
def clean_text(text: str) -> str:
    text = re.sub(r'[\'""]+', '"', text)
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Function to calculate bias score on a scale of 0-100 ---
def calculate_bias_score(logits):
    """
    Calculate a bias score from 0-100 based on the model's output logits.
    Higher scores indicate stronger bias (either left or right).
    
    Args:
        logits: The raw output logits from the model
        
    Returns:
        Tuple containing (bias_score, probabilities array)
    """
    # Convert logits to probabilities using softmax
    probabilities = tf.nn.softmax(logits).numpy()
    
    # Determine how many classes we have
    num_classes = len(probabilities)
    
    if num_classes == 3:  # If we have Left (0), Center (1), Right (2)
        # Calculate bias score: higher weight to extreme positions
        # Center (index 1) gets low bias score
        # Left and right get higher scores based on their probability
        left_prob = probabilities[0]
        center_prob = probabilities[1]
        right_prob = probabilities[2]
        
        # Calculate left and right bias strength
        left_bias = left_prob * 100  # Scaled to 0-100
        right_bias = right_prob * 100  # Scaled to 0-100
        
        # Take the maximum bias (either left or right)
        bias_score = max(left_bias, right_bias)
        
        # Reduce score if there's a strong center probability
        bias_score = bias_score * (1 - (center_prob * 0.5))  # Center reduces bias
        
    elif num_classes == 2:  # If we have Neutral (0) and Biased (1)
        # Simpler case - bias score is just the probability of the "biased" class
        bias_score = probabilities[1] * 100
    
    else:  # For any other number of classes, use distance from center
        # Assume classes are ordered from left to right on political spectrum
        mid_point = (num_classes - 1) / 2
        weighted_sum = 0
        
        for i, prob in enumerate(probabilities):
            # Calculate distance from center position
            distance = abs(i - mid_point) / mid_point  # Normalized to 0-1
            weighted_sum += distance * prob
            
        # Scale to 0-100
        bias_score = weighted_sum * 100
    
    return round(bias_score, 1), probabilities

# --- Pydantic Models (NewsTextInput, WineFeaturesInput) ---
class NewsTextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="The news article text to classify.")

class WineFeaturesInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    chlorides: float
    free_sulfur_dioxide: float
    density: float
    alcohol: float
    type_white: float = Field(..., example=1.0, description="Typically 1.0 for white, 0.0 for red. MUST BE SCALED if original data was scaled.")
    model_config = { "json_schema_extra": { "examples": [ { "fixed_acidity": 0.5, "volatile_acidity": -0.2, "citric_acid": 0.1, "chlorides": -1.5, "free_sulfur_dioxide": 0.8, "density": 0.0, "alcohol": 1.2, "type_white": 1.0 } ] } } # Example with SCALED values

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Multi-Model API. Use /docs for details."}

@app.post("/classify", tags=["News Bias"])
async def classify_news_text(input: NewsTextInput):
    if not bias_model or not bias_tokenizer:
        raise HTTPException(status_code=503, detail="News Bias model is not available.")
    try:
        cleaned_text = clean_text(input.text)
        inputs = bias_tokenizer(cleaned_text, return_tensors="tf", padding="max_length", truncation=True, max_length=512)
        outputs = bias_model(inputs)
        
        # Get raw logits
        logits = outputs.logits.numpy()[0]
        
        # Calculate bias score and get probabilities
        bias_score, probabilities = calculate_bias_score(logits)
        
        # Get predicted label
        predicted_label = int(tf.argmax(logits, axis=0).numpy())
        
        # Format probabilities as percentages
        confidence_scores = {
            f"label_{i}": float(round(prob * 100, 2)) for i, prob in enumerate(probabilities)
        }
        
        # Create response dict with bias classification results
        response = {
            "label": predicted_label,
            "label_meaning": BIAS_LABEL_MEANINGS.get(predicted_label, "Unknown"),
            "confidence": float(round(probabilities[predicted_label] * 100, 2)),
            "confidence_scores": confidence_scores,
            "bias_score": bias_score
        }
        
        # Add sentiment analysis if language service is available
        if language_service:
            try:
                # Run Azure Language analysis on the text
                language_analysis = language_service.analyze_text(input.text)
                
                # Add sentiment results to response
                response["sentiment"] = language_analysis.get("sentiment", {})
                
                # Add highlighted phrases (biased/emotionally charged language)
                highlighted_phrases = language_analysis.get("highlighted_phrases", [])
                response["highlighted_phrases"] = highlighted_phrases
                
                # Only include key phrases if there are some
                if language_analysis.get("key_phrases"):
                    response["key_phrases"] = language_analysis.get("key_phrases", [])
                    
            except Exception as e:
                print(f"❌ Azure Language analysis failed, but continuing with bias analysis: {e}")
                # Don't fail the entire request if just the sentiment analysis fails
                response["sentiment_analysis_error"] = str(e)
        
        return response
    except Exception as e:
        print(f"❌ Error during news classification: {e}")
        raise HTTPException(status_code=500, detail=f"News classification prediction failed: {str(e)}")

@app.post("/predict_wine", tags=["Wine Quality"])
async def predict_wine_quality(features: WineFeaturesInput):
    if not wine_model:
        raise HTTPException(status_code=503, detail="Wine Quality model is not available.")
    try:
        feature_values = [getattr(features, col) for col in WINE_FEATURE_ORDER]
        input_data = np.array([feature_values], dtype=np.float32)
        print(f"Input data shape for wine prediction: {input_data.shape}")
        print(f"Input wine data sample: {input_data}")
        # Use the rebuilt model with loaded weights for prediction
        prediction_result = wine_model.predict(input_data)
        print(f"Raw wine prediction result: {prediction_result}")
        predicted_quality = float(prediction_result[0][0])
        return {"predicted_quality": predicted_quality}
    except Exception as e:
        print(f"❌ Error during wine prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Wine quality prediction failed: {str(e)}")

@app.post("/upload_pdf", tags=["PDF Processing"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file for processing through Azure Document Intelligence API.
    The API will extract the text and classify it for bias.
    """
    if not pdf_parser:
        raise HTTPException(status_code=503, detail="PDF parser is not available.")
    
    # Check if file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
        
    try:
        # Process the PDF using Azure Document Intelligence
        extracted_content = await pdf_parser.process_pdf(file)
        
        # Initialize response
        response = {
            "content": extracted_content,
        }
        
        # Classify the extracted text for news bias if available
        if bias_model and bias_tokenizer and extracted_content:
            cleaned_text = clean_text(extracted_content)
            inputs = bias_tokenizer(cleaned_text, return_tensors="tf", padding="max_length", truncation=True, max_length=512)
            outputs = bias_model(inputs)
            
            # Get raw logits
            logits = outputs.logits.numpy()[0]
            
            # Calculate bias score and get probabilities
            bias_score, probabilities = calculate_bias_score(logits)
            
            # Get predicted label
            predicted_label = int(tf.argmax(logits, axis=0).numpy())
            
            # Format probabilities as percentages
            confidence_scores = {
                f"label_{i}": float(round(prob * 100, 2)) for i, prob in enumerate(probabilities)
            }
            
            # Add classification results to response
            response["bias_label"] = predicted_label
            response["label_meaning"] = BIAS_LABEL_MEANINGS.get(predicted_label, "Unknown")
            response["confidence"] = float(round(probabilities[predicted_label] * 100, 2))
            response["confidence_scores"] = confidence_scores
            response["bias_score"] = bias_score
            
        # Add sentiment analysis if language service is available
        if language_service and extracted_content:
            try:
                # Run Azure Language analysis on the extracted text
                language_analysis = language_service.analyze_text(extracted_content)
                
                # Add sentiment results to response
                response["sentiment"] = language_analysis.get("sentiment", {})
                
                # Add highlighted phrases (biased/emotionally charged language)
                highlighted_phrases = language_analysis.get("highlighted_phrases", [])
                response["highlighted_phrases"] = highlighted_phrases
                
                # Only include key phrases if there are some
                if language_analysis.get("key_phrases"):
                    response["key_phrases"] = language_analysis.get("key_phrases", [])
                    
            except Exception as e:
                print(f"❌ Azure Language analysis failed, but continuing with bias analysis: {e}")
                # Don't fail the entire request if just the sentiment analysis fails
                response["sentiment_analysis_error"] = str(e)
            
        return response
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Error during PDF processing: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

# --- Main block (for local Uvicorn execution) ---
if __name__ == "__main__":
    import uvicorn
    print("--- Starting FastAPI server locally with Uvicorn ---")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)