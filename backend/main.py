# backend/main.py
# --- CORRECTED VERSION USING build_wine_model() and load_weights() ---
import os
import re
import numpy as np
import tensorflow as tf
import keras # Import Keras separately
from keras import layers # Import layers from Keras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

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

# --- Helper Functions (clean_text) ---
def clean_text(text: str) -> str:
    text = re.sub(r'[\'""]+', '"', text)
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
        predicted_label = tf.argmax(outputs.logits, axis=1).numpy()[0]
        return {"label": int(predicted_label)}
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

# --- Main block (for local Uvicorn execution) ---
if __name__ == "__main__":
    import uvicorn
    print("--- Starting FastAPI server locally with Uvicorn ---")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)