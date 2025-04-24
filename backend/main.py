# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import re

app = FastAPI(
    title="News Bias Classification API",
    description="Classifies news articles into bias categories (0-3) with pre-trained DistilBERT.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-app.vercel.app",  # Replace with your Next.js app URL
        "http://localhost:3000",       # For local Next.js development
        "http://localhost:5173",       # For local Next.js development
        # Add other frontend URLs if needed
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

MODEL_DIR = "bias_classification_model"
try:
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {e}")

class TextInput(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=10000, description="The news article text to classify."
    )

def clean_text(text: str) -> str:
    """Cleans input text by normalizing quotes and whitespace."""
    text = re.sub(r'[\'""]+', '"', text)
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.post("/classify")
def classify_text(input: TextInput):
    try:
        cleaned_text = clean_text(input.text)
        inputs = tokenizer(
            cleaned_text,
            return_tensors="tf",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        outputs = model(inputs)
        predicted_label = tf.argmax(outputs.logits, axis=1).numpy()[0]
        return {"label": int(predicted_label)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")