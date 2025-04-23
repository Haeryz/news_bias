# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import re

app = FastAPI(
    title="News Bias Classification API",
    description="Classifies news articles into bias categories (0-3) using a pre-trained DistilBERT model.",
    version="1.0.0"
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
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.post("/classify")
def classify_text(input: TextInput):
    """
    Accepts a news article text, cleans it, and returns its predicted bias label (0, 1, 2, or 3).
    - 0, 1, 2, 3 represent different bias categories (e.g., right-wing, left-wing, etc.).
    """
    try:
        # Clean the input text before tokenization
        cleaned_text = clean_text(input.text)
        
        # Tokenize the cleaned text (matches training setup)
        inputs = tokenizer(
            cleaned_text,
            return_tensors="tf",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        # Run model inference
        outputs = model(inputs)
        predicted_label = tf.argmax(outputs.logits, axis=1).numpy()[0]
        
        # Return the predicted label
        return {"label": int(predicted_label)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")