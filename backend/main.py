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
from lime_explainer import create_bias_explainer  # Import LIME explainer
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

# --- Initialize LIME Explainer ---
bias_explainer = None
if bias_model and bias_tokenizer:
    try:
        # Import here to avoid issues if lime is not installed
        bias_explainer = create_bias_explainer(
            bias_model, 
            bias_tokenizer, 
            {0: "Republican", 1: "Liberal", 2: "Neutral", 3: "Other"}
        )
        if bias_explainer:
            print("✅ LIME explainer initialized successfully!")
        else:
            print("⚠️ LIME explainer initialization failed")
    except ImportError:
        print("⚠️ LIME not available - install with: pip install lime")
    except Exception as e:
        print(f"⚠️ Failed to initialize LIME explainer: {e}")
        bias_explainer = None

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

# Wine model is temporarily disabled due to architecture issues
wine_model = None
print("⚠️ Wine Quality model is temporarily disabled. Prediction endpoint will be unavailable.")

# Commented out wine model loading section
# try:
#     if os.path.exists(WINE_WEIGHTS_PATH):
#         print(f"-> Attempting to build wine model architecture...")
#         # 1. Create the architecture first
#         wine_model = build_wine_model(N_WINE_FEATURES)
# 
#         print(f"-> Attempting to load weights from: {WINE_WEIGHTS_PATH}")
#         # 2. Load the weights into the architecture
#         wine_model.load_weights(WINE_WEIGHTS_PATH) # Use load_weights() !
# 
#         print(f"✅ Wine Quality model architecture built and weights loaded successfully!")
#     else:
#         print(f"⚠️ Wine Quality model weights file not found at {WINE_WEIGHTS_PATH}. Prediction endpoint will be unavailable.")
#         wine_model = None
# except Exception as e:
#     print(f"❌ Failed to build Wine Quality model or load weights: {e}")
#     import traceback
#     traceback.print_exc() # Print full error details
#     wine_model = None

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

class ExplainableNewsInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="The news article text to classify and explain.")
    include_explanation: bool = Field(default=True, description="Whether to include LIME explanation.")
    num_features: int = Field(default=5, ge=3, le=15, description="Number of features to analyze (3-15, lower is faster).")
    fast_mode: bool = Field(default=True, description="Use fast mode for quicker responses (recommended for web apps).")

class NewsBatchInput(BaseModel):
    file_content: str = Field(..., description="The CSV file content as a string (must contain a 'content' column).")
    expected_labels: bool = Field(default=False, description="Whether the CSV includes expected labels for evaluation.")
    has_header: bool = Field(default=True, description="Whether the CSV has a header row.")
    save_report: bool = Field(default=False, description="Whether to save a detailed HTML report of results.")
    export_results: bool = Field(default=False, description="Whether to export results as CSV.")

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

@app.post("/classify_batch", tags=["News Bias"])
async def classify_news_batch(input: NewsBatchInput):
    """
    Process a batch of news articles from a CSV file for bias classification.
    
    The CSV must contain a 'content' column with the text to classify.
    If expected_labels is True, the CSV should also include a 'label' column with expected labels (0-3).
    
    Returns detailed statistics and results of the batch classification.
    Optionally can save detailed HTML reports and export results as CSV.
    """
    if not bias_model or not bias_tokenizer:
        raise HTTPException(status_code=503, detail="News Bias model is not available.")
    
    try:
        # Import pandas for CSV processing and visualization libraries
        import pandas as pd
        import io
        import datetime
        import os
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
        
        # Parse CSV content from string
        csv_file = io.StringIO(input.file_content)
        
        # Read CSV
        if input.has_header:
            df = pd.read_csv(csv_file)
        else:
            df = pd.read_csv(csv_file, header=None, names=["content"] if not input.expected_labels else ["content", "label"])
        
        # Validate content column exists
        if "content" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'content' column")
        
        # Validate label column if expected
        if input.expected_labels and "label" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'label' column when expected_labels=True")
        
        # Prepare empty lists for results
        predictions = []
        probabilities_list = []
        bias_scores = []
        processing_times = []
        
        # Record start time for overall process
        batch_start_time = datetime.datetime.now()
        
        # Process each text in the batch
        total_records = len(df)
        print(f"Starting batch classification of {total_records} records")
        
        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"Processing record {idx}/{total_records}")
            
            try:
                # Record start time for this prediction
                pred_start_time = datetime.datetime.now()
                
                text = row["content"]
                if pd.isna(text) or not isinstance(text, str):
                    predictions.append(None)
                    probabilities_list.append(None)
                    bias_scores.append(None)
                    processing_times.append(None)
                    continue
                
                # Clean and classify the text
                cleaned_text = clean_text(text)
                inputs = bias_tokenizer(
                    cleaned_text, 
                    return_tensors="tf", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=512
                )
                outputs = bias_model(inputs)
                
                # Get raw logits
                logits = outputs.logits.numpy()[0]
                
                # Calculate bias score and get probabilities
                bias_score, probabilities = calculate_bias_score(logits)
                
                # Get predicted label
                predicted_label = int(tf.argmax(logits, axis=0).numpy())
                
                # Calculate processing time in milliseconds
                pred_end_time = datetime.datetime.now()
                processing_time = (pred_end_time - pred_start_time).total_seconds() * 1000
                
                predictions.append(predicted_label)
                probabilities_list.append(probabilities)
                bias_scores.append(bias_score)
                processing_times.append(round(processing_time, 2))
            
            except Exception as e:
                print(f"Error processing record {idx}: {e}")
                predictions.append(None)
                probabilities_list.append(None)
                bias_scores.append(None)
                processing_times.append(None)
        
        # Record end time for overall process
        batch_end_time = datetime.datetime.now()
        total_processing_time = (batch_end_time - batch_start_time).total_seconds()
        
        # Create results DataFrame
        results_df = df.copy()
        results_df["predicted_label"] = predictions
        results_df["bias_score"] = bias_scores
        results_df["processing_time_ms"] = processing_times
        
        # Add label meanings
        results_df["predicted_meaning"] = results_df["predicted_label"].map(lambda x: BIAS_LABEL_MEANINGS.get(x, "Unknown") if x is not None else "Error")
        if input.expected_labels:
            results_df["expected_meaning"] = results_df["label"].map(lambda x: BIAS_LABEL_MEANINGS.get(x, "Unknown") if x is not None else "Error") 
        
        # Add probability columns
        for label_idx in range(len(BIAS_LABEL_MEANINGS)):
            label_name = BIAS_LABEL_MEANINGS.get(label_idx, f"Label_{label_idx}")
            results_df[f"prob_{label_name}"] = [
                round(float(p[label_idx] * 100), 2) if p is not None else None 
                for p in probabilities_list
            ]
        
        # Add prediction status (correct, incorrect, missing) if labels are provided
        if input.expected_labels:
            def get_prediction_status(row):
                if pd.isna(row["predicted_label"]) or pd.isna(row["label"]):
                    return "missing"
                return "correct" if row["predicted_label"] == row["label"] else "incorrect"
            
            results_df["prediction_status"] = results_df.apply(get_prediction_status, axis=1)
        
        # Evaluation metrics if expected labels are provided
        evaluation_metrics = {}
        confusion_fig = None
        distribution_fig = None
        
        if input.expected_labels:
            # Filter out rows where prediction failed
            valid_mask = results_df["predicted_label"].notnull() & results_df["label"].notnull()
            valid_df = results_df[valid_mask]
            
            if not valid_df.empty:
                # Convert to integers
                true_labels = valid_df["label"].astype(int).tolist()
                pred_labels = valid_df["predicted_label"].astype(int).tolist()
                
                # Calculate metrics
                accuracy = accuracy_score(true_labels, pred_labels)
                precision, recall, f1, support = precision_recall_fscore_support(
                    true_labels, pred_labels, average=None
                )
                
                # Classification report
                class_names = [BIAS_LABEL_MEANINGS.get(i, f"Label_{i}") for i in range(len(BIAS_LABEL_MEANINGS))]
                report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
                
                # Confusion matrix
                cm = confusion_matrix(true_labels, pred_labels)
                cm_dict = {
                    "matrix": cm.tolist(),
                    "labels": class_names
                }
                
                # Track incorrect predictions for detailed review
                incorrect_mask = valid_df["label"] != valid_df["predicted_label"]
                incorrect_predictions = valid_df[incorrect_mask].to_dict(orient="records")
                
                # Calculate per-class metrics
                class_metrics = []
                for i, label in enumerate(class_names):
                    if i < len(precision):  # Make sure we have metrics for this class
                        class_metrics.append({
                            "label": i,
                            "label_name": label,
                            "precision": float(round(precision[i], 3)),
                            "recall": float(round(recall[i], 3)),
                            "f1": float(round(f1[i], 3)),
                            "support": int(support[i]),
                            "true_positives": int(cm[i, i]) if i < len(cm) else 0,
                            "false_positives": int(cm[:, i].sum() - cm[i, i]) if i < len(cm) else 0,
                            "false_negatives": int(cm[i, :].sum() - cm[i, i]) if i < len(cm) else 0
                        })
                
                # Metrics for response
                evaluation_metrics = {
                    "accuracy": float(accuracy),
                    "classification_report": report,
                    "confusion_matrix": cm_dict,
                    "class_metrics": class_metrics,
                    "total_samples": int(len(valid_df)),
                    "correct_predictions": int(sum(valid_df["label"] == valid_df["predicted_label"])),
                    "incorrect_predictions": int(sum(incorrect_mask)),
                    "incorrect_examples": incorrect_predictions,
                    "avg_precision": float(report["weighted avg"]["precision"]),
                    "avg_recall": float(report["weighted avg"]["recall"]),
                    "avg_f1": float(report["weighted avg"]["f1-score"])
                }
                  # Create visualizations if requested
                if input.save_report:
                    try:
                        # Try to import visualization libraries
                        visualization_available = False
                        try:
                            import matplotlib
                            matplotlib.use('Agg')  # Use non-interactive backend
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            from io import BytesIO
                            import base64
                            visualization_available = True
                            print("Visualization libraries are available.")
                        except ImportError as ie:
                            print(f"Visualization libraries not available: {ie}")
                            visualization_available = False
                        
                        if visualization_available:
                            try:
                                # Create confusion matrix heatmap
                                plt.figure(figsize=(10, 8))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                            xticklabels=class_names, yticklabels=class_names)
                                plt.xlabel('Predicted')
                                plt.ylabel('True')
                                plt.title('Confusion Matrix')
                                
                                # Save to BytesIO
                                cm_img = BytesIO()
                                plt.savefig(cm_img, format='png', bbox_inches='tight')
                                plt.close()
                                cm_img.seek(0)
                                
                                # Convert to base64 for the response
                                cm_b64 = base64.b64encode(cm_img.read()).decode('utf-8')
                                evaluation_metrics["confusion_matrix_img"] = cm_b64
                                
                                # Create class distribution bar chart
                                plt.figure(figsize=(12, 6))
                                
                                # True vs Predicted counts
                                true_counts = pd.Series(true_labels).value_counts().sort_index()
                                pred_counts = pd.Series(pred_labels).value_counts().sort_index()
                                
                                # Ensure all classes are represented
                                all_classes = range(len(class_names))
                                true_counts = pd.Series([true_counts.get(i, 0) for i in all_classes], index=all_classes)
                                pred_counts = pd.Series([pred_counts.get(i, 0) for i in all_classes], index=all_classes)
                                
                                # Create the bar chart
                                width = 0.35
                                x = np.arange(len(class_names))
                                fig, ax = plt.subplots(figsize=(12, 6))
                                ax.bar(x - width/2, true_counts, width, label='True')
                                ax.bar(x + width/2, pred_counts, width, label='Predicted')
                                
                                ax.set_xticks(x)
                                ax.set_xticklabels(class_names)
                                ax.set_ylabel('Count')
                                ax.set_title('True vs Predicted Class Distribution')
                                ax.legend()
                                
                                # Save to BytesIO
                                dist_img = BytesIO()
                                plt.savefig(dist_img, format='png', bbox_inches='tight')
                                plt.close()
                                dist_img.seek(0)
                                
                                # Convert to base64 for the response
                                dist_b64 = base64.b64encode(dist_img.read()).decode('utf-8')
                                evaluation_metrics["class_distribution_img"] = dist_b64
                                print("Generated visualizations successfully.")
                            except Exception as viz_err:
                                print(f"Error generating visualizations: {viz_err}")
                        
                    except Exception as e:
                        print(f"Warning: Visualization generation failed: {e}")
        
        # Calculate processing time statistics if available
        valid_times = [t for t in processing_times if t is not None]
        time_stats = {}
        if valid_times:
            time_stats = {
                "avg_processing_time_ms": round(sum(valid_times) / len(valid_times), 2),
                "min_processing_time_ms": round(min(valid_times), 2),
                "max_processing_time_ms": round(max(valid_times), 2),
                "total_batch_time_seconds": round(total_processing_time, 2)
            }
        
        # Convert results to records for response
        results_records = results_df.to_dict(orient="records")
        
        # Generate response
        response = {
            "total_processed": total_records,
            "successful_predictions": sum(p is not None for p in predictions),
            "failed_predictions": sum(p is None for p in predictions),
            "processing_stats": time_stats,
            "results": results_records
        }
        
        # Add evaluation metrics if available
        if evaluation_metrics:
            response["evaluation"] = evaluation_metrics
        
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export results to CSV if requested
        if input.export_results:
            try:
                # Create output directory if it doesn't exist
                output_dir = "batch_results"
                os.makedirs(output_dir, exist_ok=True)
                
                # Export to CSV
                csv_filename = f"{output_dir}/batch_results_{timestamp}.csv"
                results_df.to_csv(csv_filename, index=False)
                response["exported_csv"] = csv_filename
                print(f"Results exported to {csv_filename}")
            except Exception as e:
                print(f"Warning: Failed to export results to CSV: {e}")
          # Generate HTML report if requested
        if input.save_report:
            try:
                # Create output directory if it doesn't exist
                output_dir = "batch_reports"
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created report directory at {os.path.abspath(output_dir)}")
                
                # Generate HTML report with simple HTML
                html_filename = f"{output_dir}/batch_report_{timestamp}.html"
                
                # Create a simple HTML report without pandas styling
                summary_html = f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #333366; }}
                        table {{ border-collapse: collapse; margin: 15px 0; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                        .metrics {{ display: flex; flex-wrap: wrap; }}
                        .metric-box {{ background-color: white; border-radius: 5px; padding: 10px; margin: 10px; flex: 1; min-width: 200px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                        .metric-value {{ font-size: 24px; font-weight: bold; color: #333366; }}
                        .metric-label {{ color: #666; }}
                        .correct {{ color: green; background-color: #d4edda; }}
                        .incorrect {{ color: red; background-color: #f8d7da; }}
                    </style>
                </head>
                <body>
                    <h1>Batch Classification Report</h1>
                    <div class="summary">
                        <h2>Summary Statistics</h2>
                        <div class="metrics">
                            <div class="metric-box">
                                <div class="metric-value">{response["total_processed"]}</div>
                                <div class="metric-label">Total Articles</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{response["successful_predictions"]}</div>
                                <div class="metric-label">Successful Predictions</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{response["failed_predictions"]}</div>
                                <div class="metric-label">Failed Predictions</div>
                            </div>
                """
                
                # Add evaluation metrics if available
                if evaluation_metrics:
                    summary_html += f"""
                            <div class="metric-box">
                                <div class="metric-value">{evaluation_metrics["accuracy"]:.2%}</div>
                                <div class="metric-label">Accuracy</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{evaluation_metrics["correct_predictions"]}</div>
                                <div class="metric-label">Correct Predictions</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{evaluation_metrics["incorrect_predictions"]}</div>
                                <div class="metric-label">Incorrect Predictions</div>
                            </div>
                        </div>
                        
                        <h2>Performance Metrics</h2>
                        <table>
                            <tr>
                                <th>Class</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1 Score</th>
                                <th>Support</th>
                            </tr>
                    """
                    
                    # Add class metrics rows
                    for class_metric in evaluation_metrics["class_metrics"]:
                        summary_html += f"""
                            <tr>
                                <td>{class_metric["label_name"]}</td>
                                <td>{class_metric["precision"]:.3f}</td>
                                <td>{class_metric["recall"]:.3f}</td>
                                <td>{class_metric["f1"]:.3f}</td>
                                <td>{class_metric["support"]}</td>
                            </tr>
                        """
                    
                    # Add weighted average row
                    summary_html += f"""
                            <tr>
                                <th>Weighted Avg</th>
                                <td>{evaluation_metrics["avg_precision"]:.3f}</td>
                                <td>{evaluation_metrics["avg_recall"]:.3f}</td>
                                <td>{evaluation_metrics["avg_f1"]:.3f}</td>
                                <td>{evaluation_metrics["total_samples"]}</td>
                            </tr>
                        </table>
                        
                        <h2>Confusion Matrix</h2>
                        <table>
                            <tr>
                                <th colspan="2" rowspan="2"></th>
                                <th colspan="{len(evaluation_metrics['confusion_matrix']['labels'])}">Predicted</th>
                            </tr>
                            <tr>
                    """
                    
                    # Add confusion matrix header
                    for label in evaluation_metrics["confusion_matrix"]["labels"]:
                        summary_html += f"<th>{label}</th>"
                    
                    summary_html += "</tr>"
                    
                    # Add confusion matrix rows
                    matrix = evaluation_metrics["confusion_matrix"]["matrix"]
                    labels = evaluation_metrics["confusion_matrix"]["labels"]
                    
                    for i, row in enumerate(matrix):
                        summary_html += f"<tr><th rowspan='1'>True</th><th>{labels[i]}</th>"
                        for cell in row:
                            bg_color = "#d4edda" if i == matrix.index(row) and cell > 0 else ""
                            summary_html += f"<td style='background-color: {bg_color}'>{cell}</td>"
                        summary_html += "</tr>"
                    
                    summary_html += "</table>"
                
                # Add processing time statistics if available
                if time_stats:
                    summary_html += f"""
                        <h2>Processing Performance</h2>
                        <div class="metrics">
                            <div class="metric-box">
                                <div class="metric-value">{time_stats["avg_processing_time_ms"]} ms</div>
                                <div class="metric-label">Average Processing Time</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{time_stats["min_processing_time_ms"]} ms</div>
                                <div class="metric-label">Min Processing Time</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{time_stats["max_processing_time_ms"]} ms</div>
                                <div class="metric-label">Max Processing Time</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{time_stats["total_batch_time_seconds"]} s</div>
                                <div class="metric-label">Total Batch Time</div>
                            </div>
                        </div>
                    """
                
                # Close summary div
                summary_html += """
                    </div>
                    <h2>Detailed Results</h2>
                    <table>
                        <tr>
                """
                
                # Add table headers for results
                cols = ["Content", "Label", "Prediction", "Bias Score"]
                if input.expected_labels:
                    cols = ["Content", "True Label", "Predicted Label", "Status", "Bias Score"]
                
                for col in cols:
                    summary_html += f"<th>{col}</th>"
                summary_html += "</tr>"
                
                # Add results rows
                for item in results_records:
                    summary_html += "<tr>"
                    
                    # Truncate content for readability
                    content = item["content"]
                    if len(content) > 100:
                        content = content[:97] + "..."
                    
                    # Add content
                    summary_html += f"<td>{content}</td>"
                    
                    if input.expected_labels:
                        # Add true label
                        true_label = f"{item['label']} ({item.get('expected_meaning', '')})"
                        summary_html += f"<td>{true_label}</td>"
                        
                        # Add predicted label
                        pred_label = f"{item['predicted_label']} ({item.get('predicted_meaning', '')})"
                        summary_html += f"<td>{pred_label}</td>"
                        
                        # Add status (correct or incorrect)
                        status_class = "correct" if item["label"] == item["predicted_label"] else "incorrect"
                        status_text = "✓" if item["label"] == item["predicted_label"] else "✗"
                        summary_html += f"<td class='{status_class}'>{status_text}</td>"
                    else:
                        # Add just the predicted label
                        pred_label = f"{item['predicted_label']} ({item.get('predicted_meaning', '')})"
                        summary_html += f"<td>N/A</td><td>{pred_label}</td>"
                    
                    # Add bias score
                    summary_html += f"<td>{item['bias_score']}</td>"
                    
                    summary_html += "</tr>"
                
                summary_html += """
                    </table>
                    <p>Report generated on: """ + timestamp + """</p>
                </body>
                </html>
                """
                
                # Write the HTML report to file
                with open(html_filename, "w", encoding="utf-8") as f:
                    f.write(summary_html)
                
                response["report_path"] = html_filename
                print(f"HTML report generated at {os.path.abspath(html_filename)}")
                
            except Exception as e:
                print(f"Warning: Failed to generate HTML report: {e}")
                response["report_error"] = str(e)
        
        return response
    
    except pd.errors.ParserError as e:
        print(f"❌ Error parsing CSV: {e}")
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    except Exception as e:
        print(f"❌ Error during batch classification: {e}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")

@app.post("/classify_with_explanation", tags=["News Bias"])
async def classify_with_explanation(input: ExplainableNewsInput):
    """
    Classify news text and provide user-friendly explanations using LIME.
    Shows which words influenced the AI's decision.
    """
    if not bias_model or not bias_tokenizer:
        raise HTTPException(status_code=503, detail="News Bias model is not available.")
    
    try:
        # Get basic classification first
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
        
        # Create base response
        response = {
            "label": predicted_label,
            "label_meaning": BIAS_LABEL_MEANINGS.get(predicted_label, "Unknown"),
            "confidence": float(round(probabilities[predicted_label] * 100, 2)),
            "confidence_scores": confidence_scores,
            "bias_score": bias_score
        }          # Add LIME explanation if requested and available
        if input.include_explanation and bias_explainer:
            try:
                explanation_result = bias_explainer.explain_prediction(
                    input.text, 
                    num_features=input.num_features,
                    num_samples=30  # Much lower for speed
                )
                response["explanation"] = explanation_result
                response["has_explanation"] = True
            except Exception as e:
                print(f"⚠️ LIME explanation failed: {e}")
                response["explanation_error"] = "Explanation generation failed"
                response["has_explanation"] = False
        else:
            response["has_explanation"] = False
            if not bias_explainer:
                response["explanation_error"] = "LIME explainer not available"
        
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
        print(f"❌ Error during explainable news classification: {e}")
        raise HTTPException(status_code=500, detail=f"Explainable news classification failed: {str(e)}")

@app.post("/predict_wine", tags=["Wine Quality"])
async def predict_wine_quality(features: WineFeaturesInput):
    if not wine_model:
        raise HTTPException(status_code=503, detail="Wine Quality model is temporarily disabled due to technical issues. Please try again later.")
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

@app.post("/upload_batch_csv", tags=["News Bias"])
async def upload_batch_csv(
    file: UploadFile = File(...),
    expected_labels: bool = Form(False),
    has_header: bool = Form(True),
    save_report: bool = Form(False),
    export_results: bool = Form(False)
):
    """
    Upload a CSV file for batch processing of news articles.
    
    The CSV must contain a 'content' column with the text to classify.
    If expected_labels is True, the CSV should also include a 'label' column with expected labels (0-3).
    
    Returns detailed statistics and results of the batch classification.
    """
    if not bias_model or not bias_tokenizer:
        raise HTTPException(status_code=503, detail="News Bias model is not available.")
    
    # Check if file is a CSV
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    try:
        # Read file content
        file_content = await file.read()
        file_content_str = file_content.decode('utf-8')
        
        # Create input object for the batch processing endpoint
        input_obj = NewsBatchInput(
            file_content=file_content_str,
            expected_labels=expected_labels,
            has_header=has_header,
            save_report=save_report,
            export_results=export_results
        )
        
        # Use the batch classification endpoint
        return await classify_news_batch(input_obj)
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV file must be UTF-8 encoded")
    except Exception as e:
        print(f"❌ Error during CSV batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"CSV batch processing failed: {str(e)}")

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
