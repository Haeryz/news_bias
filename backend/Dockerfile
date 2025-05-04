# backend/Dockerfile
FROM python:3.9-slim

# Disable GPU usage to suppress CUDA warnings
ENV CUDA_VISIBLE_DEVICES=""

# Set working directory
WORKDIR /app

# Copy backend-specific requirements
COPY requirements.txt .
# Install dependencies without caching to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown for Google Drive downloads (for bias model)
RUN pip install gdown

# --- Bias Model Handling ---
# Create bias model directory
RUN mkdir -p bias_classification_model
# Download tf_model.h5 from Google Drive into the bias model directory
RUN gdown --id 1b8xl5sWVx6SKWtujo957m73XgZgiMI0f -O bias_classification_model/tf_model.h5
# Copy bias model config files (vocab, json configs)
COPY bias_classification_model/ ./bias_classification_model/

# --- Wine Model Handling ---
# Copy the wine quality model file directly from the build context
COPY wine_quality_tf_model.weights.h5 .

# --- Application Code ---
# Copy all the Python application files
COPY main.py PDF_parser.py sentiment.py ./

# --- Runtime Configuration ---
# Expose port 80 for Azure Container Apps (or other deployment targets)
EXPOSE 80

# Run FastAPI with Uvicorn on port 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]