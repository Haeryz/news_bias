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

# Install wget
RUN apt-get update && apt-get install -y wget

# Create model directory and download tf_model.h5 from Google Drive
RUN mkdir -p bias_classification_model
RUN wget -O bias_classification_model/tf_model.h5 "https://drive.google.com/uc?export=download&id=1b8xl5sWVx6SKWtujo957m73XgZgiMI0f"

# Copy FastAPI app and model configs (excluding tf_model.h5)
COPY main.py .
COPY bias_classification_model/ ./bias_classification_model/

# Expose port 80 for Azure Container Apps
EXPOSE 80

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]