# backend/Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy backend-specific requirements
COPY requirements.txt .
# Install dependencies without caching to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app and model files
COPY main.py .
COPY bias_classification_model/ ./bias_classification_model/

# Expose port 80 for Azure Container Apps
EXPOSE 80

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]