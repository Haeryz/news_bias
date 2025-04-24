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

RUN apt-get update && apt-get install -y wget
RUN wget -O bias_classification_model/tf_model.h5 "https://drive.google.com/file/d/1b8xl5sWVx6SKWtujo957m73XgZgiMI0f/view?usp=drive_link"

COPY main.py .
COPY bias_classification_model/ ./bias_classification_model/

EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]