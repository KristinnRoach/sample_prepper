FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/firebase-credentials.json

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY SamplePrepper.ipynb .

# Install Jupyter for running the notebook
RUN pip install notebook

# Expose port for Flask
EXPOSE 5000

# Command to run the notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]