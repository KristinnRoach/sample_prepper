FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    libsndfile1 \
    sox \
    ffmpeg \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p uploads output \
    && chmod 755 uploads \
    && chmod 755 output

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose the port
EXPOSE 5001

# Command to run the application
CMD ["python3", "main.py"]