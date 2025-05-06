FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create NLTK data directory
RUN mkdir -p /root/nltk_data

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/root/nltk_data'); \
    nltk.download('stopwords', download_dir='/root/nltk_data'); \
    nltk.download('averaged_perceptron_tagger', download_dir='/root/nltk_data'); \
    nltk.download('vader_lexicon', download_dir='/root/nltk_data')"

# Copy application code
COPY . .

# Set environment variables
ENV NLTK_DATA=/root/nltk_data
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 