FROM python:3.11.9-slim

WORKDIR /app

# Install system dependencies (needed for numpy, pillow, tensorflow etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements first (for caching)
COPY backend/requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full backend code
COPY backend/ .

# Expose port
EXPOSE 10000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
