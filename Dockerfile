# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create directories with proper permissions
RUN mkdir -p /app/data /app/uploads /app/models /app/static /app/templates /app/instance

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set proper permissions
RUN chmod -R 755 /app
RUN chmod -R 777 /app/data /app/uploads /app/instance

# Create a non-root user
RUN adduser --disabled-password --gecos '' --uid 1000 user

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "=== Krishi Sahyog Starting ==="\n\
echo "Current user: $(whoami)"\n\
echo "Working directory: $(pwd)"\n\
echo "Python version: $(python --version)"\n\
echo "Available space: $(df -h / | tail -1)"\n\
\n\
# Test database creation\n\
echo "Testing database initialization..."\n\
python3 -c "from app import initialize_database; initialize_database()"\n\
\n\
echo "Starting Krishi Sahyog application..."\n\
exec "$@"' > /app/start.sh && chmod +x /app/start.sh

# Change ownership to non-root user
RUN chown -R user:user /app

# Switch to non-root user
USER user

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV PORT=7860

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Use startup script
ENTRYPOINT ["/app/start.sh"]
CMD ["python", "app.py"]