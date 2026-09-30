# Use official lightweight Python image
FROM python:3.12-slim

# Set environment variables for optimized Python runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface_cache

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system --group appuser

# Set the working directory
WORKDIR /app

# Install system dependencies (clearing cache to save space)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Create ML cache directory and assign permissions BEFORE switching users
RUN mkdir -p /tmp/huggingface_cache && chown -R appuser:appgroup /tmp/huggingface_cache

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
# Inside your Dockerfile, update the pip install step to:
# Install dependencies directly from requirements.txt
# Install dependencies directly from requirements.txt with Airflow 2 constraints
RUN pip install --no-cache-dir -r requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt"
# Copy the rest of the application codebase
COPY . .

# Explicitly create Airflow directories and give ownership to the non-root user
RUN mkdir -p /app/airflow/logs /app/airflow/dags && \
    chown -R appuser:appgroup /app

# Switch to the secure non-root user
USER appuser

# Expose the Flask port
EXPOSE 5000

# Run the application using Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "600", "app:create_app()"]