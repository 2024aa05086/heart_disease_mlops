FROM python:3.11-slim

# Create working directory
WORKDIR /app

# Install system dependencies for MLflow
RUN apt-get update && \
    apt-get install -y build-essential default-libmysqlclient-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the API port
EXPOSE 8000

# Default command to run the API using gunicorn and uvicorn workers
CMD ["gunicorn", "app.main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]