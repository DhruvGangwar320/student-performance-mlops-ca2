
# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip --default-timeout=300 install --retries 10 --prefer-binary --no-cache-dir -r requirements.txt
# Copy application code and trained model
COPY main.py .
COPY model.joblib .

# Expose FastAPI port
EXPOSE 8000

# Start the API server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
