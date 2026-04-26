FROM python:3.9-slim

WORKDIR /app

# Copy requirement and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Set UTF-8 encoding for emoji support in logs
ENV PYTHONUTF8=1

# Run the training pipeline during build so the image has a model baked in
RUN python run_pipeline.py

# Expose FastAPI port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
