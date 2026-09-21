FROM python:3.12-slim

# Install system dependencies required for OpenCV headless execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY pitch_engine/ ./pitch_engine/
COPY synthetic_generator.py .
COPY synthetic_field_prototype.py .
COPY cli.py .

ENV PYTHONUNBUFFERED=1
ENV MOCK_API_URL=http://mock_api:5000

# Default entrypoint runs the thin CLI runner
CMD ["python", "cli.py"]
