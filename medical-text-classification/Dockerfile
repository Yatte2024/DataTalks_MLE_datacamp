FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (optional)
# RUN apt-get update && apt-get install -y --no-install-recommends ...
# && rm -rf /var/lib/apt/lists/*

# Copy dependencies file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose Flask port
EXPOSE 8000

# Default command: start the Flask service
CMD ["python", "predict.py"]
