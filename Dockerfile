# Use official Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable to prevent .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# Default command
CMD ["python", "main.py"]