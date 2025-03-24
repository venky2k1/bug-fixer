# Use Python as the base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install flask torch transformers

# Expose the API port
EXPOSE 5000

# Start Flask server
CMD ["python", "app.py"]
