# Use an official lightweight Python runtime matching your development environment
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install minimal system utilities needed for compiling extensions and running healthchecks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install dependencies inside the container isolation layer
RUN pip install --no-cache-dir -r requirements.txt

# Copy your core application scripts and modules
COPY src/ ./src/
COPY data/ ./data/
COPY app.py .
COPY predict_match.py .
COPY implied_xg.py .

# Expose the default Streamlit web server port
EXPOSE 8501

# Add a healthcheck to ensure the container is running optimally
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app binding to port 8501 and listening on all network interfaces
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]