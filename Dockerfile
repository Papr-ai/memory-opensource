# Use an official Python runtime as a parent image
FROM --platform=$TARGETPLATFORM python:3.11.7-bullseye

# Set build-time platform argument
ARG TARGETPLATFORM="linux/amd64"

# Set the working directory in the container to /app
WORKDIR /app

# Handle apt-get update with a retry strategy to mitigate network issues
RUN for i in {1..5}; do apt-get update && break || sleep 15; done

# Install necessary packages
RUN apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    gnupg \
    ca-certificates \
    git \
    libmagic1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install poetry with a specific version
RUN pip install --no-cache-dir poetry==1.7.1

# Copy only pyproject.toml and poetry.lock first
COPY pyproject.toml poetry.lock ./

# Configure poetry and install dependencies with better error handling
RUN poetry config virtualenvs.create false \
    && poetry config installer.max-workers 10 \
    && poetry install --no-interaction --no-ansi --no-root --no-cache

# Download the spaCy language model
RUN python -m spacy download en_core_web_sm

# Now copy the rest of the application
COPY . .

# Install the project itself with no cache
RUN poetry install --no-interaction --no-ansi --no-cache

# Verify FastAPI installation
RUN python -c "import fastapi" || pip install fastapi==0.115.6

# Make port 5001 available
EXPOSE 5001

# Make the startup scripts executable
RUN chmod +x start_all_services.sh start_all_services.py

# Make the open source entrypoint executable (for auto-bootstrap)
RUN chmod +x scripts/opensource/docker_entrypoint_opensource.sh

# Run all services (web server + Temporal workers) when the container launches
# Using Python version for better error handling and process management
CMD ["python", "start_all_services.py"]