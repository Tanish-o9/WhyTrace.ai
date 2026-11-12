FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    graphviz \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project specification and install dependencies
COPY pyproject.toml README.md /app/
COPY src /app/src/
RUN pip install --no-cache-dir -e ".[dashboard,dev]"

# Copy all application files
COPY . /app/

EXPOSE 8080 8501

# Default command: start whyTrace collector REST server
CMD ["whytrace", "serve", "--host", "0.0.0.0", "--port", "8080"]

