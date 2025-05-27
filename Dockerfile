# Dockerfile for Self-Improving Coding Agent
# Designed to work on university servers without home directory permission issues

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HOME=/tmp
ENV USER=agent

# Create app directory (not in /home to avoid permission issues)
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY base_agent/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir swebench

# Copy the entire codebase
COPY . /app/

# Create necessary directories with proper permissions
RUN mkdir -p /app/results && \
    mkdir -p /app/results/interactive_output && \
    mkdir -p /app/results/interactive_output/agent_output && \
    mkdir -p /app/benchmark_data && \
    mkdir -p /tmp/.cache && \
    chmod -R 755 /app && \
    chmod -R 777 /tmp

# Create a non-root user (but don't use home directory)
RUN useradd -r -s /bin/bash -d /app agent && \
    chown -R agent:agent /app

# Set up environment for the agent
ENV GEMMA_API_URL=http://host.docker.internal:8000
ENV LOG_LEVEL=INFO
ENV WORK_DIR=/app/results/interactive_output

# Switch to the agent user
USER agent

# Expose the web server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command - interactive shell
CMD ["/bin/bash"]
