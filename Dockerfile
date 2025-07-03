# Use Docker BuildKit syntax for advanced features
# syntax=docker/dockerfile:1.3.1
#####################################################
# Stage 1: Build wheels for all Python dependencies #
#####################################################
FROM python:3.11-slim AS builder
WORKDIR /app

# Install build tools and Git, combine into single RUN and clean apt caches
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest
COPY requirements.txt .

# Install Python dependencies  
RUN pip install --no-cache-dir -r requirements.txt \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

####################################################
# Stage 2: Create minimal runtime image            #
####################################################
FROM python:3.11-slim
WORKDIR /app

# Copy prebuilt wheels and requirements file from builder
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install all Python dependencies from wheel cache
RUN pip install --no-cache /wheels/*

# Copy the dataset directory
COPY dataset/ dataset/

# Copy the training script and documentation
COPY train_gpt_language_model.py .
COPY train_gpt_language_model.md .

# Default command to launch the GPT training script
CMD ["python", "train_gpt_language_model.py"]
