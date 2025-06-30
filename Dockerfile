# Use Docker BuildKit syntax for advanced features
# syntax=docker/dockerfile:1.3.1
#####################################################
# Stage 1: Build wheels for all Python dependencies #
#####################################################
FROM python:3.11-slim AS builder
WORKDIR /app

# Install build tools and Git, combine into single RUN and clean apt caches  [oai_citation:7‡benjamintoll.com](https://benjamintoll.com/2023/06/10/on-dockerfile-best-practices/?utm_source=chatgpt.com) [oai_citation:8‡opensource.com](https://opensource.com/article/20/5/optimize-container-builds?utm_source=chatgpt.com)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest
COPY requirements.txt .

# Install CPU-only PyTorch wheel and build wheels for other requirements  [oai_citation:9‡stackoverflow.com](https://stackoverflow.com/questions/76395122/how-to-build-an-efficient-and-fast-dockerfile-for-a-pytorch-model-running-on)
RUN pip install --no-cache-dir torch==2.0.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

####################################################
# Stage 2: Create minimal runtime image            #
####################################################
FROM python:3.11-slim
WORKDIR /app

# Copy prebuilt wheels and requirements file from builder
COPY --from=builder /wheels /wheels
COPY --from=builder requirements.txt .

# Install all Python dependencies from wheel cache, including torchtext  [oai_citation:10‡docs.vultr.com](https://docs.vultr.com/how-to-build-a-pytorch-container-image?utm_source=chatgpt.com)
RUN pip install --no-cache /wheels/* \
    && pip install --no-cache-dir torchtext

# Copy the rest of the application code
COPY . .

# Default command to launch the training script
CMD ["python", "train_lstm_language_model.py"]