# Orchestra Musician — Synthetic (no-GPU) image
# ================================================
# Minimal Python image for running the synthetic train.py without CUDA.
# Build:  docker build -f examples/synthetic/Dockerfile -t orchestra-musician-synthetic .
# Run:    docker run --rm -v $WORKSPACE:/workspace orchestra-musician-synthetic

FROM python:3.11-slim

LABEL maintainer="Orchestra SDK"
LABEL description="Minimal Musician image for synthetic/no-GPU testing"

# No additional dependencies needed — train.py uses only stdlib
WORKDIR /workspace

# The runner mounts the workspace at /workspace and runs train.py
CMD ["python", "train.py"]
