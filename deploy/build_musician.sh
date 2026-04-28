#!/usr/bin/env bash
# Orchestra SDK — Build the Musician training container
# =======================================================
# Builds the orchestra-musician:latest Docker image from docker/Dockerfile.musician.
# Supports NVIDIA GPU (default) and CPU-only (--cpu) variants.
#
# Usage:
#   chmod +x deploy/build_musician.sh
#   ./deploy/build_musician.sh              # NVIDIA CUDA image
#   ./deploy/build_musician.sh --cpu        # CPU-only / synthetic test image
#   ./deploy/build_musician.sh --tag v1.2   # Custom image tag

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="orchestra-musician:latest"
DOCKERFILE="$REPO_ROOT/docker/Dockerfile.musician"
CPU_MODE=false

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            CPU_MODE=true
            TAG="orchestra-musician-cpu:latest"
            DOCKERFILE="$REPO_ROOT/examples/synthetic/Dockerfile"
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--cpu] [--tag TAG]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ── Pre-flight checks ────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "✗  Docker not found. Install from https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info &>/dev/null; then
    echo "✗  Docker daemon is not running. Start Docker and retry."
    exit 1
fi

if [[ ! -f "$DOCKERFILE" ]]; then
    echo "✗  Dockerfile not found: $DOCKERFILE"
    exit 1
fi

# ── NVIDIA check (GPU builds only) ──────────────────────────────────────────
if [[ "$CPU_MODE" == "false" ]]; then
    if ! command -v nvidia-smi &>/dev/null; then
        echo "⚠  nvidia-smi not found — CUDA image may not work at runtime."
        echo "   Use --cpu for a CPU-only build, or install NVIDIA drivers."
    else
        echo "✓  NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -2
    fi
fi

# ── Build ────────────────────────────────────────────────────────────────────
echo ""
echo "Building Docker image: $TAG"
echo "  Dockerfile: $DOCKERFILE"
echo "  Context:    $REPO_ROOT"
echo ""

docker build \
    --file "$DOCKERFILE" \
    --tag "$TAG" \
    --label "org.opencontainers.image.title=orchestra-musician" \
    --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$REPO_ROOT"

echo ""
echo "✓  Image built: $TAG"
echo ""
echo "Test the image manually:"
echo "  docker run --rm $TAG python3 -c \"import torch; print(torch.__version__)\""
if [[ "$CPU_MODE" == "false" ]]; then
    echo "  docker run --rm --gpus all $TAG nvidia-smi"
fi
