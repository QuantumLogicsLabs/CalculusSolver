# Neural Inference Service Deployment Guide

This document explains how to deploy `api/neural_app.py` as a standalone
microservice so the Vercel API can proxy neural requests to it via `NEURAL_INFERENCE_URL`.

## Overview

The neural model (`inference/solve.py` + `checkpoints/final/best.pt`) cannot
run on Vercel due to the 250MB bundle limit. Instead, deploy it as a separate
HTTP service on any container platform.

## Step 1: Train and export the checkpoint locally

```bash
# Install neural deps
pip install -r requirements-neural.txt

# Train (saves checkpoints/final/best.pt when val loss improves)
python train.py

# Verify checkpoint exists
python -c "import torch; s = torch.load('checkpoints/final/best.pt'); print('Keys:', list(s.keys())[:3])"
```

## Step 2: Upload the checkpoint

The `best.pt` file must be accessible at runtime. Options:

- **Render**: Mount a disk at `/data/` and set `MODEL_PATH=/data/best.pt`
- **Fly.io**: Use a Fly volume or include it in the Docker image (if < 300MB)
- **HuggingFace Spaces**: Upload `checkpoints/final/best.pt` via the Files tab in the Space UI

## Step 3: Deploy with Docker

A `Dockerfile.neural` is included in the repo root. Build and push:

```bash
docker build -f Dockerfile.neural -t calculus-neural:latest .
docker run -e MODEL_PATH=/checkpoints/best.pt \
  -v $(pwd)/checkpoints/final:/checkpoints \
  -p 8080:8080 calculus-neural:latest
```

### Render (recommended)
1. Create a new Web Service → "Deploy an existing image" or connect the GitHub repo
2. Set Docker file to `Dockerfile.neural`
3. Set environment variable: `MODEL_PATH=/data/best.pt`
4. Mount a disk at `/data/` and upload `best.pt`
5. Note the service URL (e.g., `https://calculus-neural.onrender.com`)

### HuggingFace Spaces (easiest free tier)
1. Create a new Space → SDK: "Docker"
2. Upload the repo contents; HF will use `Dockerfile.neural`
3. Upload `best.pt` under Files → `checkpoints/final/best.pt`
4. Space URL is your neural inference URL (use POST `/solve` endpoint)

### Fly.io
```bash
fly launch --dockerfile Dockerfile.neural
fly secrets set MODEL_PATH=/data/best.pt
fly volumes create model_data --size 1
fly deploy
```

## Step 4: Wire up the Vercel deployment

Once the neural service is running, set the environment variable in Vercel:

```
NEURAL_INFERENCE_URL=https://your-neural-service.onrender.com/solve
```

`api/_shared.py` `get_solver()` will automatically detect this env var and
proxy all solve requests to the neural service with `solver_mode="neural"`.

## Step 5: Verify

```bash
# Health check
curl https://your-neural-service.onrender.com/health

# Solve test
curl -X POST https://your-neural-service.onrender.com/solve \
  -H "Content-Type: application/json" \
  -d '{"op":"diff","var":"x","expr":{"numi":{"terms":[{"coeff":3,"var":{"x":2}}]},"deno":1}}'
```

## Phase 2 (not in scope today)

- Trig/exp/log vocabulary expansion — requires `vocab.json` changes + full dataset regen + retraining
- Auth/rate-limiting on the neural service
- Model versioning / blue-green deployments
