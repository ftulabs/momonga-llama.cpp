# Dedicated llama.cpp Systemd Services (Jetson AGX Xavier)

This directory contains the production `systemd` service definitions, environment configurations, and startup wrappers for hosting **`embeddinggemma-2`** and **`zerank-2`** as dedicated, co-resident background services on the NVIDIA Jetson AGX Xavier.

---

## 1. Services Overview

| Service Name | Default Port | Model / Architecture | Primary Endpoint | Memory Footprint |
|---|---|---|---|---|
| `llama-embedding.service` | `8086` | EmbeddingGemma-2 Q8_0 + Vision mmproj | `POST /v1/embeddings` | ~1.0 GB RSS (~1.8 GB VRAM) |
| `llama-rerank.service` | `8012` | zerank-2 Q8_0 Causal Generative Reranker | `POST /rerank` | ~2.9 GB RSS (~5.5 GB VRAM) |

Both services run simultaneously on the Jetson AGX Xavier, consuming **under 7.5 GB of unified memory combined** and leaving **22+ GB of free RAM headroom**.

---

## 2. Installed System Paths

- Unit files: `/etc/systemd/system/llama-embedding.service`, `/etc/systemd/system/llama-rerank.service`
- Startup scripts: `/mnt/sdcard/opt/llama/llama-embedding-start.sh`, `/mnt/sdcard/opt/llama/llama-rerank-start.sh`
- Environment config: `/mnt/sdcard/opt/llama/llama-embedding.env`, `/mnt/sdcard/opt/llama/llama-rerank.env`

---

## 3. Management Commands

### Status Check
```bash
sudo systemctl status llama-embedding.service llama-rerank.service --no-pager
```

### Stop / Start / Restart
```bash
# Manage Embedding Service
sudo systemctl stop llama-embedding.service
sudo systemctl start llama-embedding.service
sudo systemctl restart llama-embedding.service

# Manage Reranker Service
sudo systemctl stop llama-rerank.service
sudo systemctl start llama-rerank.service
sudo systemctl restart llama-rerank.service
```

### View Live Logs (journalctl)
```bash
journalctl -u llama-embedding.service -f
journalctl -u llama-rerank.service -f
```

---

## 4. Health & Verification Queries

### Embedding Service (Port 8086)
```bash
curl -s http://127.0.0.1:8086/health

curl -s -X POST http://127.0.0.1:8086/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Autonomous edge robotics on Jetson Xavier"}'
```

### Reranker Service (Port 8012)
```bash
curl -s http://127.0.0.1:8012/health

curl -s -X POST http://127.0.0.1:8012/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital of France.",
      "The recipe for boiling pasta takes 10 minutes.",
      "Madrid is the capital of Spain."
    ]
  }'
```
