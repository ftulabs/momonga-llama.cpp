# Dedicated llama.cpp Systemd Services (Jetson AGX Xavier)

This directory contains the production `systemd` service definitions, environment configurations, and startup wrappers for hosting **`embeddinggemma-2`** and **`bge-reranker-v2-m3`** as dedicated, co-resident background services on the NVIDIA Jetson AGX Xavier.

---

## 1. Services Overview

| Service Name | Default Port | Model / Architecture | Primary Endpoint | Memory Footprint |
|---|---|---|---|---|
| `llama-embedding.service` | `8086` | EmbeddingGemma-2 Q8_0 + Vision mmproj | `POST /v1/embeddings` | ~1.2 GB RSS |
| `llama-rerank.service` | `8012` | bge-reranker-v2-m3 Q8_0 Encoder Reranker | `POST /v1/rerank` | ~364 MB RSS |

Both services run simultaneously on the Jetson AGX Xavier, consuming **~1.6 GB of unified memory combined**.

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

curl -s -X POST http://127.0.0.1:8012/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3",
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital of France.",
      "The recipe for boiling pasta takes 10 minutes.",
      "Madrid is the capital of Spain."
    ]
  }'
```
