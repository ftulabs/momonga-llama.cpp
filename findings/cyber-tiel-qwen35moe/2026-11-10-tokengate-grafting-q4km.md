# TokenGate Grafting: Layer 3 Donor Gate onto Q4KM-all-q4k

**Date:** 2026-11-10  
**Base Model:** `/mnt/ssd/models/Q4KM-all-q4k.gguf` (20.46 GB, NVMe)  
**Donor Gate:** `/mnt/sdcard/home/minh/src/g3_gate.npz` (Layer 3 checkpoint from Qwen3.6-35B-A3B, step 3002)  
**Donor LoRA:** `/mnt/sdcard/home/minh/src/g3_gate.lora.npz`  
**Output Model:** `/mnt/ssd/models/Cyber-Tiel-35B-A3B-Q4KM-tg.gguf` (20.46 GB)  
**Output LoRA Adapter:** `/mnt/ssd/models/Cyber-Tiel-35B-A3B-tg-lora.gguf` (7.9 MB, 20 LoRA pairs, alpha 32.0, rank 16)  

---

## 1. Summary

The Layer 3 TokenGate donor checkpoint from Qwen3.6-35B-A3B was grafted onto the high-throughput `Q4KM-all-q4k.gguf` re-quantized base model using `/home/minh/src/llama.cpp-tokengate/tools/tokengate/convert_tokengate_to_gguf.py`.

The script verified the underlying `qwen35moe` geometry and appended the 21 gate tensors across the 9 full-attention layers above Layer 3:
* Gate Layer: 3
* Gated Layers: `[7, 11, 15, 19, 23, 27, 31, 35, 39]` (16 heads)
* Added Tensors:
  - `blk.3.tg_gate.weight` (f16)
  - `blk.3.tg_up.weight` (f16)
  - `blk.3.tg_down` (f32)
  - `blk.{7,11,15,19,23,27,31,35,39}.tg_qscale`
  - `blk.{7,11,15,19,23,27,31,35,39}.tg_kbias`
* Paired LoRA: Exported to `/mnt/ssd/models/Cyber-Tiel-35B-A3B-tg-lora.gguf`.

Both files are located directly in `/mnt/ssd/models` for active use by `llama-server` and Unsloth Studio.
