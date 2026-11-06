#!/usr/bin/env python3
"""Convert a TokenGate checkpoint (arXiv 2610.26754) into a single GGUF for llama-tokengate.

Inputs: the HF base model directory, the trained gate (gate_*.pt) and its paired LoRA (lora_*.pt).

  1. the LoRA is merged into the base weights: W' = W + (alpha/r) (A B)^T, rounded once to the base dtype
  2. the merged model is converted by convert_hf_to_gguf.py
  3. the gate is appended:
       blk.L.tg_gate.weight / blk.L.tg_up.weight  scoring branch (a clone of layer L's MLP)
       blk.L.tg_down                              absorbed 1-wide readout (f32)
       blk.l.tg_qscale  (l > L)                   gamma_lh = softplus(attn_scale)        [1, n_head]
       blk.l.tg_kbias   (l > L)                   c_lh = softplus(gate_scale) * mean_h(gamma_l) * sqrt(d)  [1, 1, n_head]
       <arch>.tokengate.gate_layer, <arch>.tokengate.bnorm_weight

The gated layers then compute softmax(gamma_lh r_q q.k / sqrt(d) + kb_k c_lh / sqrt(d)) where r is the
scaled-softmax ratio and kb_k the RMS-normalized gate score of the kept key times its own r (see
tokengate.cpp); c is stored premultiplied by sqrt(d) because llama.cpp scales the KQ bias with the logits.
On a hybrid model (qwen35moe: gated delta net layers + a full-attention layer every `full_attention_interval`)
only the full-attention layers above the gate are gated, in order.

Quantized base models (no HF weights at hand, e.g. a tuned IQ4_XS GGUF) take the other path:
  --gguf BASE.gguf      append the gate to an existing GGUF instead of steps 1-2
  --lora-out A.gguf     write the paired LoRA as a llama.cpp adapter (llama-tokengate --lora) instead of merging it
  --export-npz G.npz    torch side only: write the gate and LoRA tensors to .npz files and stop, so that the GGUF
                        side (--gate G.npz --lora G.lora.npz) runs where the base GGUF lives without torch
  --neutral             zero gate bias and unit temperature: the gated model must reproduce the base model at
                        K = all, which tests the split itself
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gguf-py"))
import gguf  # noqa: E402


def load_pt(path: str) -> dict:
    # the checkpoints hold tensors and plain python metadata only
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def load_gate_file(path: str) -> dict:
    """A gate as {**flags, "metadata": {...}, "state_dict": {name: np.ndarray}}, from a .pt or an exported .npz."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        head = json.loads(str(z["__head__"]))
        return {**head, "state_dict": {k: z[k] for k in z.files if k != "__head__"}}
    gate = load_pt(path)
    return {**{k: v for k, v in gate.items() if k != "state_dict"},
            "state_dict": {k: v.float().numpy() for k, v in gate["state_dict"].items()}}


def load_lora_file(path: str) -> dict:
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    return {k: v.float().numpy() for k, v in load_pt(path).items()}


def export_npz(gate: dict, lora: dict, out: str) -> None:
    head = {k: v for k, v in gate.items() if k != "state_dict"}
    np.savez(out, __head__=np.array(json.dumps(head, default=str)), **gate["state_dict"])
    np.savez(out.replace(".npz", ".lora.npz"), **lora)
    print(f"wrote {out} and {out.replace('.npz', '.lora.npz')}")


def check_gate(gate: dict) -> None:
    """Fail loudly on any gate variant this port does not implement."""
    required = {
        "selection_mode": "topk",
        "per_layer_gate": True,
        "shared_readout": True,
        "absorbed_readout": True,
        "readout_residual": False,
        "gate_branch_mode": "mlp",
        "normed_gate": True,
        "softplus_gate_scale": True,
        "use_attn_scale": True,
        "scaled_softmax_scale": True,
        "learnable_sss_scale": True,
        "head_agg": "avg",
    }
    for key, want in required.items():
        got = gate.get(key, want)
        if got != want:
            raise ValueError(f"unsupported gate: {key}={got!r} (this port implements {key}={want!r})")


def merge_lora(base_dir: Path, lora: dict, meta: dict, out_dir: Path) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    lora = {k: torch.from_numpy(v) for k, v in lora.items()}
    scaling = float(meta["lora_alpha"]) / float(meta["lora_rank"])
    shards = sorted(base_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {base_dir}")
    n_merged = 0
    for shard in shards:
        sd = load_file(str(shard))
        for name in list(sd):
            if not name.endswith(".weight"):
                continue
            stem = name[: -len(".weight")]
            a = lora.get(f"{stem}.lora_A")
            b = lora.get(f"{stem}.lora_B")
            if a is None:
                continue
            w = sd[name]
            delta = (a.float() @ b.float()).t() * scaling
            sd[name] = (w.float() + delta).to(w.dtype)
            n_merged += 1
        save_file(sd, str(out_dir / shard.name), metadata={"format": "pt"})
    n_lora = sum(1 for k in lora if k.endswith(".lora_A"))
    if n_merged != n_lora:
        raise RuntimeError(f"merged {n_merged} of {n_lora} LoRA pairs - name mismatch with the base model")
    for f in base_dir.iterdir():
        if f.suffix != ".safetensors" and f.is_file():
            shutil.copy(f, out_dir / f.name)
    print(f"merged {n_merged} LoRA pairs (scaling {scaling})")


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, np.asarray(x, dtype=np.float64))


def gated_layers(reader, arch: str, gate_layer: int) -> list[int]:
    """The layers above the gate that take the gated softmax: every one on a dense model, the full-attention ones
    on a hybrid model (recurrent layers have no softmax)."""
    n_layer = int(reader.fields[f"{arch}.block_count"].contents())
    nextn = reader.fields.get(f"{arch}.nextn_predict_layers")
    n_layer -= int(nextn.contents()) if nextn is not None else 0
    interval = reader.fields.get(f"{arch}.full_attention_interval")
    if interval is None:
        return list(range(gate_layer + 1, n_layer))
    k = int(interval.contents())
    return [il for il in range(gate_layer + 1, n_layer) if (il + 1) % k == 0]


def append_gate(src: Path, dst: Path, gate: dict, meta: dict, head_dim: int | None = None,
                neutral: bool = False) -> None:
    reader = gguf.GGUFReader(str(src))
    arch = str(reader.fields["general.architecture"].contents())
    gate_layer = int(meta["gate_layer"])
    if head_dim is None:
        key_len = reader.fields.get(f"{arch}.attention.key_length")
        if key_len is not None:
            head_dim = int(key_len.contents())
        else:
            head_dim = int(reader.fields[f"{arch}.embedding_length"].contents()) // \
                int(reader.fields[f"{arch}.attention.head_count"].contents())

    sd = gate["state_dict"]
    gs = softplus(sd["gate_scale"])  # [n_gated, n_head]
    gamma = softplus(sd["attn_scale"])  # [n_gated, n_head]
    if neutral:  # no bias, unit temperature: the gated layers reduce to the base model's softmax
        gs = np.zeros_like(gs)
        gamma = np.ones_like(gamma)
    n_gated, n_head = gs.shape
    layers = gated_layers(reader, arch, gate_layer)
    if n_gated != len(layers):
        raise ValueError(f"gate has {n_gated} gated layers, model has {len(layers)} above layer {gate_layer}: {layers}")

    writer = gguf.GGUFWriter(str(dst), arch=arch, endianess=reader.endianess)

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    writer.add_uint32(f"{arch}.tokengate.gate_layer", gate_layer)
    writer.add_float32(f"{arch}.tokengate.bnorm_weight", float(np.asarray(sd["b_norm.weight"]).reshape(-1)[0]))

    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    extra: list[tuple[str, np.ndarray]] = []
    br = "gate_branch."
    extra.append((f"blk.{gate_layer}.tg_gate.weight", np.asarray(sd[br + "gate_proj.weight"], np.float32).astype(np.float16)))
    extra.append((f"blk.{gate_layer}.tg_up.weight", np.asarray(sd[br + "up_proj.weight"], np.float32).astype(np.float16)))
    extra.append((f"blk.{gate_layer}.tg_down", np.asarray(sd[br + "down_proj.weight"], np.float32)))
    for row, il in enumerate(layers):
        qs = gamma[row]
        kb = gs[row] * gamma[row].mean() * math.sqrt(head_dim)
        extra.append((f"blk.{il}.tg_qscale", qs.astype(np.float32).reshape(n_head, 1)))
        extra.append((f"blk.{il}.tg_kbias", kb.astype(np.float32).reshape(n_head, 1, 1)))
    for name, arr in extra:
        writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data)
    for _, arr in extra:
        writer.write_tensor_data(arr)
    writer.close()
    print(f"appended {len(extra)} gate tensors (gate layer {gate_layer}, gated layers {layers} x {n_head} heads"
          f"{', NEUTRAL' if neutral else ''})")


# LoRA target module -> llama.cpp tensor of the same layer
LORA_TARGETS = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output"}


def write_lora_adapter(lora: dict, meta: dict, arch: str, out: str) -> None:
    """The paired LoRA as a llama.cpp adapter: blk.N.<tensor>.weight.lora_{a,b}, alpha in the metadata.

    The trainer stores A as (in, r) and B as (r, out) with W' = W + (alpha/r) (A B)^T; llama.cpp wants lora_a as
    ggml [in, r] (numpy (r, in)) and lora_b as ggml [r, out] (numpy (out, r)), and scales by alpha/r itself.
    """
    import re

    writer = gguf.GGUFWriter(out, arch=arch)
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", float(meta["lora_alpha"]))
    n = 0
    for name in sorted(lora):
        if not name.endswith(".lora_A"):
            continue
        m = re.search(r"layers\.(\d+)\..*\.([a-z_]+)\.lora_A$", name)
        if not m or m.group(2) not in LORA_TARGETS:
            raise ValueError(f"unsupported LoRA tensor {name}")
        a = np.asarray(lora[name], np.float32)
        b = np.asarray(lora[name[: -len("lora_A")] + "lora_B"], np.float32)
        base = f"blk.{int(m.group(1))}.{LORA_TARGETS[m.group(2)]}.weight"
        writer.add_tensor(f"{base}.lora_a", np.ascontiguousarray(a.T))
        writer.add_tensor(f"{base}.lora_b", np.ascontiguousarray(b.T))
        n += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {out}: {n} LoRA pairs, alpha {meta['lora_alpha']}, rank {meta.get('lora_rank')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", help="HF base model directory (e.g. Qwen3-1.7B): merge the LoRA and convert")
    ap.add_argument("--gguf", help="an existing GGUF of the base model: append the gate to it instead")
    ap.add_argument("--gate", required=True, help="gate_*.pt, or the .npz written by --export-npz")
    ap.add_argument("--lora", required=True, help="the paired lora_*.pt (or .lora.npz)")
    ap.add_argument("--outfile", help="the GGUF with the gate")
    ap.add_argument("--lora-out", help="with --gguf: write the LoRA as an adapter GGUF here")
    ap.add_argument("--export-npz", help="write the gate (and <name>.lora.npz) as numpy arrays and stop")
    ap.add_argument("--neutral", action="store_true", help="zero gate bias, unit temperature (split test)")
    ap.add_argument("--outtype", default="f16", help="passed to convert_hf_to_gguf.py")
    ap.add_argument("--tmpdir", default=None)
    args = ap.parse_args()

    gate = load_gate_file(args.gate)
    meta = gate.get("metadata", {})
    check_gate(gate)
    lora = load_lora_file(args.lora)
    print(f"gate: base_model={meta.get('base_model')} gate_layer={meta.get('gate_layer')} step={meta.get('step')}")

    if args.export_npz:
        export_npz(gate, lora, args.export_npz)
        return

    if args.gguf:
        if args.outfile:
            append_gate(Path(args.gguf), Path(args.outfile), gate, meta, neutral=args.neutral)
            print(f"wrote {args.outfile}")
        if args.lora_out:
            reader = gguf.GGUFReader(args.gguf)
            write_lora_adapter(lora, meta, str(reader.fields["general.architecture"].contents()), args.lora_out)
        return

    if not args.base or not args.outfile:
        ap.error("--base and --outfile (or --gguf) are required")
    base = Path(args.base)
    cfg = json.loads((base / "config.json").read_text())
    head_dim = int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"])

    with tempfile.TemporaryDirectory(dir=args.tmpdir) as tmp:
        merged = Path(tmp) / "merged"
        merged.mkdir()
        merge_lora(base, lora, meta, merged)

        plain = Path(tmp) / "plain.gguf"
        subprocess.run([sys.executable, str(REPO / "convert_hf_to_gguf.py"), str(merged),
                        "--outfile", str(plain), "--outtype", args.outtype], check=True)
        shutil.rmtree(merged)

        append_gate(plain, Path(args.outfile), gate, meta, head_dim, neutral=args.neutral)
    print(f"wrote {args.outfile}")


if __name__ == "__main__":
    main()
