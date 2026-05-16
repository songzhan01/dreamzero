"""Offline T5 prompt embedding cache.

Reader and serialization helpers shared between the cache builder
(scripts/encode_t5_offline.py) and the runtime
(WANPolicyHead.encode_prompt).

Storage layout:
- One LMDB environment (key-value store) keyed by sha256(input_ids.tobytes())
  where input_ids is the post-tokenizer T5 input row (shape (max_seq_len,) int).
- Each value is `struct.pack("<I", seq_len) + bf16_emb_as_uint16.tobytes()`,
  i.e. 4 bytes header + (seq_len * hidden_dim * 2) bytes payload. Trailing
  zero-padded positions (>= seq_len) are NOT stored — the runtime re-pads
  with zeros to match `prompt_emb[:, v:] = 0` semantics in encode_prompt.
- A sibling `meta.json` records tokenizer model id, T5 checkpoint sha256,
  build timestamp, and entry count for version validation.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import lmdb
except ImportError as e:
    raise ImportError(
        "lmdb is required for the T5 prompt cache. "
        "Install via `pip install lmdb` (use proxy http://192.168.112.80:18000 if no internet)."
    ) from e


META_FILENAME = "meta.json"


def hash_input_ids(ids_row: torch.Tensor) -> bytes:
    """sha256 of a single (max_seq_len,) input_ids row, as raw bytes.

    The tokenizer is deterministic, so sha256(input_ids) is a stable key
    for a given (raw_prompt, embodiment_id, decoration_template, tokenizer)
    tuple. Any upstream change that perturbs input_ids invalidates the
    entry naturally (cache miss).
    """
    arr = ids_row.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).digest()


def pack_entry(seq_len: int, emb_bf16: torch.Tensor) -> bytes:
    """Serialize (seq_len, embedding) to bytes.

    `emb_bf16` must be shape (seq_len, hidden_dim), dtype bfloat16, on any
    device (will be moved to CPU). bf16 is stored via uint16 reinterpretation
    because numpy lacks native bfloat16; the round-trip is exact.
    """
    assert emb_bf16.dtype == torch.bfloat16, f"expected bf16, got {emb_bf16.dtype}"
    assert emb_bf16.shape[0] == seq_len, f"seq_len mismatch: {emb_bf16.shape[0]} vs {seq_len}"
    u16 = emb_bf16.detach().cpu().contiguous().view(torch.uint16).numpy()
    return struct.pack("<I", seq_len) + u16.tobytes()


def unpack_entry(blob: bytes, hidden_dim: int) -> tuple[int, np.ndarray]:
    """Inverse of pack_entry. Returns (seq_len, np.uint16 array of shape (seq_len, hidden_dim))."""
    seq_len = struct.unpack("<I", blob[:4])[0]
    u16 = np.frombuffer(blob[4:], dtype=np.uint16).reshape(seq_len, hidden_dim)
    return seq_len, u16


def write_meta(cache_dir: str, *, count: int, tokenizer: str, t5_ckpt_sha: str,
               max_seq_len: int = 512, hidden_dim: int = 4096) -> None:
    """Write meta.json into the cache directory (parent of the lmdb data files)."""
    import time
    meta = {
        "tokenizer": tokenizer,
        "t5_ckpt_sha256": t5_ckpt_sha,
        "max_seq_len": max_seq_len,
        "hidden_dim": hidden_dim,
        "count": count,
        "build_timestamp": int(time.time()),
        "format_version": 1,
    }
    meta_path = Path(cache_dir) / META_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2))


def read_meta(cache_dir: str) -> dict:
    meta_path = Path(cache_dir) / META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json missing at {meta_path}")
    return json.loads(meta_path.read_text())


class T5PromptCache:
    """Read-only T5 prompt embedding cache.

    Open once per process (one rank); LMDB readonly + lock=False allows
    arbitrary number of concurrent readers across ranks. Each `lookup_batch`
    hashes every row in the input batch and atomically attempts to fetch
    all entries; if any row misses, returns None (caller decides fallback).
    """

    def __init__(self, path: str, *, max_seq_len: int = 512, hidden_dim: int = 4096,
                 expected_tokenizer: str = "google/umt5-xxl"):
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"T5 cache path does not exist: {path}")

        meta = read_meta(path)
        if meta["tokenizer"] != expected_tokenizer:
            raise ValueError(
                f"T5 cache tokenizer mismatch: meta.json says {meta['tokenizer']!r}, "
                f"runtime expects {expected_tokenizer!r}. Rebuild cache."
            )
        if meta["max_seq_len"] != max_seq_len or meta["hidden_dim"] != hidden_dim:
            raise ValueError(
                f"T5 cache shape mismatch: meta {meta['max_seq_len']}x{meta['hidden_dim']} "
                f"vs runtime {max_seq_len}x{hidden_dim}. Rebuild cache."
            )

        # Builder writes the lmdb data into a subdirectory `data/`; the
        # surrounding directory holds meta.json. This separates metadata
        # from lmdb's mmap files for cleaner version updates.
        lmdb_path = os.path.join(path, "data")
        layout = "subdir(data/)"
        if not os.path.exists(lmdb_path):
            # Fallback: legacy single-file layout (path itself is the lmdb env)
            lmdb_path = path
            layout = "legacy(path-as-env)"

        is_subdir = os.path.isdir(lmdb_path)
        print(
            f"[T5Cache] open {lmdb_path} layout={layout} "
            f"subdir={is_subdir} entries={meta['count']}"
        )
        self._env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=128,
            subdir=is_subdir,
        )
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.tokenizer = meta["tokenizer"]
        self.entry_count = meta["count"]
        self.miss_count = 0
        self.hit_count = 0

    def lookup_batch(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Try to fetch the embedding for every row in the batch.

        Returns:
            (B, max_seq_len, hidden_dim) bfloat16 tensor on `input_ids.device`
            with positions >= seq_len zeroed (matching encode_prompt
            semantics), or None if any row missed.
        """
        B = input_ids.shape[0]
        out = torch.zeros(
            (B, self.max_seq_len, self.hidden_dim),
            dtype=torch.bfloat16, device=input_ids.device,
        )
        with self._env.begin(buffers=False) as txn:
            for i in range(B):
                key = hash_input_ids(input_ids[i])
                blob = txn.get(key)
                if blob is None:
                    self.miss_count += 1
                    return None
                seq_len, u16 = unpack_entry(blob, self.hidden_dim)
                # zero-copy view into bf16; copy_ moves to GPU
                tensor_cpu = torch.from_numpy(u16.copy()).view(torch.bfloat16)
                out[i, :seq_len].copy_(tensor_cpu, non_blocking=True)
        self.hit_count += B
        return out

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
