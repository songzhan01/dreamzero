"""Offline T5 prompt embedding cache builder for DreamZero VLA training.

Walks a DROID dataset's parquet files, extracts every unique
language-instruction string across the three annotation columns
(`language_instruction`, `language_instruction_2`, `language_instruction_3`),
runs UMT5-XXL once on each (after the same template decoration the runtime
collator applies), and writes (sha256(input_ids), seq_len + bf16 emb) into an
LMDB store. The training runtime then attaches that cache via
`DREAMZERO_T5_CACHE_PATH` and skips the T5 GPU forward, freeing ~11 GB of
HBM per rank.

Output layout:
  <output-dir>/meta.json          # tokenizer + ckpt sha + entry count + shape
  <output-dir>/data/              # LMDB env (subdir mode)
    data.mdb / lock.mdb

Per-entry payload format is shared with the runtime reader; see
`groot/vla/model/dreamzero/cache/t5_prompt_cache.py`.

Usage example:
  python scripts/encode_t5_offline.py \
      --droid-data-root /pfs/.../datasets/dreamzero_droid \
      --t5-checkpoint /pfs/.../Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
      --tokenizer-path /pfs/.../Wan2.1-I2V-14B-480P/google/umt5-xxl \
      --output-dir ./caches/t5_droid_full \
      --batch-size 32
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch

# Make `groot.*` importable when running from a clean checkout.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lmdb  # noqa: E402

from groot.vla.model.dreamzero.cache.t5_prompt_cache import (  # noqa: E402
    hash_input_ids,
    pack_entry,
    read_meta,
    write_meta,
)
from groot.vla.model.dreamzero.modules.wan_video_text_encoder import (  # noqa: E402
    WanTextEncoder,
)
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (  # noqa: E402
    NEG_PROMPT,
    DefaultDataCollator,
)


# === DROID-specific runtime constants (mirror configs/* + transform code) ===

DROID_EMBODIMENT_ID = 17  # configs/model/dreamzero/transform/base.yaml: oxe_droid: 17

LANG_COLS: tuple[str, ...] = (
    "annotation.language.language_instruction",
    "annotation.language.language_instruction_2",
    "annotation.language.language_instruction_3",
)

# Mirrors the strip in dreamzero_cotrain._prepare_language so any prompt with
# these tags hashes the same way the runtime does (DROID prompts shouldn't
# contain them, but defensiveness costs nothing).
TAGS_TO_STRIP: tuple[str, ...] = ("<LAPA>", "<DREAM>", "<COTRAIN>")

# NEG_PROMPT is imported from dreamzero_cotrain (single source of truth).
# Stored alongside positives so the runtime negative-prompt path also lands
# in cache (T1 keeps positives only; negative is a single constant string,
# hashed once below as well).

# Verbatim copy of configs/model/dreamzero/transform/base.yaml. Only the
# oxe_droid id matters for DROID feature dispatch, but we pass the full map
# for API parity with the training-time collator.
EMBODIMENT_TAG_MAPPING: dict[str, int] = {
    "real_gr1_arms_only": 0,
    "real_gr1_arms_only_annotated": 1,
    "real_gr1_arms_waist": 2,
    "real_gr1_arms_waist_annotated": 3,
    "dexmg_gr1_arms_only_inspire": 4,
    "dexmg_gr1_arms_only_fourier": 5,
    "dexmg_gr1_arms_waist_fourier": 6,
    "robocasa_single_arm": 7,
    "onex_eve_gripper": 8,
    "robocasa_gr1_arms_only_inspire_hands": 9,
    "robocasa_gr1_arms_only_fourier_hands": 10,
    "robocasa_gr1_fixed_lower_body_inspire_hands": 11,
    "robocasa_gr1_fixed_lower_body_fourier_hands": 12,
    "robocasa_panda_omron": 13,
    "robocasa_bimanual_panda_parallel_gripper": 15,
    "robocasa_bimanual_panda_inspire_hand": 16,
    "oxe_droid": 17,
    "oxe_fractal": 18,
    "oxe_language_table": 19,
    "oxe_bridge": 20,
    "real_panda_single_arm": 21,
    "hot3d_hands_only": 23,
    "gr1_unified": 24,
    "robocasa_gr1_arms_waist_fourier_hands": 25,
    "agibot": 26,
    "lapa": 27,
}


# ---------- helpers ----------

def strip_tags(s: str) -> str:
    for t in TAGS_TO_STRIP:
        s = s.replace(t, "")
    return s


def load_tasks_lookup(meta_dir: Path) -> dict[int, str]:
    """meta/tasks.jsonl → {task_index: task_string}."""
    out: dict[int, str] = {}
    with open(meta_dir / "tasks.jsonl") as f:
        for line in f:
            row = json.loads(line)
            out[int(row["task_index"])] = row["task"]
    return out


def enumerate_parquet_prompts(
    parquet_path: Path, tasks_lookup: dict[int, str]
) -> set[str]:
    """All unique language strings across the three lang columns in this
    parquet. Numeric columns are resolved through tasks.jsonl, mirroring the
    runtime's `get_language` parquet-column branch."""
    df = pd.read_parquet(parquet_path, columns=list(LANG_COLS))
    out: set[str] = set()
    for col in LANG_COLS:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            for idx in s.dropna().unique():
                out.add(tasks_lookup[int(idx)])
        else:
            for v in s.dropna().astype(str).unique():
                out.add(v)
    return out


def chunked(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--droid-data-root", required=True, type=Path,
                    help="DROID root containing meta/tasks.jsonl + data/chunk-*/episode_*.parquet")
    ap.add_argument("--t5-checkpoint", required=True, type=str,
                    help="Path to models_t5_umt5-xxl-enc-bf16.pth")
    ap.add_argument("--tokenizer-path", required=True, type=str,
                    help="UMT5-XXL tokenizer path (e.g. <wan_ckpt>/google/umt5-xxl)")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Output cache directory (will hold meta.json + data/lmdb)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512,
                    help="Tokenizer max_length; must match runtime collator")
    ap.add_argument("--num-views", type=int, default=3,
                    help="Mirrors runtime num_views; DROID branch is num_views-independent")
    ap.add_argument("--map-size-gb", type=int, default=8,
                    help="LMDB map_size in GB (must exceed final lmdb data size)")
    ap.add_argument("--limit-episodes", type=int, default=0,
                    help="Cap parquet count for dry runs (0 = all)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Incremental mode: load existing LMDB keys at startup "
                         "and skip prompts whose hash is already cached. Lets "
                         "you grow the cache (e.g. add new prompts after a "
                         "tasks.jsonl extension) without re-encoding everything.")
    args = ap.parse_args()

    droid_root: Path = args.droid_data_root
    out_dir: Path = args.output_dir
    lmdb_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    print(f"[builder] droid_root      = {droid_root}")
    print(f"[builder] t5_checkpoint   = {args.t5_checkpoint}")
    print(f"[builder] tokenizer       = {args.tokenizer_path}")
    print(f"[builder] output_dir      = {out_dir}")
    print(f"[builder] batch_size      = {args.batch_size}")
    print(f"[builder] map_size_gb     = {args.map_size_gb}")

    # --- Stage 1: enumerate raw prompts ---
    print("[builder] Stage 1: load tasks.jsonl and walk parquets")
    tasks_lookup = load_tasks_lookup(droid_root / "meta")
    print(f"[builder]   tasks.jsonl entries: {len(tasks_lookup)}")

    parquets = sorted(droid_root.glob("data/chunk-*/episode_*.parquet"))
    if args.limit_episodes > 0:
        parquets = parquets[: args.limit_episodes]
    print(f"[builder]   parquets to scan: {len(parquets)}")
    if not parquets:
        raise SystemExit(f"No parquets under {droid_root / 'data'}")

    raw_prompts: set[str] = set()
    t0 = time.time()
    for i, p in enumerate(parquets):
        try:
            raw_prompts.update(enumerate_parquet_prompts(p, tasks_lookup))
        except Exception as e:
            print(f"[builder]   WARN parquet {p.name}: {e}")
        if (i + 1) % 100 == 0 or (i + 1) == len(parquets):
            print(
                f"[builder]   scanned {i+1}/{len(parquets)} parquets, "
                f"unique prompts so far: {len(raw_prompts)}, "
                f"elapsed {time.time()-t0:.1f}s"
            )

    cleaned_prompts = sorted({strip_tags(p).strip() for p in raw_prompts})
    cleaned_prompts = [p for p in cleaned_prompts if p]
    print(f"[builder] Stage 1 done. raw={len(raw_prompts)} → cleaned={len(cleaned_prompts)}")

    # --- Stage 2: build collator + load T5 ---
    print("[builder] Stage 2: load collator + T5 (this may take a few minutes for GPFS mmap)")
    t1 = time.time()
    collator = DefaultDataCollator(
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        num_views=args.num_views,
        embodiment_tag_mapping=EMBODIMENT_TAG_MAPPING,
    )
    print(f"[builder]   collator ready ({time.time()-t1:.1f}s)")

    t2 = time.time()
    text_encoder = WanTextEncoder()
    sd = torch.load(args.t5_checkpoint, map_location="cpu")
    text_encoder.load_state_dict(sd)
    del sd
    gc.collect()
    text_encoder = text_encoder.cuda().to(torch.bfloat16).eval()
    text_encoder.requires_grad_(False)
    print(f"[builder]   T5 loaded + moved to bf16 cuda ({time.time()-t2:.1f}s)")

    print("[builder]   computing T5 ckpt sha256 (used in meta.json for runtime validation)")
    t5_sha = sha256_file(args.t5_checkpoint)
    print(f"[builder]   T5 sha256: {t5_sha[:16]}…")

    # --- Stage 3: encode + write ---
    map_size = int(args.map_size_gb) << 30
    print(f"[builder] Stage 3: open lmdb at {lmdb_dir} (map_size={args.map_size_gb} GB)")
    env = lmdb.open(str(lmdb_dir), map_size=map_size, subdir=True,
                    writemap=False, sync=False)

    # Bootstrap: pre-populate `already_cached` with the LMDB's existing keys
    # in incremental mode. This lets the dedup loop below skip them naturally
    # without re-encoding. In default (full-rebuild) mode this is empty.
    #
    # CRITICAL: validate meta.json BEFORE adopting old keys. If the existing
    # cache was built against a different tokenizer or T5 checkpoint, mixing
    # old + new entries silently corrupts the store — runtime would later
    # see meta.json claiming the new sha while half the entries actually came
    # from the old encoder. Refuse the merge in that case; user must rebuild.
    already_cached: set[bytes] = set()
    if args.skip_existing:
        try:
            existing_meta = read_meta(str(out_dir))
        except FileNotFoundError:
            existing_meta = None
        if existing_meta is not None:
            if existing_meta.get("tokenizer") != "google/umt5-xxl":
                raise SystemExit(
                    f"[builder] --skip-existing refused: existing meta.json "
                    f"tokenizer={existing_meta.get('tokenizer')!r}, builder uses "
                    f"'google/umt5-xxl'. Drop --skip-existing or rebuild from scratch."
                )
            if existing_meta.get("t5_ckpt_sha256") != t5_sha:
                raise SystemExit(
                    f"[builder] --skip-existing refused: existing meta.json "
                    f"t5_ckpt_sha256={existing_meta.get('t5_ckpt_sha256','?')[:16]}…, "
                    f"new ckpt sha256={t5_sha[:16]}…. Mixing old/new T5 embeddings "
                    f"would silently corrupt the store. Drop --skip-existing or "
                    f"point --t5-checkpoint at the original ckpt."
                )
            print(
                f"[builder]   skip-existing meta check OK "
                f"(tokenizer + t5_ckpt_sha match)"
            )
        with env.begin() as txn:
            with txn.cursor() as cursor:
                for key, _ in cursor:
                    already_cached.add(bytes(key))
        print(
            f"[builder]   skip-existing mode: {len(already_cached)} pre-existing "
            f"entries will be skipped"
        )

    seen: set[bytes] = set()
    n_written = 0
    n_skip_dup = 0
    n_skip_existing = 0
    n_truncated = 0
    t3 = time.time()

    # T1 scope: positive prompts (DROID template decoration applied) plus the
    # one constant NEG_PROMPT (tokenized via collate's text_negative branch,
    # which skips template decoration). Caching the negative once costs ~1.5
    # MB but lets inference-with-CFG run in drop-T5 mode without crashing on
    # the negative path.
    enqueue: list[tuple[str, int]] = [(s, DROID_EMBODIMENT_ID) for s in cleaned_prompts]

    for batch_idx, batch in enumerate(chunked(enqueue, args.batch_size)):
        # Wrap raw prompt as `str([s])` so the runtime collator's
        # ast.literal_eval try-branch (dreamzero_cotrain.collate L101-129) is
        # exercised. The except-fallback branch produces the same final
        # string today, but staying on the try-path keeps builder + runtime
        # on a single code path so future edits to either branch can't drift.
        feats = [
            {"text": str([s]), "text_negative": NEG_PROMPT, "embodiment_id": eid}
            for (s, eid) in batch
        ]
        out = collator(feats)
        ids = out["text"]                    # (B, max_seq_len) long, cpu
        mask = out["text_attention_mask"]    # (B, max_seq_len) long, cpu

        keep_idx: List[int] = []
        keep_keys: List[bytes] = []
        for i in range(ids.shape[0]):
            k = hash_input_ids(ids[i])
            if k in already_cached:
                n_skip_existing += 1
                continue
            if k in seen:
                n_skip_dup += 1
                continue
            seen.add(k)
            keep_idx.append(i)
            keep_keys.append(k)
        if not keep_idx:
            continue

        ids_k = ids[keep_idx].cuda(non_blocking=True)
        mask_k = mask[keep_idx].cuda(non_blocking=True)
        with torch.no_grad():
            emb = text_encoder(ids_k, mask_k).clone().to(torch.bfloat16)
        seq_lens = mask_k.gt(0).sum(dim=1).long().tolist()
        # Over-length detection: any seq_len reaching max_length means the
        # tokenizer truncated this prompt. Embeddings still encode the kept
        # tokens, but trailing-token semantics are lost. Counted, surfaced at
        # the end so the dataset author can inspect.
        n_truncated += sum(1 for sl in seq_lens if sl >= args.max_length)

        with env.begin(write=True) as txn:
            for j, k in enumerate(keep_keys):
                sl = int(seq_lens[j])
                txn.put(k, pack_entry(sl, emb[j, :sl].contiguous()))
        n_written += len(keep_keys)

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - t3
            print(
                f"[builder]   batch {batch_idx+1}, written {n_written}, "
                f"dup_skip {n_skip_dup}, elapsed {elapsed:.1f}s, "
                f"throughput {n_written/max(elapsed,1e-3):.1f}/s"
            )

    env.sync()
    print(
        f"[builder] Stage 3 done. positive entries: {n_written}, "
        f"dup_skip: {n_skip_dup}, existing_skip: {n_skip_existing}"
    )

    # --- Stage 4: cache the constant negative prompt (single entry) ---
    # Runtime path: collate's text_negative branch (L156-160) tokenizes
    # without DROID template decoration, so we route NEG_PROMPT through the
    # same branch by feeding it as `text_negative` and reading the
    # corresponding output keys.
    print("[builder] Stage 4: encode constant NEG_PROMPT (no template decoration)")
    feats_neg = [{
        "text": str(["placeholder"]),  # try-branch parity; output discarded
        "text_negative": NEG_PROMPT,
        "embodiment_id": DROID_EMBODIMENT_ID,
    }]
    out_neg = collator(feats_neg)
    neg_ids = out_neg["text_negative"]                    # (1, max_seq_len)
    neg_mask = out_neg["text_attention_mask_negative"]    # (1, max_seq_len)
    neg_key = hash_input_ids(neg_ids[0])
    if neg_key in already_cached:
        print("[builder]   NEG_PROMPT already in cache (skipped)")
    elif neg_key in seen:
        print("[builder]   NEG_PROMPT collides with an existing positive key (skipped)")
    else:
        with torch.no_grad():
            neg_emb = text_encoder(
                neg_ids.cuda(non_blocking=True),
                neg_mask.cuda(non_blocking=True),
            ).clone().to(torch.bfloat16)
        neg_sl = int(neg_mask.gt(0).sum(dim=1).item())
        if neg_sl >= args.max_length:
            n_truncated += 1
        with env.begin(write=True) as txn:
            txn.put(neg_key, pack_entry(neg_sl, neg_emb[0, :neg_sl].contiguous()))
        n_written += 1
        seen.add(neg_key)
        print(f"[builder]   NEG_PROMPT cached (seq_len={neg_sl})")

    env.sync()
    env.close()
    print(f"[builder] writing meta.json")
    # In incremental mode, total count = newly-written + pre-existing (no
    # overlap because already_cached keys are skipped before encode). Equals
    # n_written in default full-rebuild mode (already_cached is empty).
    final_count = n_written + len(already_cached)
    write_meta(
        str(out_dir),
        count=final_count,
        tokenizer="google/umt5-xxl",
        t5_ckpt_sha=t5_sha,
        max_seq_len=args.max_length,
        hidden_dim=4096,
    )

    # Disk usage report
    try:
        size_bytes = sum(p.stat().st_size for p in lmdb_dir.glob("*"))
        print(f"[builder] lmdb size: {size_bytes / 1e9:.2f} GB")
    except Exception:
        pass

    if n_truncated > 0:
        print(
            f"[builder] WARNING: {n_truncated} prompt(s) reached "
            f"max_seq_len={args.max_length} (tokenizer truncated). "
            f"Trailing-token semantics are lost for those entries; inspect "
            f"the dataset's longest prompts. Bump --max-length and rebuild "
            f"if this is unexpected."
        )

    print(f"[builder] DONE. total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
