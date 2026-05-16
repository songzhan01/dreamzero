#!/bin/bash
# DreamZero DROID Full Fine-Tuning Script with optimization knobs (8x H100/Pro6000, ZeRO-2).
#
# Same interface as scripts/train/droid_training_full_finetune.sh; adds perf
# env knobs (OMP threads, DiT torch.compile, T5 prompt cache) and switches
# DeepSpeed at runtime: when T5 is dropped via the prompt cache the optimizer
# state stays on GPU (zero2_no_offload_optimized.json, ~5 s/step optimizer
# win); without that headroom it falls back to CPU offload
# (zero2_offload_optimized.json). Both configs enable overlap_comm=true and
# reduce_bucket_size=5e8 vs the base script.
#
# Usage:
#   bash scripts/train/droid_training_full_finetune_optimization.sh
#
# Prerequisites:
#   - DROID dataset in LeRobot format at DROID_DATA_ROOT
#     Download: huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
#     Or convert from scratch: see scripts/data/convert_droid.py
#   - Wan2.1-I2V-14B-480P weights (auto-downloaded or pre-downloaded from HuggingFace)
#     Download: huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
#   - umt5-xxl tokenizer (auto-downloaded or pre-downloaded from HuggingFace)
#     Download: huggingface-cli download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
#   - Optional: T5 prompt embedding cache (drops T5 from GPU, frees ~11 GB).
#     Build: python scripts/encode_t5_offline.py --droid-data-root ... --output-dir ./caches/t5_droid_full
#     Enable: export DREAMZERO_T5_CACHE_PATH=./caches/t5_droid_full

export HYDRA_FULL_ERROR=1

# ============ USER CONFIGURATION ============
# Dataset path (DROID in LeRobot format)
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"./data/droid_lerobot"}

# Output directory for training checkpoints
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_droid_full_finetune"}

# Number of GPUs to use (8x H100/Pro6000 for ZeRO-2 full fine-tuning)
NUM_GPUS=${NUM_GPUS:-8}

# Per-GPU batch size. Default 2; relies on the T5 cache + no-offload path to
# fit on 96 GB Pro6000. Set BS=1 to fall back to the safe baseline.
# Exported so the dataset worker processes can see it (BS>1 triggers the
# ragged-chunk skip in lerobot_sharded.py).
export BS=${BS:-2}

# Model weight paths (download from HuggingFace if not already present)
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
# ================================================

# Validate dataset exists
if [ ! -d "$DROID_DATA_ROOT" ]; then
    echo "ERROR: DROID dataset not found at $DROID_DATA_ROOT"
    echo "Download with: huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir $DROID_DATA_ROOT"
    exit 1
fi

# ============ PERF KNOBS ============
# torchrun pins OMP_NUM_THREADS=1 by default, which slows DeepSpeedCPUAdam to a
# single thread per rank (~5.8 s/step gap on 14B). 8 ranks * 20 = 160 host
# threads stays under typical 180-CPU host budgets.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-20}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}

# DiT torch.compile toggle (default ON; set DREAMZERO_COMPILE=0 to disable).
# Required on Pro6000 (sm_120, 99 KB shared-mem cap): mode=default and
# TORCHINDUCTOR_MIX_ORDER_REDUCTION=0; reduce-overhead overflows the cap.
export DREAMZERO_COMPILE=${DREAMZERO_COMPILE:-1}
export DREAMZERO_COMPILE_MODE=${DREAMZERO_COMPILE_MODE:-default}
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=${TORCHINDUCTOR_MIX_ORDER_REDUCTION:-0}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"./inductor_cache"}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# T5 prompt cache toggle (resolution priority in encode_prompt: env > yaml > None).
# `${VAR+x}` distinguishes "unset" from "set to empty" so the yaml default
# applies cleanly when the caller leaves the env unset.
#   unset            -> yaml default (no cache, original T5 forward)
#   set to a path    -> drop T5 from GPU, lookup-only (~11 GB headroom)
if [[ -n "${DREAMZERO_T5_CACHE_PATH+x}" ]]; then export DREAMZERO_T5_CACHE_PATH; fi

# Auto-pick DeepSpeed config. When T5 is dropped from GPU (cache set) the
# freed ~11 GB lets the optimizer state stay on GPU instead of CPU offload
# — saves ~5 s/step on the optimizer step. Without that headroom we keep
# offload to avoid OOM.
if [[ -n "${DREAMZERO_T5_CACHE_PATH:-}" ]]; then
    DS_CONFIG="groot/vla/configs/deepspeed/zero2_no_offload_optimized.json"
    echo "[optimization] T5 cache → no-offload (optim state on GPU)"
else
    DS_CONFIG="groot/vla/configs/deepspeed/zero2_offload_optimized.json"
    echo "[optimization] No T5 headroom → CPU offload (optim state on CPU)"
fi
# ====================================

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/droid_relative \
    wandb_project=dreamzero \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="$DS_CONFIG" \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$BS \
    max_steps=100 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=no \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
