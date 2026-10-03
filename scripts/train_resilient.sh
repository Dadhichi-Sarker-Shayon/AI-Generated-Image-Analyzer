#!/bin/bash
# Re-launch scripts/train.py with --resume after a crash (e.g. Windows GPU driver reset) until it finishes.
OUT="G:/My Drive/AI Image analyzer/checkpoints_v2"
LOG=/c/ai_data/train_v2.log
for attempt in $(seq 1 30); do
  RESUME=""
  [ -f "$OUT/last_efficientnet_b0_binary.pt" ] && RESUME="--resume $OUT/last_efficientnet_b0_binary.pt"
  echo "=== attempt $attempt $(date) $RESUME" >> $LOG
  PYTHONIOENCODING=utf-8 python scripts/train.py --data-dir C:/ai_data/detector --model efficientnet_b0 \
    --epochs 20 --samples-per-epoch 28000 --batch-size 16 --accum-steps 2 --img-size 192 --num-workers 4 \
    --use-randaugment --loss-type label_smoothing --use-warmup --lr-warmup-epochs 1 \
    --early-stopping-patience 4 --output-dir "$OUT" $RESUME >> $LOG 2>&1
  grep -q "Best validation AUC" $LOG && break
  sleep 20
done
echo "=== FINISHED $(date)" >> $LOG
