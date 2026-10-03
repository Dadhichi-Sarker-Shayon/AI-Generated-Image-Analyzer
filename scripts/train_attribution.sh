#!/bin/bash
# Generator attribution (which AI generator made it): 11-way classifier on the ai/<generator> folders.
# Re-launches with --resume after a crash (e.g. Windows GPU driver reset).
OUT="G:/My Drive/AI Image analyzer/checkpoints_attr"
LOG=/c/ai_data/train_attr.log
GENS="adm biggan dalle3 glide midjourney sd15 sd21 sd3 sdxl vqdm wukong"
for attempt in $(seq 1 30); do
  RESUME=""
  [ -f "$OUT/last_efficientnet_b0_generator.pt" ] && RESUME="--resume $OUT/last_efficientnet_b0_generator.pt"
  echo "=== attempt $attempt $(date) $RESUME" >> $LOG
  PYTHONIOENCODING=utf-8 python scripts/train.py --data-dir C:/ai_data/detector --task generator --generators $GENS \
    --model efficientnet_b0 --epochs 12 --samples-per-epoch 28000 --batch-size 16 --accum-steps 2 --img-size 192 \
    --num-workers 4 --use-randaugment --loss-type label_smoothing --use-warmup --lr-warmup-epochs 1 \
    --early-stopping-metric acc --early-stopping-patience 4 --output-dir "$OUT" $RESUME >> $LOG 2>&1
  grep -q "Best validation AUC" $LOG && break
  sleep 20
done
echo "=== FINISHED $(date)" >> $LOG
