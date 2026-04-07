export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/KDD \
  --model_id KDD \
  --model birch \
  --data KDD \
  --features M \
  --seq_len 1 \
  --pred_len 0 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 1 \
  --birch_threshold 0.25 \
  --birch_branching_factor 50
