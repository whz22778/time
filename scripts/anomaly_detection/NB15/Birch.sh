export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/NB15 \
  --model_id NB15_95f_45t \
  --model birch \
  --data NB15 \
  --features M \
  --seq_len 1 \
  --pred_len 0 \
  --anomaly_ratio 0.5 \
  --batch_size 128 \
  --train_epochs 1 \
  --birch_threshold 0.45 \
  --birch_branching_factor 95
