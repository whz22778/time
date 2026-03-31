export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/CIC \
  --model_id CIC \
  --model dbscan \
  --data CIC \
  --features M \
  --seq_len 20 \
  --pred_len 0 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 1 \
  --dbscan_eps 0.5 \
  --dbscan_min_samples 5
