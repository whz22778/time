export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/KDD \
  --model_id KDD \
  --model dbscan \
  --data KDD \
  --features M \
  --seq_len 20 \
  --pred_len 0 \
  --anomaly_ratio 35 \
  --batch_size 128 \
  --train_epochs 1 \
  --dbscan_eps 0.6 \
  --dbscan_min_samples 5
