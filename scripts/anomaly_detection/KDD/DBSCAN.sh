export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/KDD \
  --model_id KDD_005_15s \
  --model dbscan \
  --data KDD \
  --features M \
  --seq_len 1 \
  --pred_len 0 \
  --anomaly_ratio 35 \
  --batch_size 64 \
  --train_epochs 1 \
  --dbscan_eps 0.05 \
  --dbscan_min_samples 15
