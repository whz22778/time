export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/NB15 \
  --model_id NB15 \
  --model kmeans \
  --data NB15 \
  --features M \
  --seq_len 1 \
  --pred_len 0 \
  --anomaly_ratio 45 \
  --batch_size 128 \
  --train_epochs 1 \
  --kmeans_n_clusters 6
