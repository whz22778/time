export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/CIC \
  --model_id CIC \
  --model kmeans \
  --data CIC \
  --features M \
  --seq_len 20 \
  --pred_len 0 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 1 \
  --kmeans_n_clusters 5
