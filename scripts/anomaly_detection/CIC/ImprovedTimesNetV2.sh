export CUDA_VISIBLE_DEVICES=0

  python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/CIC \
  --model_id CIC_5k \
  --model ImprovedTimesNetV2 \
  --data CIC \
  --features M \
  --seq_len 30 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 2 \
  --enc_in 20 \
  --c_out 20 \
  --top_k 1 \
  --n_clusters 6 \
  --feature_k 5 \
  --interaction_topk fixed \
  --cluster_init kmeans \
  --cluster_freeze 1 \
  --anomaly_ratio 6 \
  --batch_size 256 \
  --train_epochs 3 \
  --result 1
