export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/KDD \
  --model_id KDD_5k_ImprovedTimesNetV2 \
  --model ImprovedTimesNetV2 \
  --data KDD \
  --features M \
  --seq_len 40 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 2 \
  --enc_in 23 \
  --c_out 23 \
  --top_k 5 \
  --n_clusters 6 \
  --feature_k 6 \
  --interaction_topk fixed \
  --cluster_init kmeans \
  --cluster_freeze \
  --anomaly_ratio 40 \
  --batch_size 128 \
  --train_epochs 3
