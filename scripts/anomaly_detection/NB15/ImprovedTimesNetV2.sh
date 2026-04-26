export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/NB15 \
  --model_id NB15_4k_256 \
  --model ImprovedTimesNetV2 \
  --data NB15 \
  --features M \
  --seq_len 40 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 2 \
  --enc_in 23 \
  --c_out 23 \
  --top_k 1 \
  --n_clusters 7 \
  --feature_k 4 \
  --interaction_topk fixed \
  --cluster_init kmeans \
  --cluster_freeze 0 \
  --anomaly_ratio 55 \
  --batch_size 256 \
  --train_epochs 3 \
  --result 1
