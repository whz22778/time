export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/CIC \
  --model_id CIC_ImprovedTimesNetV2 \
  --model ImprovedTimesNetV2 \
  --data CIC \
  --features M \
  --seq_len 40 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 2 \
  --enc_in 20 \
  --c_out 20 \
  --top_k 1 \
  --n_clusters 5 \
  --feature_k 6 \
  --interaction_topk fixed \
  --cluster_init kmeans \
  --cluster_freeze \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3
