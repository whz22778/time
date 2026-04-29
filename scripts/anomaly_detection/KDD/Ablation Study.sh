#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

for i in {0..6}
do
    echo "正在测试 KDD 消融实验: asn=$i"
    python -u run.py \
      --task_name anomaly_detection \
      --is_training 1 \
      --root_path ./dataset/KDD \
      --model_id "KDD_asn$i" \
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
      --n_clusters 5 \
      --feature_k 4 \
      --interaction_topk fixed \
      --cluster_init kmeans \
      --cluster_freeze 1 \
      --anomaly_ratio 41 \
      --batch_size 128 \
      --train_epochs 3 \
      --result 0 \
      --asn $i
done