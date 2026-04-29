#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 遍历 as 参数从 0 到 7
for i in {0..6}
do
    echo "正在测试 CIC 消融实验: asn=$i"
    python -u run.py \
      --task_name anomaly_detection \
      --is_training 1 \
      --root_path ./dataset/CIC \
      --model_id "CIC_asn$i" \
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
      --result 0 \
      --asn $i
done