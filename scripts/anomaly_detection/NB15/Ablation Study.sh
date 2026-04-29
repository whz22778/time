#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

for i in {0..6}
do
    echo "正在测试 NB15 消融实验: asn=$i"
    python -u run.py \
      --task_name anomaly_detection \
      --is_training 1 \
      --root_path ./dataset/NB15 \
      --model_id "NB15_asn$i" \
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
      --result 0 \
      --asn $i
done