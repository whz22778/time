export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/CIC \
  --model_id CIC_40f_10 \
  --model birch \
  --data CIC \
  --features M \
  --seq_len 1 \
  --pred_len 0 \
  --anomaly_ratio 10 \
  --batch_size 256 \
  --train_epochs 1 \
  --birch_threshold 0.5 \
  --birch_branching_factor 40
