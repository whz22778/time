export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/CIC \
  --model_id CIC \
  --model Transformer \
  --data CIC \
  --features M \
  --seq_len 30 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 20 \
  --c_out 20 \
  --anomaly_ratio 15 \
  --batch_size 256 \
  --train_epochs 3