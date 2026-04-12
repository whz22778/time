export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/NB15 \
  --model_id NB15 \
  --model TimesNet \
  --data NB15 \
  --features M \
  --seq_len 20 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 2 \
  --enc_in 23 \
  --c_out 23 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3