nohup python -u finetune.py \
  --base_model './llama_model/' \
  --data_path './alpaca_gpt4_data_enzh_40000.json' \
  --batch_size 128  \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --wandb_project 'alpaca-lora' \
  --wandb_run_name 'alpaca-lora-en&zh' \
  --wandb_watch 'all' \
  >> log.out 2>&1 &