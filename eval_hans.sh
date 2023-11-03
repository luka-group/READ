CUDA_VISIBLE_DEVICES=3 python run_text_classification.py \
  --model_name_or_path outputs/temp/main \
  --model_stage eval_main \
  --dataset_name hans \
  --do_eval \
  --max_seq_length 128 \
  --per_device_eval_batch_size 32 \
  --output_dir outputs/temp/hans