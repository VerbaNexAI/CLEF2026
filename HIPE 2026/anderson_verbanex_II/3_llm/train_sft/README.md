# Create the dataset
python data/distill_dataset_po/create_training_dataset_nemotron.py  --at_model_path prompt_optimization/models/gepa_at_gpt-oss-20b.json   --isat_model_path prompt_optimization/models/gepa_isat_gpt-oss-20b.json

# Upload & train on Modal
modal run sft/nemotron_nano_sft_with_unsloth.py --action upload_data
modal volume put nemotron-nano-finetune-volume ./sft/nemotron_training_data.jsonl /data/nemotron_training_data.jsonl
modal run sft/nemotron_nano_sft_with_unsloth.py --action train  --repo-name andersonscode/Nemotron-3-Nano-4B-HIPE --quantization Q3_K_M,q4_k_m,q8_0




# Create a secret in Modal (replace <your_hf_token> with the token from
# https://huggingface.co/settings/tokens — do NOT commit the literal value)
modal secret create huggingface-token HF_TOKEN=<your_hf_token>

#    (local path → volume remote path)
modal run sft/nemotron_nano_sft_with_unsloth.py --action upload_data
modal volume put nemotron-nano-finetune-volume sft/dataset/nemotron_training_data_gpt-oss-20b.jsonl /data/nemotron_training_data.jsonl

# Train
modal run lfm2_5_sft_with_unsloth.py --action train --repo-name andersonscode/LFM2.5-1.2B-Finetuned-HIPE-moonshotai_kimi-k2.5 --quantization q4_k_m,q5_k_m,q8_0,f16

# 1. Create the data directory inside the Modal volume
modal run sft/nemotron_nano_sft_with_unsloth.py --action upload_data --quantization q4_k_m,q5_k_m,q8_0

# 3. Run training
modal run sft/nemotron_nano_sft_with_unsloth.py --action train

