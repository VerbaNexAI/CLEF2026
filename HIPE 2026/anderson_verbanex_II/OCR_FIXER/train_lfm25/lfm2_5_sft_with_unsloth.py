# -*- coding: utf-8 -*-
"""
💧 LFM2.5 SFT with Unsloth on Modal

Based on the Unsloth Colab notebook, adapted to run on Modal's cloud infrastructure.

To run this:
1. modal run lfm2_5_sft_with_unsloth.py --action upload_data
2. Upload your training data to Modal volume
3. Train: modal run lfm2_5_sft_with_unsloth.py --action train
4. Train + Auto-Deploy: modal run lfm2_5_sft_with_unsloth.py --action train --repo-name yourusername/model-name
5. Test: modal run lfm2_5_sft_with_unsloth.py --action test

Provide the HuggingFace token via the ``HF_TOKEN`` environment variable or
the ``--hf-token`` CLI flag; never commit the literal value to the repo.
Generate one at https://huggingface.co/settings/tokens and replace the
placeholder username in ``--repo-name`` accordingly.
"""

import json
import os
from pathlib import Path

import modal

# Configuration
VOL_MOUNT_PATH = Path("/vol")
BASE_MODEL = "unsloth/LFM2.5-1.2B-Instruct"

# Read the HuggingFace token from the environment at import time so the
# literal value never has to live inside this file. Override at runtime
# with --hf-token if a different identity is needed.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Training configuration
MAX_SEQ_LENGTH = 2048
R = 16
LORA_ALPHA = 16
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_STEPS = 60
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

# Create Modal image with all dependencies
# Matches the working Colab install pattern from lfm2_5_sft_with_unsloth_example.py
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "curl", "libcurl4-openssl-dev", "libssl-dev", "git", "cmake", "build-essential",
    )
    .pip_install(
        "torch",
        "torchvision",
        "sentencepiece",
        "protobuf",
        "datasets==4.3.0",
        "huggingface-hub>=0.34.0",
        "hf_transfer",
        "transformers==4.57.3",
        "psutil",
    )
    .run_commands(
        "pip install --no-deps bitsandbytes accelerate xformers peft trl triton cut_cross_entropy unsloth_zoo",
        "pip install --no-deps unsloth",
        "pip install --no-deps trl==0.22.2",
    )
)

app = modal.App(name="lfm2.5-finetune", image=image)

# Create persistent volume for data and models
output_vol = modal.Volume.from_name("lfm2.5-finetune-volume", create_if_missing=True)


@app.function(
    gpu="A10G",  # Unsloth works well with T4, can use A10G or A100 for larger models
    timeout=7200,  # 2 hours
    volumes={VOL_MOUNT_PATH: output_vol},
)
def finetune(
    model_id: str = BASE_MODEL,
    max_seq_length: int = MAX_SEQ_LENGTH,
    batch_size: int = BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    max_steps: int = MAX_STEPS,
    learning_rate: float = LEARNING_RATE,
    num_train_epochs: int = NUM_EPOCHS,
    output_model_name: str = None,
    repo_name: str = None,
    hf_token: str = None,
    quantization_method: str = "f16",
):
    """
    Finetune LFM2.5 model on the provided dataset using Unsloth.

    Args:
        model_id: HuggingFace model ID to finetune
        max_seq_length: Maximum sequence length
        batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        max_steps: Maximum training steps (use None for full epochs)
        learning_rate: Learning rate
        num_train_epochs: Number of training epochs (if max_steps is None)
        output_model_name: Name for the output model directory
    """
    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import standardize_data_formats, train_on_responses_only

    # Generate output model name
    if output_model_name is None:
        model_name = model_id.split("/")[-1]
        output_model_name = f"finetuned_{model_name}"

    print(f"Starting finetuning of {model_id}")
    print(f"Configuration: batch_size={batch_size}, max_steps={max_steps}, lr={learning_rate}")

    # Load model with Unsloth
    print("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=R,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj",
                        "w1", "w2", "w3"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Check if data files exist
    data_path = VOL_MOUNT_PATH / "data" / "training_data.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}. Please upload first.")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    # Standardize dataset format
    dataset = standardize_data_formats(dataset)

    # Apply chat template
    def formatting_prompts_func(examples):
        texts = tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": [x.removeprefix(tokenizer.bos_token) for x in texts]}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Remove conversation-style columns so TRL 0.22 doesn't auto-detect them
    # and override dataset_text_field="text" with its own collator.
    cols_to_remove = [c for c in [
        "messages", "conversations", "prompt", "completion", "original_text"
    ] if c in dataset.column_names]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    # Setup training arguments
    training_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=5,
        max_steps=max_steps if max_steps else None,
        num_train_epochs=num_train_epochs if not max_steps else 1,
        learning_rate=learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=training_args,
    )

    # Train only on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    print("Starting training...")
    trainer_stats = trainer.train()

    # Show final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save model locally to volume
    output_path = VOL_MOUNT_PATH / "models" / output_model_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to {output_path}...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Commit volume
    output_vol.commit()

    print("✅ Finetuning completed successfully!")
    print(f"Model saved to: {output_path}")

    # Push to HuggingFace Hub as merged weights + GGUF if repo_name is provided
    # The merged weights are required for vLLM serving;
    # GGUF is kept for llama.cpp / local use.
    if repo_name and hf_token:
        from huggingface_hub import login

        login(token=hf_token)

        print(f"\nPushing merged 16-bit model to https://huggingface.co/{repo_name}")
        model.push_to_hub_merged(
            repo_name,
            tokenizer,
            save_method="merged_16bit",
            token=hf_token,
        )
        print(f"✅ Merged model pushed to https://huggingface.co/{repo_name}")

        print(f"\nPushing GGUF to https://huggingface.co/{repo_name}")
        print(f"Quantization: {quantization_method}")
        model.push_to_hub_gguf(
            repo_name,
            tokenizer,
            quantization_method=quantization_method,
            token=hf_token,
        )
        print(f"✅ GGUF pushed to https://huggingface.co/{repo_name}")
    else:
        print(f"\nTo download locally:")
        print(f"  modal volume get lfm2.5-finetune-volume /models/{output_model_name} ./{output_model_name}")

    return str(output_path)


@app.function(volumes={VOL_MOUNT_PATH: output_vol})
def upload_data():
    """
    Setup data directory in Modal volume.
    Run this first, then upload your training_data.jsonl file.
    """
    data_dir = VOL_MOUNT_PATH / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory created at: {data_dir}")
    print("Upload your training data:")
    print(f"  modal volume put lfm2.5-finetune-volume ./training_data.jsonl /data/training_data.jsonl")

    output_vol.commit()


@app.function(volumes={VOL_MOUNT_PATH: output_vol}, gpu="T4")
def push_to_hub(
    hf_token: str,
    model_name: str,
    repo_name: str,
    quantization_method: str | list[str] = "f16",
):
    """
    Fallback: Push a previously saved model to HuggingFace Hub as GGUF.
    Prefer using finetune() with repo_name param for best results.

    Args:
        hf_token: HuggingFace API token
        model_name: Name of the model directory in volume
        repo_name: Repository name on HuggingFace (e.g., "username/model-name")
        quantization_method: Quantization method(s) - single string or list of strings
                           Options: "f16", "q4_k_m", "q5_k_m", "q8_0", etc.
    """
    from huggingface_hub import login
    from unsloth import FastLanguageModel

    model_path = VOL_MOUNT_PATH / "models" / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
    )

    print(f"Logging in to HuggingFace Hub...")
    login(token=hf_token)

    print(f"Pushing model to https://huggingface.co/{repo_name}")
    print(f"Quantization: {quantization_method}")

    model.push_to_hub_gguf(
        repo_name,
        tokenizer,
        quantization_method=quantization_method,
        token=hf_token,
    )

    print(f"✅ Model successfully pushed to https://huggingface.co/{repo_name}")


@app.local_entrypoint()
def main(
    action: str = "train",
    model_id: str = BASE_MODEL,
    batch_size: int = BATCH_SIZE,
    max_steps: int = MAX_STEPS,
    epochs: int = NUM_EPOCHS,
    output_model_name: str = None,
    repo_name: str = None,
    quantization: str = "f16",
    hf_token: str = None,
):
    """
    Local entrypoint for running the finetuning pipeline.

    When action="train" and repo_name is provided, the model will automatically
    be deployed to HuggingFace Hub after training completes.

    The HuggingFace token is resolved at runtime from the ``HF_TOKEN``
    environment variable (or the ``--hf-token`` CLI override). Do not hard
    code the literal value in this file. Replace 'hf' with your own user
    when passing ``--repo-name``.

    Args:
        action: Action to perform (train, upload_data)
        model_id: Model ID to finetune
        batch_size: Training batch size
        max_steps: Maximum training steps
        epochs: Number of epochs (if max_steps is 0)
        output_model_name: Name of the finetuned model directory
        repo_name: HuggingFace repo name for auto-deployment (e.g., "yourusername/model-name")
                  If provided with --action train, model deploys automatically after training
        quantization: Quantization method(s) for GGUF upload, comma-separated for multiple
                      Options: f16, q4_k_m, q5_k_m, q8_0, etc.
                      Example: "q4_k_m,q5_k_m,q8_0" for multiple formats
        hf_token: Optional override for the HF_TOKEN environment variable
    """
    if action == "upload_data":
        print("Setting up data directory...")
        upload_data.remote()
        print("\nNext, upload your data file:")
        print("  modal volume put lfm2.5-finetune-volume ./training_data.jsonl /data/training_data.jsonl")

    elif action == "train":
        print(f"Starting finetuning on Modal with GPU...")
        effective_max_steps = max_steps if max_steps > 0 else None
        effective_epochs = epochs if not effective_max_steps else 1

        if output_model_name is None:
            model_name_base = model_id.split("/")[-1]
            effective_model_name = f"finetuned_{model_name_base}"
        else:
            effective_model_name = output_model_name

        effective_hf_token = hf_token or HF_TOKEN

        # Parse quantization methods
        if "," in quantization:
            quant_methods = [q.strip() for q in quantization.split(",")]
        else:
            quant_methods = quantization

        # Train + push in one call (same model object, matches original notebook)
        result = finetune.remote(
            model_id=model_id,
            batch_size=batch_size,
            max_steps=effective_max_steps,
            num_train_epochs=effective_epochs,
            output_model_name=effective_model_name,
            repo_name=repo_name,
            hf_token=effective_hf_token if repo_name else None,
            quantization_method=quant_methods,
        )
        print(f"\n✅ Training complete! Model saved at: {result}")

        if repo_name:
            print(f"✅ Model also pushed to: https://huggingface.co/{repo_name}")
        else:
            print(f"\nTo auto-deploy after training, add: --repo-name yourusername/model-name")

    else:
        print(f"Unknown action: {action}")
        print("Valid actions: upload_data, train")


if __name__ == "__main__":
    import sys

    action = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(action=action)
