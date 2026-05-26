"""Run the HIPE train_bert.py classifier training on Modal.

This wrapper keeps the local trainer unchanged and executes it on a larger GPU.
It packages only the trainer code/configs into the Modal image. Upload the
prepared dataset JSONL files manually to the Modal volume before training.

Example:
    modal run low_models/train_bert/train_bert_modal.py --action upload_data
    modal volume put hipe-train-bert-volume low_models/train_bert/dataset/at_nli_train.jsonl /data/dataset/at_nli_train.jsonl
    modal volume put hipe-train-bert-volume low_models/train_bert/dataset/at_nli_val.jsonl /data/dataset/at_nli_val.jsonl
    modal volume put hipe-train-bert-volume low_models/train_bert/dataset/isat_nli_train.jsonl /data/dataset/isat_nli_train.jsonl
    modal volume put hipe-train-bert-volume low_models/train_bert/dataset/isat_nli_val.jsonl /data/dataset/isat_nli_val.jsonl
    modal run low_models/train_bert/train_bert_modal.py   --config low_models/train_bert/configs/optimized.yaml   --task both
    modal run low_models/train_bert/train_bert_modal.py --config low_models/train_bert/configs/longctxlm.yaml
    modal run low_models/train_bert/train_bert_modal.py --action push_models_to_hub \
                                                        --run-name longctx_modernbert_v2 \ 
                                                        --at-repo-name andersonscode/ModernBERT-large-zeroshot-v2.0-HIPE-at-EN-all   \
                                                        --isat-repo-name andersonscode/ModernBERT-large-zeroshot-v2.0-HIPE-isat-EN-all

Download results:
    modal volume get hipe-train-bert-volume /runs/longctx_v1 ./longctx_v1
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path, PurePosixPath

import modal


PROJECT_ROOT = Path("/workspace/HIPE-2026-andersons")
VOL_MOUNT_PATH = Path("/vol")
# Modal add_local_* validates remote_path with PurePosixPath; str(Path(...)) on Windows
# uses backslashes, which fails that check — build Linux paths with PurePosixPath instead.
_IMAGE_TRAIN_BERT_DIR = PurePosixPath("/workspace/HIPE-2026-andersons/low_models/train_bert")
VOLUME_NAME = "hipe-train-bert-volume"

LOCAL_TRAIN_BERT_DIR = Path(__file__).resolve().parent
REMOTE_TRAIN_BERT_DIR = PROJECT_ROOT / "low_models" / "train_bert"
LOCAL_CONFIGS_DIR = LOCAL_TRAIN_BERT_DIR / "configs"

DEFAULT_CONFIG = "low_models/train_bert/configs/longctxlm.yaml"
DEFAULT_TASK = "both"
VOLUME_DATASET_DIR = VOL_MOUNT_PATH / "data" / "dataset"
REQUIRED_DATASET_FILES = (
    "at_nli_train.jsonl",
    "at_nli_val.jsonl",
    "isat_nli_train.jsonl",
    "isat_nli_val.jsonl",
)
MODEL_DIRS = {
    "at": Path("at_nli_model/final_model"),
    "isat": Path("isat_nli_model/final_model"),
}

# Trainer pickle sidecars — not needed for from_pretrained; trigger Hub/ClamAV scans.
# https://huggingface.co/docs/hub/en/security-pickle
_HF_UPLOAD_IGNORE_INFERENCE = (
    "training_args.bin",
    "optimizer.pt",
    "scheduler.pt",
    "scheduler.bin",
    "rng_state.pth",
)


image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        setup_dockerfile_commands=["ENTRYPOINT []", "CMD []"],
    )
    .pip_install(
        "accelerate>=0.26",
        "datasets>=2.16",
        "huggingface_hub>=0.24",
        "mlflow>=2.0",
        "numpy>=1.24",
        "protobuf>=4.25",
        "pyyaml>=6.0",
        "scikit-learn>=1.0",
        "sentencepiece>=0.1.99",
        "transformers>=4.40",
    )
    .add_local_file(
        LOCAL_TRAIN_BERT_DIR / "train_bert.py",
        remote_path=str(_IMAGE_TRAIN_BERT_DIR / "train_bert.py"),
    )
    .add_local_dir(
        LOCAL_CONFIGS_DIR,
        remote_path=str(_IMAGE_TRAIN_BERT_DIR / "configs"),
    )
)

app = modal.App(name="hipe-train-bert", image=image)
output_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    gpu="A10",
    timeout=24 * 60 * 60,
    volumes={VOL_MOUNT_PATH: output_vol},
    # Same HF_TOKEN as push_to_hub: avoids unauthenticated Hub throttling during model/tokenizer downloads.
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def train_on_modal(
    config: str = DEFAULT_CONFIG,
    task: str = DEFAULT_TASK,
    run_name: str | None = None,
) -> str:
    """Run low_models/train_bert/train_bert.py and persist artifacts to volume."""
    import os
    import yaml

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HOME", str(VOL_MOUNT_PATH / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(VOL_MOUNT_PATH / "hf_cache"))

    config_path = PROJECT_ROOT / config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found in Modal image: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    effective_run_name = run_name or cfg["run_name"]
    missing_files = [
        filename for filename in REQUIRED_DATASET_FILES if not (VOLUME_DATASET_DIR / filename).exists()
    ]
    if missing_files:
        formatted_missing = "\n  ".join(missing_files)
        raise FileNotFoundError(
            f"Dataset files are missing from {VOLUME_DATASET_DIR}:\n  {formatted_missing}\n"
            f"Run:\n"
            f"  modal run low_models/train_bert/train_bert_modal.py --action upload_data\n"
            f"  modal volume put {VOLUME_NAME} low_models/train_bert/dataset/at_nli_train.jsonl /data/dataset/at_nli_train.jsonl\n"
            f"  modal volume put {VOLUME_NAME} low_models/train_bert/dataset/at_nli_val.jsonl /data/dataset/at_nli_val.jsonl\n"
            f"  modal volume put {VOLUME_NAME} low_models/train_bert/dataset/isat_nli_train.jsonl /data/dataset/isat_nli_train.jsonl\n"
            f"  modal volume put {VOLUME_NAME} low_models/train_bert/dataset/isat_nli_val.jsonl /data/dataset/isat_nli_val.jsonl"
        )

    cfg["data_dir"] = str(VOLUME_DATASET_DIR)
    runtime_config = REMOTE_TRAIN_BERT_DIR / "configs" / f"_modal_runtime_{effective_run_name}.yaml"
    with runtime_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    output_dir = PROJECT_ROOT / cfg.get("output_dir", "low_models/train_bert/runs")
    run_dir = output_dir / effective_run_name

    command = [
        "python",
        "low_models/train_bert/train_bert.py",
        "--config",
        str(runtime_config),
        "--task",
        task,
    ]
    if run_name:
        command.extend(["--run_name", run_name])

    print(f"Running on Modal GPU with config={config}, task={task}, run_name={effective_run_name}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    if not run_dir.exists():
        raise FileNotFoundError(f"Training completed but run directory was not found: {run_dir}")

    volume_run_dir = VOL_MOUNT_PATH / "runs" / effective_run_name
    if volume_run_dir.exists():
        shutil.rmtree(volume_run_dir)
    volume_run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, volume_run_dir)

    registry_path = PROJECT_ROOT / cfg.get("registry_path", "low_models/train_bert/registry.csv")
    if registry_path.exists():
        registry_dest = VOL_MOUNT_PATH / "registry.csv"
        shutil.copy2(registry_path, registry_dest)

    output_vol.commit()
    print(f"Training artifacts saved to Modal volume: {volume_run_dir}")
    return str(volume_run_dir)


@app.function(volumes={VOL_MOUNT_PATH: output_vol})
def upload_data() -> None:
    """Create data/output directories in the Modal volume."""
    VOLUME_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (VOL_MOUNT_PATH / "runs").mkdir(parents=True, exist_ok=True)
    (VOL_MOUNT_PATH / "hf_cache").mkdir(parents=True, exist_ok=True)
    output_vol.commit()
    print(f"Modal volume ready: {VOLUME_NAME}")
    print("Upload the prepared dataset files with:")
    for filename in REQUIRED_DATASET_FILES:
        print(
            f"  modal volume put {VOLUME_NAME} "
            f"low_models/train_bert/dataset/{filename} /data/dataset/{filename}"
        )


@app.function(
    timeout=60 * 60,
    volumes={VOL_MOUNT_PATH: output_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def push_run_to_hub(
    run_name: str,
    repo_name: str,
    private: bool = False,
) -> str:
    """Upload a saved train_bert run folder from the Modal volume to Hugging Face."""
    import os

    from huggingface_hub import HfApi

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN was not found. Create the Modal secret first:\n"
            "  modal secret create huggingface-token HF_TOKEN=your_huggingface_token"
        )

    run_dir = VOL_MOUNT_PATH / "runs" / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found in Modal volume: {run_dir}")

    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=repo_name,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_name,
        repo_type="model",
        folder_path=str(run_dir),
        path_in_repo=".",
        commit_message=f"Upload train_bert run {run_name}",
    )
    print(f"Uploaded {run_dir} to https://huggingface.co/{repo_name}")
    return f"https://huggingface.co/{repo_name}"


@app.function(
    timeout=60 * 60,
    volumes={VOL_MOUNT_PATH: output_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def push_models_to_hub(
    run_name: str,
    repo_name: str,
    at_repo_name: str | None = None,
    isat_repo_name: str | None = None,
    private: bool = False,
) -> dict[str, str]:
    """Upload the two saved classifier models from a Modal volume run."""
    import os

    from huggingface_hub import HfApi

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN was not found. Create the Modal secret first:\n"
            "  modal secret create huggingface-token HF_TOKEN=your_huggingface_token"
        )

    run_dir = VOL_MOUNT_PATH / "runs" / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found in Modal volume: {run_dir}")

    repos = {
        "at": at_repo_name or f"{repo_name}-at",
        "isat": isat_repo_name or f"{repo_name}-isat",
    }

    api = HfApi(token=hf_token)
    uploaded_urls: dict[str, str] = {}
    for task_name, model_subdir in MODEL_DIRS.items():
        model_dir = run_dir / model_subdir
        if not model_dir.exists():
            raise FileNotFoundError(f"{task_name} model folder not found: {model_dir}")

        task_repo = repos[task_name]
        api.create_repo(
            repo_id=task_repo,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=task_repo,
            repo_type="model",
            folder_path=str(model_dir),
            path_in_repo=".",
            commit_message=f"Upload {task_name} train_bert model from run {run_name}",
            ignore_patterns=list(_HF_UPLOAD_IGNORE_INFERENCE),
        )
        uploaded_urls[task_name] = f"https://huggingface.co/{task_repo}"
        print(f"Uploaded {task_name} model from {model_dir} to {uploaded_urls[task_name]}")

    return uploaded_urls


@app.local_entrypoint()
def main(
    action: str = "train",
    config: str = DEFAULT_CONFIG,
    task: str = DEFAULT_TASK,
    run_name: str | None = None,
    repo_name: str | None = None,
    at_repo_name: str | None = None,
    isat_repo_name: str | None = None,
    private: bool = False,
) -> None:
    """Local entrypoint for Modal CLI."""
    if action in {"upload_data", "setup_volume"}:
        upload_data.remote()
        print(f"Volume initialized: {VOLUME_NAME}")
        return

    if action == "push_to_hub":
        if not run_name:
            print("For action=push_to_hub, pass --run-name <run_name>.")
            return
        if not repo_name:
            print("For action=push_to_hub, pass --repo-name yourusername/model-name.")
            return
        repo_url = push_run_to_hub.remote(
            run_name=run_name,
            repo_name=repo_name,
            private=private,
        )
        print(f"Uploaded run to Hugging Face: {repo_url}")
        return

    if action == "push_models_to_hub":
        if not run_name:
            print("For action=push_models_to_hub, pass --run-name <run_name>.")
            return
        if not repo_name and not (at_repo_name and isat_repo_name):
            print(
                "For action=push_models_to_hub, pass --repo-name base/name "
                "or both --at-repo-name and --isat-repo-name."
            )
            return
        urls = push_models_to_hub.remote(
            run_name=run_name,
            repo_name=repo_name or "unused/base-name",
            at_repo_name=at_repo_name,
            isat_repo_name=isat_repo_name,
            private=private,
        )
        print(f"Uploaded at model: {urls['at']}")
        print(f"Uploaded isAt model: {urls['isat']}")
        return

    if action != "train":
        print(
            "Unknown action. Valid actions: upload_data, setup_volume, train, "
            "push_to_hub, push_models_to_hub"
        )
        return

    result_path = train_on_modal.remote(
        config=config,
        task=task,
        run_name=run_name,
    )
    effective_run_name = Path(result_path).name
    print(f"Training complete. Artifacts are in Modal volume at: {result_path}")
    if repo_name or (at_repo_name and isat_repo_name):
        urls = push_models_to_hub.remote(
            run_name=effective_run_name,
            repo_name=repo_name or "unused/base-name",
            at_repo_name=at_repo_name,
            isat_repo_name=isat_repo_name,
            private=private,
        )
        print(f"Uploaded at model: {urls['at']}")
        print(f"Uploaded isAt model: {urls['isat']}")
    print("Download with:")
    print(f"  modal volume get {VOLUME_NAME} {result_path.replace(str(VOL_MOUNT_PATH), '')} ./{effective_run_name}")

