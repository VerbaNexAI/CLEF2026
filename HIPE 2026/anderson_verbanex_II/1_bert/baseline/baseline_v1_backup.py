"""Zero-shot NLI baseline for HIPE-2026 using multilingual DeBERTa."""

import argparse
import json
import sys
from pathlib import Path

from transformers import pipeline


def build_prompt(text: str, person: str, location: str, date: str | None) -> str:
    """Build a premise prompt for the NLI classifier."""
    date_str = f" (publication date: {date})" if date else ""
    # Truncate very long OCR texts to avoid hitting token limits
    max_len = 2000
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return (
        f"Document{date_str}: {text}\n\n"
        f"Based on the above document, consider the person '{person}' and the location '{location}'."
    )


def classify_at(classifier, text: str, person: str, location: str, date: str | None) -> str:
    """Classify the 'at' relationship (TRUE / PROBABLE / FALSE)."""
    prompt = build_prompt(text, person, location, date)
    labels = [
        f"{person} was at {location} at some point before the document was published.",
        f"{person} was probably at {location} before the document was published.",
        f"{person} was not at {location} before the document was published.",
    ]
    result = classifier(prompt, labels, multi_label=False)
    label_map = {
        labels[0]: "TRUE",
        labels[1]: "PROBABLE",
        labels[2]: "FALSE",
    }
    top_label = result["labels"][0]
    return label_map[top_label]


def classify_isat(classifier, text: str, person: str, location: str, date: str | None) -> str:
    """Classify the 'isAt' relationship (TRUE / FALSE)."""
    prompt = build_prompt(text, person, location, date)
    labels = [
        f"{person} was at {location} within approximately one month before the document was published.",
        f"{person} was not at {location} within approximately one month before the document was published.",
    ]
    result = classifier(prompt, labels, multi_label=False)
    label_map = {
        labels[0]: "TRUE",
        labels[1]: "FALSE",
    }
    top_label = result["labels"][0]
    return label_map[top_label]


def process_file(input_path: Path, output_path: Path, model_name: str, device: int, limit: int | None = None):
    """Process a HIPE-2026 JSONL file and write predictions."""
    print(f"Loading zero-shot NLI model: {model_name} on device {device}", file=sys.stderr)
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON line: {e}", file=sys.stderr)

    if limit is not None:
        records = records[:limit]

    total_pairs = sum(len(r.get("sampled_pairs", [])) for r in records)
    print(f"Processing {len(records)} documents, {total_pairs} pairs...", file=sys.stderr)

    processed = 0
    for record in records:
        text = record.get("text", "")
        date = record.get("date")
        for pair in record.get("sampled_pairs", []):
            person = ", ".join(pair.get("pers_mentions_list", []) or ["Unknown Person"])
            location = ", ".join(pair.get("loc_mentions_list", []) or ["Unknown Location"])

            pair["at"] = classify_at(classifier, text, person, location, date)
            pair["isAt"] = classify_isat(classifier, text, person, location, date)

            processed += 1
            if processed % 10 == 0 or processed == total_pairs:
                print(f"  {processed}/{total_pairs} pairs done ({processed / total_pairs * 100:.1f}%)", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote predictions to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot NLI baseline for HIPE-2026 person-place relation extraction."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path to the input HIPE-2026 JSONL file.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to the output JSONL file with predictions.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        help="HuggingFace model name for zero-shot NLI (default: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device to run inference on. -1 for CPU, >=0 for CUDA device index (default: -1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to the first N documents (for quick testing).",
    )
    args = parser.parse_args()

    process_file(
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
