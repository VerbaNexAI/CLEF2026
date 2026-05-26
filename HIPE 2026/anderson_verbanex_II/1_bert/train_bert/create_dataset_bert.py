"""Create NLI datasets for fine-tuning mDeBERTa on HIPE-2026 at/isAt tasks."""

import argparse
import json
import random
import sys
from pathlib import Path


def build_premise(
    text: str,
    person: str,
    location: str,
    date: str | None,
    text_max_chars: int | None = None,
) -> str:
    """Build the premise (context) for the NLI classifier.

    By default the full OCR ``text`` is kept; the training tokenizer then truncates
    to ``max_length``. Set ``text_max_chars`` to a positive int to cap characters
    (e.g. for smaller JSONL); that cap must match inference if you train on it.
    """
    date_str = f" (publication date: {date})" if date else ""
    if text_max_chars is not None and text_max_chars > 0 and len(text) > text_max_chars:
        text = text[:text_max_chars] + "..."
    return (
        f"Document{date_str}: {text}\n\n"
        f"Based on the above document, consider the person '{person}' and the location '{location}'."
    )


def build_hypothesis_at(person: str, location: str) -> str:
    return f"{person} was at {location} at some point before the document was published."


def build_hypothesis_isat(person: str, location: str) -> str:
    return f"{person} was at {location} within approximately one month before the document was published."


def load_jsonl_records(path: Path) -> list[dict]:
    """Load HIPE-2026 JSONL records."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON line: {e}", file=sys.stderr)
    return records


def create_at_dataset(records: list[dict], text_max_chars: int | None = None) -> list[dict]:
    """Convert HIPE-2026 records into NLI examples for the 'at' task."""
    label_map = {
        "TRUE": "entailment",
        "PROBABLE": "neutral",
        "FALSE": "contradiction",
    }
    examples = []
    for record in records:
        text = record.get("text", "")
        date = record.get("date")
        document_id = record.get("document_id", "")
        for pair in record.get("sampled_pairs", []):
            person = ", ".join(pair.get("pers_mentions_list", []) or ["Unknown Person"])
            location = ", ".join(pair.get("loc_mentions_list", []) or ["Unknown Location"])
            label = pair.get("at")
            if label is None:
                label = "FALSE"
            if label not in label_map:
                continue
            examples.append({
                "premise": build_premise(text, person, location, date, text_max_chars),
                "hypothesis": build_hypothesis_at(person, location),
                "label": label_map[label],
                "document_id": document_id,
            })
    return examples


def create_isat_dataset(records: list[dict], text_max_chars: int | None = None) -> list[dict]:
    """Convert HIPE-2026 records into NLI examples for the 'isAt' task."""
    label_map = {
        "TRUE": "entailment",
        "FALSE": "contradiction",
    }
    examples = []
    for record in records:
        text = record.get("text", "")
        date = record.get("date")
        document_id = record.get("document_id", "")
        for pair in record.get("sampled_pairs", []):
            person = ", ".join(pair.get("pers_mentions_list", []) or ["Unknown Person"])
            location = ", ".join(pair.get("loc_mentions_list", []) or ["Unknown Location"])
            label = pair.get("isAt")
            if label is None:
                label = "FALSE"
            if label not in label_map:
                continue
            examples.append({
                "premise": build_premise(text, person, location, date, text_max_chars),
                "hypothesis": build_hypothesis_isat(person, location),
                "label": label_map[label],
                "document_id": document_id,
            })
    return examples


def split_and_save(examples: list[dict], output_dir: Path, name: str, seed: int):
    """Shuffle, split 80/20 train/val, and write CSVs."""
    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.8)
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)

    def _write(data: list[dict], suffix: str):
        path = output_dir / f"{name}_{suffix}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} examples to {path}", file=sys.stderr)

    _write(train, "train")
    _write(val, "val")

    # Print label distribution
    for split, data in [("train", train), ("val", val)]:
        counts = {}
        for ex in data:
            counts[ex["label"]] = counts.get(ex["label"], 0) + 1
        print(f"  {split} distribution: {counts}", file=sys.stderr)


def main():
    default_jsonl = Path(
        "data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-en-ocr-fix-deepseekv3.2.jsonl"
    )
    parser = argparse.ArgumentParser(
        description="Create NLI datasets for HIPE-2026 from one or more gold JSONL files "
        "(concatenated in the order given)."
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        nargs="+",
        default=[default_jsonl],
        metavar="PATH",
        help=(
            "One or more HIPE-2026 gold JSONL paths; documents are appended in order into "
            "a single corpus before splitting into train/val. "
            f"Default: {default_jsonl}"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("low_models/train_bert/dataset"),
        help="Directory to write CSV datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--text_max_chars",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, truncate each document's OCR text to N characters in the premise "
            "(omit for full text; tokenizer still truncates at train time via max_length). "
            "Must match inference when you rely on character caps."
        ),
    )
    args = parser.parse_args()

    records: list[dict] = []
    for jp in args.jsonl:
        if not jp.exists():
            raise FileNotFoundError(f"Input JSONL not found: {jp}")
        chunk = load_jsonl_records(jp)
        print(f"Loaded {len(chunk)} documents from {jp}", file=sys.stderr)
        records.extend(chunk)
    print(f"Total merged documents: {len(records)} (from {len(args.jsonl)} file(s))", file=sys.stderr)
    if args.text_max_chars is not None and args.text_max_chars <= 0:
        print(
            "Warning: --text_max_chars <= 0 is treated as no character cap.",
            file=sys.stderr,
        )
        text_cap: int | None = None
    else:
        text_cap = args.text_max_chars

    cap_msg = (
        "full OCR text in premise"
        if text_cap is None
        else f"OCR capped at {text_cap} chars in premise"
    )
    print(f"Premise text policy: {cap_msg}", file=sys.stderr)

    at_examples = create_at_dataset(records, text_cap)
    isat_examples = create_isat_dataset(records, text_cap)

    print(f"\nCreating 'at' dataset...", file=sys.stderr)
    split_and_save(at_examples, args.output_dir, "at_nli", args.seed)

    print(f"\nCreating 'isAt' dataset...", file=sys.stderr)
    split_and_save(isat_examples, args.output_dir, "isat_nli", args.seed)


if __name__ == "__main__":
    main()
