"""Fix OCR and optionally translate HIPE-2026 newspaper text using an OpenAI-compatible LLM.

This script reads a HIPE-2026 JSONL file and rewrites each document's `text` field
by calling an LLM (OpenRouter, LM Studio, or a fine-tuned local model). By default it
requests OCR correction and translation to English; use ``--no_translate`` to only
correct OCR while keeping the document language.

"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

DEFAULT_MODEL = "openrouter/openai/gpt-oss-120b"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_MAX_RETRIES = 2


SYSTEM_PROMPT_TEMPLATE = (
    "You are given a historical newspaper text. The text may contain OCR errors "
    "(broken words, missing letters, substituted characters, garbled words). "
    "The text is in {language}.\n\n"
    "Your task is to:\n"
    "1. Fix all OCR errors in the text. Correct broken words, fix character substitutions, "
    "restore readability, and fix spelling mistakes while preserving the original paragraph "
    "and line structure as much as possible.\n"
    "2. Translate the fully corrected text into fluent, natural English.\n\n"
    "Output ONLY the corrected and translated text. Do not add any introductions, explanations, "
    "or metadata."
)

SYSTEM_PROMPT_OCR_ONLY_TEMPLATE = (
    "You are given a historical newspaper text. The text may contain OCR errors "
    "(broken words, missing letters, substituted characters, garbled words). "
    "The text is in {language}.\n\n"
    "Your task is to fix all OCR errors: correct broken words, fix character substitutions, "
    "restore readability, and fix spelling mistakes while preserving the original paragraph "
    "and line structure as much as possible.\n\n"
    "Output the corrected text **in the same language** ({language}). Do not translate. "
    "Do not add any introductions, explanations, or metadata."
)


async def fix_document_text(
    client: AsyncOpenAI,
    model: str,
    text: str,
    language: str,
    max_retries: int,
    no_translate: bool,
) -> str:
    """Call the LLM to fix OCR (and optionally translate) a single document's text."""
    if not text or not text.strip():
        return text

    lang = language if language else "the original language"
    if no_translate:
        system_prompt = SYSTEM_PROMPT_OCR_ONLY_TEMPLATE.format(language=lang)
    else:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language=lang)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty content")
            return content.strip()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(
                    f"  Retry {attempt + 1}/{max_retries} for doc after {wait}s: {e}",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)

    print(f"  Error after retries, keeping original: {last_error}", file=sys.stderr)
    return text


async def process_file(
    input_path: Path,
    output_path: Path,
    model: str,
    api_base: str,
    api_key: str | None,
    max_concurrent: int,
    max_retries: int,
    limit: int | None,
    diff_csv: Path | None,
    update_language: bool,
    no_translate: bool,
):
    """Process the input JSONL and write the transformed output JSONL."""
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if diff_csv is not None:
        diff_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading documents from {input_path} ...", file=sys.stderr)
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

    total_docs = len(records)
    print(f"Total documents to process: {total_docs}", file=sys.stderr)
    if no_translate:
        print("Mode: OCR fix only (same language, no translation)", file=sys.stderr)
    else:
        print("Mode: OCR fix + translate to English", file=sys.stderr)
    if update_language and no_translate:
        print(
            "Note: --update_language ignored when --no_translate is set.",
            file=sys.stderr,
        )
    print(f"Using max {max_concurrent} concurrent API calls", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)
    print(f"API base: {api_base}", file=sys.stderr)

    client = AsyncOpenAI(base_url=api_base, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    diff_rows: list[dict] = []

    async def _process_doc(doc: dict, idx: int) -> tuple[int, dict]:
        """Process one document and return (original_index, doc)."""
        doc_id = doc.get("document_id", idx)
        original_text = doc.get("text", "")
        language = doc.get("language", "")

        if not original_text or not original_text.strip():
            if diff_csv is not None:
                diff_rows.append(
                    {
                        "document_id": doc_id,
                        "status": "empty",
                        "orig_len": 0,
                        "proc_len": 0,
                        "original_text": "",
                        "processed_text": "",
                    }
                )
            return idx, doc

        async with semaphore:
            processed_text = await fix_document_text(
                client=client,
                model=model,
                text=original_text,
                language=language,
                max_retries=max_retries,
                no_translate=no_translate,
            )

        status = "fixed" if processed_text != original_text else "unchanged"
        if processed_text == original_text and max_retries > 0:
            status = "failed_fallback"

        doc["text"] = processed_text
        if update_language and not no_translate:
            doc["language"] = "en"

        if diff_csv is not None:
            diff_rows.append(
                {
                    "document_id": doc_id,
                    "status": status,
                    "orig_len": len(original_text),
                    "proc_len": len(processed_text),
                    "original_text": original_text,
                    "processed_text": processed_text,
                }
            )

        return idx, doc

    # Launch all tasks concurrently
    tasks = [_process_doc(doc, i) for i, doc in enumerate(records)]
    processed_map: dict[int, dict] = {}

    start_time = time.time()
    for coro in asyncio.as_completed(tasks):
        idx, result_doc = await coro
        processed_map[idx] = result_doc
        done = len(processed_map)
        if done % 10 == 0 or done == total_docs:
            elapsed = time.time() - start_time
            avg = elapsed / done if done else 0
            print(
                f"Processed {done}/{total_docs} documents ... "
                f"({elapsed:.1f}s elapsed, {avg:.2f}s/doc)",
                file=sys.stderr,
            )

    # Preserve original order
    ordered_records = [processed_map[i] for i in range(len(records))]

    print(f"Writing output to {output_path} ...", file=sys.stderr)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in ordered_records:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    if diff_csv is not None:
        print(f"Writing diff CSV to {diff_csv} ...", file=sys.stderr)
        with open(diff_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "document_id",
                    "status",
                    "orig_len",
                    "proc_len",
                    "original_text",
                    "processed_text",
                ],
            )
            writer.writeheader()
            writer.writerows(diff_rows)

    total_elapsed = time.time() - start_time
    print(
        f"Done. {total_docs} documents processed in {total_elapsed:.1f}s. "
        f"Output saved to: {output_path.resolve()}",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fix OCR on HIPE-2026 text (optional English translation) using an OpenAI-compatible LLM"
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
        help="Path to the output JSONL file with fixed/translated text.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model identifier (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"API base URL (default: {DEFAULT_API_BASE}).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPEN_ROUTER_API"),
        help="API key (defaults to OPEN_ROUTER_API env var).",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f"Maximum concurrent API calls (default: {DEFAULT_MAX_CONCURRENT}).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Retries per document on API failure (default: {DEFAULT_MAX_RETRIES}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to the first N documents (for testing).",
    )
    parser.add_argument(
        "--diff_csv",
        type=Path,
        default=None,
        help="Optional path to a CSV file recording original vs processed text.",
    )
    parser.add_argument(
        "--update_language",
        action="store_true",
        help="If set, set the 'language' field to 'en' after translation (ignored with --no_translate).",
    )
    parser.add_argument(
        "--no_translate",
        action="store_true",
        help="Only fix OCR; keep the text in the same language (no English translation).",
    )
    args = parser.parse_args()

    asyncio.run(
        process_file(
            input_path=args.input_path,
            output_path=args.output_path,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            max_concurrent=args.max_concurrent,
            max_retries=args.max_retries,
            limit=args.limit,
            diff_csv=args.diff_csv,
            update_language=args.update_language,
            no_translate=args.no_translate,
        )
    )


if __name__ == "__main__":
    main()
