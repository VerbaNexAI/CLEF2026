# Example: French dataset
python translate/baseline/create_ocr_fix_dataset.py   --input_path data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-en.jsonl   --output_path sft\dataset\ocr_fix_to_en.jsonl  --model deepseek/deepseek-v3.2 --max_concurrent 10

# Example: German dataset
python translate/baseline/create_ocr_fix_dataset.py \
  --input_path data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-de.jsonl \
  --output_path data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-de-en.jsonl \
  --update_language

# Example: English dataset (OCR fix only, translation naturally skipped by prompt)
python translate/baseline/create_ocr_fix_dataset.py \
  --input_path data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-en.jsonl \
  --output_path data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-en-fix.jsonl \
  --update_language

# Smoke test on first 5 documents
python translate/baseline/create_ocr_fix_dataset.py \
  --input_path data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-de.jsonl \
  --output_path translate/baseline/test_output.jsonl \
  --limit 5
Key CLI flags
Flag	Description
--input_path	Source JSONL
--output_path	Destination JSONL
--model	OpenRouter model (default: openrouter/openai/gpt-oss-120b)
--max_concurrent	Async concurrency limit (default: 5)
--max_retries	Retries per document on API errors (default: 2)
--limit	Process only first N docs for testing
--diff_csv	Optional audit CSV with original vs processed text
--update_language	Overwrite language field to "en"