import json
import re
import sys
import argparse
import os
import asyncio
import time
from typing import Literal
from dotenv import load_dotenv
import mlflow



# Load environment variables from .env file
load_dotenv()

# Initialize DSPy
import dspy


VALID_AT_LABELS = {"TRUE", "FALSE", "PROBABLE"}
VALID_ISAT_LABELS = {"TRUE", "FALSE"}


# ============================================================================
# DSPy Signatures for Classification
# ============================================================================

class AtClassification(dspy.Signature):
    """
    Analyze the text and determine whether there is evidence that the person was at the specified location at any time before the document's publication date.
    Use NLP techniques if necessary to interpret the text and identify relevant contextual or temporal clues indicating presence.
    """

    text = dspy.InputField(desc="The document text to analyze")
    person = dspy.InputField(desc="The person mentioned in the text")
    location = dspy.InputField(desc="The location mentioned in the text")
    date = dspy.InputField(desc="The document publication date")

    classification = dspy.OutputField(desc="One of: TRUE, FALSE, PROBABLE")


class IsAtClassification(dspy.Signature):
    """
    Analyze the text and determine whether the person was present at the specified location within approximately one month prior to the document's publication date.
    Use NLP techniques if needed to interpret temporal references, contextual cues, and event descriptions.
    This task is stricter than general presence detection and should focus specifically on recent presence.
    """

    text = dspy.InputField(desc="The document text to analyze")
    person = dspy.InputField(desc="The person mentioned in the text")
    location = dspy.InputField(desc="The location mentioned in the text")
    date = dspy.InputField(desc="The document publication date")

    classification = dspy.OutputField(desc="One of: TRUE, FALSE")


def _extract_json_from_text(text: str) -> dict | None:
    """Try to find and parse a JSON object from freeform LLM text."""
    json_match = re.search(r'\{[^{}]*"classification"\s*:\s*"[^"]+?"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


def _normalize_at_label(value: str) -> str | None:
    """Map freeform LLM values to valid 'at' labels: TRUE, FALSE, PROBABLE."""
    v = value.strip().upper()
    for label in ("TRUE", "PROBABLE", "FALSE"):
        if label in v:
            return label
    positive = {"YES", "PRESENT", "CONFIRMED", "EVIDENCE"}
    if any(kw in v for kw in positive):
        return "TRUE"
    negative = {"NO", "ABSENT", "NONE", "NOT"}
    if any(kw in v for kw in negative):
        return "FALSE"
    return None


def _normalize_isat_label(value: str) -> str | None:
    """Map freeform LLM values to valid 'isAt' labels: TRUE, FALSE."""
    v = value.strip().upper()
    if "TRUE" in v:
        return "TRUE"
    if "FALSE" in v:
        return "FALSE"
    positive = {"YES", "PRESENT", "CONFIRMED", "RECENT"}
    if any(kw in v for kw in positive):
        return "TRUE"
    negative = {"NO", "ABSENT", "NONE", "NOT"}
    if any(kw in v for kw in negative):
        return "FALSE"
    return None


def _fallback_extract_at(raw_content: str) -> tuple[str, str] | None:
    """Attempt to extract (classification, reasoning) for 'at' from raw LLM text."""
    data = _extract_json_from_text(raw_content)
    if data:
        cls_value = data.get("classification") or data.get("evidence_of_presence") or data.get("relationship") or data.get("was_present")
        reasoning = data.get("reasoning") or data.get("explanation") or data.get("context") or data.get("comment") or ""
        if cls_value:
            label = _normalize_at_label(str(cls_value))
            if label:
                return label, str(reasoning) if reasoning else "Extracted from fallback parsing."
    label = _normalize_at_label(raw_content)
    if label:
        return label, raw_content[:200].strip()
    return None


def _fallback_extract_isat(raw_content: str) -> tuple[str, str] | None:
    """Attempt to extract (classification, reasoning) for 'isAt' from raw LLM text."""
    data = _extract_json_from_text(raw_content)
    if data:
        cls_value = data.get("classification") or data.get("relationship") or data.get("was_present") or data.get("person_present_at_location")
        reasoning = data.get("reasoning") or data.get("explanation") or data.get("context") or data.get("comment") or ""
        if cls_value:
            label = _normalize_isat_label(str(cls_value))
            if label:
                return label, str(reasoning) if reasoning else "Extracted from fallback parsing."
    label = _normalize_isat_label(raw_content)
    if label:
        return label, raw_content[:200].strip()
    return None


def build_at_program(at_model_path: str | None = None) -> dspy.ChainOfThought:
    """Build the at classification program and optionally load a saved model."""
    program = dspy.ChainOfThought(AtClassification)
    if at_model_path is not None and os.path.exists(at_model_path):
        program.load(at_model_path)
        print(f"Loaded optimized At model from: {at_model_path}")
    return program


def build_isat_program(isat_model_path: str | None = None) -> dspy.ChainOfThought:
    """Build the isAt classification program and optionally load a saved model."""
    program = dspy.ChainOfThought(IsAtClassification)
    if isat_model_path is not None and os.path.exists(isat_model_path):
        program.load(isat_model_path)
        print(f"Loaded optimized IsAt model from: {isat_model_path}")
    return program


async def classify_at_relationship(text, person_mentions, location_mentions, date, at_program: dspy.ChainOfThought, semaphore=None, max_retries: int = 1):
    """
    Call DSPy program to classify if the person was at the location.
    Returns a tuple: (label, explanation, prompt, full_response_json)
    """
    person_str = ", ".join(person_mentions) if person_mentions else "Unknown Person"
    location_str = ", ".join(location_mentions) if location_mentions else "Unknown Location"
    date_str = date if date else "Unknown Date"

    async with semaphore:
        last_error = None
        raw_content = None

        for attempt in range(max_retries + 1):
            try:
                # Use acall() for async execution
                result = await at_program.acall(
                    text=text,
                    person=person_str,
                    location=location_str,
                    date=date_str
                )

                label = result.classification.upper().strip() if result.classification else ""
                reasoning = result.reasoning if hasattr(result, "reasoning") else ""

                if label in VALID_AT_LABELS:
                    full_response_json = json.dumps({
                        "classification": label,
                        "reasoning": reasoning
                    }, ensure_ascii=False)
                    return label, reasoning, None, full_response_json

                # Try fallback parsing if label is invalid
                raw_content = str(result)
                fallback = _fallback_extract_at(raw_content)
                if fallback:
                    label, reasoning = fallback
                    full_response_json = json.dumps({
                        "classification": label,
                        "reasoning": reasoning
                    }, ensure_ascii=False)
                    print(f"Fallback recovered 'at' classification: {label}", file=sys.stderr)
                    return label, reasoning, None, full_response_json

                raise ValueError(f"Invalid classification value: '{result.classification}'")

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"Retry {attempt + 1}/{max_retries} for 'at' classification...", file=sys.stderr)
                    await asyncio.sleep(1)

        print(f"Error calling DSPy for 'at' classification (after retries): {last_error}", file=sys.stderr)
        return None, None, None, None


async def classify_isat_relationship(text, date, person_mentions, location_mentions, isat_program: dspy.ChainOfThought, semaphore=None, max_retries: int = 1):
    """
    Call DSPy program to classify if the person was at the location within ~1 month before publication.
    Returns a tuple: (label, explanation, prompt, full_response_json)
    """
    person_str = ", ".join(person_mentions) if person_mentions else "Unknown Person"
    location_str = ", ".join(location_mentions) if location_mentions else "Unknown Location"
    date_str = date if date else "Unknown Date"

    async with semaphore:
        last_error = None
        raw_content = None

        for attempt in range(max_retries + 1):
            try:
                # Use acall() for async execution
                result = await isat_program.acall(
                    text=text,
                    person=person_str,
                    location=location_str,
                    date=date_str
                )

                label = result.classification.upper().strip() if result.classification else ""
                reasoning = result.reasoning if hasattr(result, "reasoning") else ""

                if label in VALID_ISAT_LABELS:
                    full_response_json = json.dumps({
                        "classification": label,
                        "reasoning": reasoning
                    }, ensure_ascii=False)
                    return label, reasoning, None, full_response_json

                # Try fallback parsing if label is invalid
                raw_content = str(result)
                fallback = _fallback_extract_isat(raw_content)
                if fallback:
                    label, reasoning = fallback
                    full_response_json = json.dumps({
                        "classification": label,
                        "reasoning": reasoning
                    }, ensure_ascii=False)
                    print(f"Fallback recovered 'isAt' classification: {label}", file=sys.stderr)
                    return label, reasoning, None, full_response_json

                raise ValueError(f"Invalid classification value: '{result.classification}'")

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"Retry {attempt + 1}/{max_retries} for 'isAt' classification...", file=sys.stderr)
                    await asyncio.sleep(1)

        print(f"Error calling DSPy for 'isAt' classification (after retries): {last_error}", file=sys.stderr)
        return None, None, None, None


async def process_pair(pair, record, at_program: dspy.ChainOfThought, isat_program: dspy.ChainOfThought, semaphore, max_retries: int = 1):
    """Process a single person-location pair and return the updated pair."""
    text = record.get("text", "")
    date = record.get("date", None)

    person_mentions = pair.get("pers_mentions_list", [])
    location_mentions = pair.get("loc_mentions_list", [])

    # Run both classifications concurrently for this pair
    at_task = classify_at_relationship(
        text, person_mentions, location_mentions, date, at_program, semaphore, max_retries=max_retries
    )
    isat_task = classify_isat_relationship(
        text, date, person_mentions, location_mentions, isat_program, semaphore, max_retries=max_retries
    )

    # Wait for both to complete
    at_result = await at_task
    isat_result = await isat_task

    at_value, at_explanation, _, _ = at_result
    isat_value, isat_explanation, _, _ = isat_result

    # Update the pair with predictions
    pair["at"] = at_value
    pair["at_explanation"] = at_explanation
    pair["isAt"] = isat_value
    pair["isAt_explanation"] = isat_explanation

    return pair


async def process_file(input_path, output_path, model="openrouter/openai/gpt-oss-120b",
                       api_base="https://openrouter.ai/api/v1",
                       at_model_path=None, isat_model_path=None,
                       max_concurrent=5, limit=None, max_retries=1,
                       mlflow_experiment=None, max_tokens=None, response_format_type=None):
    """
    Process the input JSONL file and write predictions to output.
    Uses async to make multiple concurrent API calls via DSPy.
    """
    total_pairs = 0
    processed_pairs = 0

    with open(input_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    # Count total pairs first (respecting limit if provided)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            pairs_in_record = len(record.get("sampled_pairs", []))
            if limit is not None:
                remaining = limit - total_pairs
                if remaining <= 0:
                    break
                pairs_in_record = min(pairs_in_record, remaining)
            total_pairs += pairs_in_record
        except json.JSONDecodeError:
            continue

    print(f"Found {total_pairs} person-location pairs to process" + (f" (limited to first {limit})" if limit else ""))
    print(f"Using max {max_concurrent} concurrent API calls")
    print(f"Base model: {model}")

    if mlflow_experiment:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(mlflow_experiment)
        mlflow.dspy.autolog(
            log_compiles=True,
            log_evals=True,
            log_traces=True,
        )

    # Configure DSPy LM
    lm_kwargs = {
        "model": model,
        "api_base": api_base,
        "api_key": os.getenv("OPEN_ROUTER_API"),
        "temperature": 0.1,
    }



    if max_tokens is not None:
        lm_kwargs["max_tokens"] = max_tokens
    if response_format_type is not None:
        lm_kwargs["extra_body"] = {"response_format": {"type": response_format_type}}


    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)

    # Build DSPy programs (load optimized models if paths provided)
    at_program = build_at_program(at_model_path)
    isat_program = build_isat_program(isat_model_path)

    # Override the LM on the programs (following pattern from test.py)
    at_program.predict.lm = lm
    isat_program.predict.lm = lm

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Collect all pairs to process
    all_tasks = []
    records_with_pairs = []

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)
            continue

        pairs = record.get("sampled_pairs", [])
        for pair in pairs:
            if limit is not None and processed_pairs >= limit:
                break

            # Create task for this pair
            task = process_pair(pair, record, at_program, isat_program, semaphore, max_retries=max_retries)
            all_tasks.append((task, record, pair))
            processed_pairs += 1

        records_with_pairs.append((record, pairs))

    print(f"Created {len(all_tasks)} async tasks")
    print("Processing...")

    start_time = time.time()

    # Process all tasks concurrently, but track progress
    completed = 0
    for task, _, _ in all_tasks:
        await task
        completed += 1
        if completed % 10 == 0 or completed == len(all_tasks):
            print(f"Completed {completed}/{len(all_tasks)} pairs ({completed/len(all_tasks)*100:.1f}%)")

    elapsed = time.time() - start_time
    print(f"Completed processing {completed} pairs in {elapsed:.1f} seconds")
    print(f"Average time per pair: {elapsed/completed:.2f} seconds" if completed > 0 else "N/A")

    # Write output with predictions
    with open(output_path, "w", encoding="utf-8") as fout:
        for record, _ in records_with_pairs:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict 'at' and 'isAt' relationships using DSPy with optional optimized models."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to output JSONL file (with predictions). Defaults to baseline/result/<model_name>.jsonl"
    )
    parser.add_argument(
        "--model",
        default="openrouter/openai/gpt-oss-120b",
        help="Base LLM model to use (default: openrouter/openai/gpt-oss-120b)"
    )
    parser.add_argument(
        "--api_base",
        default="https://openrouter.ai/api/v1",
        help="API base URL (default: https://openrouter.ai/api/v1)"
    )
    parser.add_argument(
        "--at_model_path",
        default=None,
        help="Path to optimized At classification model JSON (e.g., prompt_optimation/models/gepa_at_gpt-oss-20b.json)"
    )
    parser.add_argument(
        "--isat_model_path",
        default=None,
        help="Path to optimized IsAt classification model JSON (e.g., prompt_optimation/models/gepa_isat_gpt-oss-20b.json)"
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=2,
        help="Maximum number of concurrent API calls (default: 5)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N person-location pairs (for testing)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="Maximum retries per classification call (default: 1)"
    )
    parser.add_argument(
        "--mlflow_experiment",
        default=None,
        help="Enable MLflow tracking under the given experiment name (default: disabled)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate per call"
    )
    parser.add_argument(
        "--response_format_type",
        default=None,
        choices=["text", "json_object"],
        help="Value for extra_body response_format type (e.g., text)"
    )
    args = parser.parse_args()

    # Determine output path
    output_path = args.output_path
    if output_path is None:
        # Sanitize model name for filename (replace / with _)
        model_filename = args.model.replace("/", "_")
        output_path = os.path.join("baseline", "result", f"{model_filename}.jsonl")

    # Run the async main function
    asyncio.run(process_file(
        args.input_path,
        output_path,
        model=args.model,
        api_base=args.api_base,
        at_model_path=args.at_model_path,
        isat_model_path=args.isat_model_path,
        max_concurrent=args.max_concurrent,
        limit=args.limit,
        max_retries=args.max_retries,
        mlflow_experiment=args.mlflow_experiment,
        max_tokens=args.max_tokens,
        response_format_type=args.response_format_type,
    ))


if __name__ == "__main__":
    main()
