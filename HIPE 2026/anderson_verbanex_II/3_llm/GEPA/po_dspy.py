"""
Prompt Optimization for HIPE-2026 using DSPY GEPA
This script loads JSONL data with person-location pairs and optimizes prompts
for 'at' and 'isAt' relationship classification.
"""

# %%
import json
import os
from pathlib import Path
from typing import Literal

import dspy
from dotenv import load_dotenv
from dspy import GEPA

#%%
# DEFAULT_MODEL_NAME = "openrouter/stepfun/step-3.5-flash"
DEFAULT_MODEL_NAME = "openai/nemotron-3-nano-4b-oss-20b"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = SCRIPT_DIR / "models"
DISTILL_OUTPUT_DIR = PROJECT_ROOT / "data" / "distill_dataset_po"
DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "newspapers"
    / "v1.0"
    / "HIPE-2026-v1.0-impresso-train-en-ocr-fix-deepseekv3.2.jsonl"
)

#%%
def configure_runtime(lm: dspy.BaseLM) -> None:
    """Configure DSPy and MLflow for a training run."""
    import mlflow

    load_dotenv()
    # mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.set_experiment("debug")
    # mlflow.dspy.autolog(
    #     log_compiles=True,
    #     log_evals=True,
    #     log_traces=True,
    # )
    dspy.configure(lm=lm)


def build_lm(model_name: str = DEFAULT_MODEL_NAME) -> dspy.LM:
    """Create the LM used for GEPA optimization and evaluation."""
    load_dotenv()
    lm= dspy.LM(
        model=model_name,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API"),
        temperature=1,
    )
    lm = dspy.LM(
        model=model_name,
        api_base="http://127.0.0.1:1234/v1",
        api_key="",
        temperature=1,
    )
    return lm


def get_model_short_name(model_name: str) -> str:
    """Extract a filename-safe model name from the full LM identifier."""
    return model_name.split("/")[-1] if "/" in model_name else model_name

#%%
def get_gepa_model_paths(
    model_short_name: str, model_dir: Path | None = None
) -> tuple[Path, Path]:
    """Return the save paths for the optimized at/isAt GEPA programs."""
    target_dir = model_dir or MODEL_DIR
    return (
        target_dir / f"gepa_at_{model_short_name}.json",
        target_dir / f"gepa_isat_{model_short_name}.json",
    )

# %%
# ============================================================================
# 1. LOAD DATA - Read JSONL and extract sampled_pairs with labels
# ============================================================================


def load_data(jsonl_path, limit=None):
    """
    Load JSONL file and extract person-location pairs with 'at' and 'isAt' labels.
    Returns list of dspy.Example objects for training.
    """
    examples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            record = json.loads(line)
            text = record.get("text", "")
            date = record.get("date", "")
            document_id = record.get("document_id", "")

            for pair in record.get("sampled_pairs", []):
                person = ", ".join(pair.get("pers_mentions_list", [])) or "Unknown"
                location = ", ".join(pair.get("loc_mentions_list", [])) or "Unknown"
                at_label = pair.get("at", "")
                isat_label = pair.get("isAt", "")

                # Skip if labels are missing
                if not at_label or not isat_label:
                    continue

                # Create example with both labels as ground truth
                example = dspy.Example(
                    text=text,
                    person=person,
                    location=location,
                    date=date,
                    document_id=document_id,
                    at_label=at_label,  # TRUE/FALSE/PROBABLE
                    isat_label=isat_label,  # TRUE/FALSE
                ).with_inputs("text", "person", "location", "date")

                examples.append(example)

                if limit and len(examples) >= limit:
                    break

            if limit and len(examples) >= limit:
                break

    print(f"Loaded {len(examples)} examples from {jsonl_path}")
    return examples


# %%
# ============================================================================
# 2. DEFINE SIGNATURE - Input/Output fields for classification
# ============================================================================


class AtClassification(dspy.Signature):
    """
    Analyze the text and determine whether there is evidence that the person was at the specified location at any time before the document’s publication date. 
    Use NLP techniques if necessary to interpret the text and identify relevant contextual or temporal clues indicating presence.
    """

    text = dspy.InputField(desc="The document text to analyze")
    person = dspy.InputField(desc="The person mentioned in the text")
    location = dspy.InputField(desc="The location mentioned in the text")
    date = dspy.InputField(desc="The document publication date")

    classification: Literal["TRUE", "FALSE", "PROBABLE"] = dspy.OutputField()
    explanation = dspy.OutputField(desc="Brief explanation for the classification")


class IsAtClassification(dspy.Signature):
    """
    Analyze the text and determine whether the person was present at the specified location within approximately one month prior to the document’s publication date. 
    Use NLP techniques if needed to interpret temporal references, contextual cues, and event descriptions. 
    This task is stricter than general presence detection and should focus specifically on recent presence.
    """

    text = dspy.InputField(desc="The document text to analyze")
    person = dspy.InputField(desc="The person mentioned in the text")
    location = dspy.InputField(desc="The location mentioned in the text")
    date = dspy.InputField(desc="The document publication date")

    classification: Literal["TRUE", "FALSE"] = dspy.OutputField()
    explanation = dspy.OutputField(desc="Brief explanation for the classification")


def build_at_program(
    lm: dspy.BaseLM | None = None, model_path: str | Path | None = None
) -> dspy.ChainOfThought:
    """Build the at GEPA program and optionally load a saved artifact."""
    program = dspy.ChainOfThought(AtClassification)
    if model_path is not None:
        program.load(model_path)
    if lm is not None:
        program.set_lm(lm)
    return program


def build_isat_program(
    lm: dspy.BaseLM | None = None, model_path: str | Path | None = None
) -> dspy.ChainOfThought:
    """Build the isAt GEPA program and optionally load a saved artifact."""
    program = dspy.ChainOfThought(IsAtClassification)
    if model_path is not None:
        program.load(model_path)
    if lm is not None:
        program.set_lm(lm)
    return program


# %%
# ============================================================================
# 3. METRIC FUNCTIONS - GEPA-compliant metrics with rich feedback
# ============================================================================


def at_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    GEPA metric for 'at' classification.

    Args:
        gold: The gold example with at_label
        pred: The predicted output with classification
        trace: The trace of the program's execution
        pred_name: Name of the predictor being optimized (provided by GEPA)
        pred_trace: Sub-trace for the predictor (provided by GEPA)

    Returns:
        dspy.Prediction with score and feedback for GEPA
    """
    correct = gold.at_label.upper()
    predicted = pred.classification.upper().strip() if pred.classification else ""

    # Normalize prediction
    if predicted not in ["TRUE", "FALSE", "PROBABLE"]:
        pred_upper = predicted.upper()
        if "TRUE" in pred_upper and "FALSE" not in pred_upper:
            predicted = "TRUE"
        elif "PROBABLE" in pred_upper:
            predicted = "PROBABLE"
        elif "FALSE" in pred_upper:
            predicted = "FALSE"

    score = 1.0 if predicted == correct else 0.0

    # Build rich feedback for GEPA reflection
    if score == 1.0:
        feedback = f"Correct! The answer is '{correct}'."
        if hasattr(pred, "explanation") and pred.explanation:
            feedback += f" Your explanation: {pred.explanation}"
    else:
        feedback = f"Incorrect. You predicted '{predicted}' but the correct answer is '{correct}'."
        feedback += (
            f"\n\nText context: Person '{gold.person}' and Location '{gold.location}'."
        )
        feedback += f"\nDocument date: {gold.date}"
        feedback += (
            f"\n\nThink about what evidence in the text indicates the person was"
        )
        feedback += f" at the location. Consider:"
        feedback += f"\n- Direct mentions of the person being at the location"
        feedback += f"\n- Temporal clues (when did this happen relative to publication)"
        feedback += f"\n- Use 'TRUE' for explicit evidence, 'PROBABLE' for strong inference, 'FALSE' for no evidence"

    return dspy.Prediction(score=score, feedback=feedback)


def isat_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    GEPA metric for 'isAt' classification (recent presence within ~1 month).

    Args:
        gold: The gold example with isat_label
        pred: The predicted output with classification
        trace: The trace of the program's execution
        pred_name: Name of the predictor being optimized (provided by GEPA)
        pred_trace: Sub-trace for the predictor (provided by GEPA)

    Returns:
        dspy.Prediction with score and feedback for GEPA
    """
    correct = gold.isat_label.upper()
    predicted = pred.classification.upper().strip() if pred.classification else ""

    # Normalize to TRUE/FALSE
    pred_upper = predicted.upper()
    if "TRUE" in pred_upper:
        predicted = "TRUE"
    elif "FALSE" in pred_upper:
        predicted = "FALSE"

    score = 1.0 if predicted == correct else 0.0

    # Build rich feedback for GEPA reflection
    if score == 1.0:
        feedback = f"Correct! The answer is '{correct}'."
        if hasattr(pred, "explanation") and pred.explanation:
            feedback += f" Your explanation: {pred.explanation}"
    else:
        feedback = f"Incorrect. You predicted '{predicted}' but the correct answer is '{correct}'."
        feedback += (
            f"\n\nText context: Person '{gold.person}' and Location '{gold.location}'."
        )
        feedback += f"\nDocument date: {gold.date}"
        feedback += f"\n\nFocus on evidence of RECENT presence (within ~1 month before publication)."
        feedback += f" Look for:"
        feedback += f"\n- Specific dates or time references"
        feedback += f"\n- Current events or ongoing activities"
        feedback += f"\n- Verbs indicating present/recent action"
        feedback += f"\nUse 'TRUE' only for clear evidence of recent presence, 'FALSE' otherwise."

    return dspy.Prediction(score=score, feedback=feedback)


# %%
# ============================================================================
# 4. HELPER FUNCTION for evaluation DataFrame conversion
# ============================================================================


def prepare_results_output(results, metric_name):
    """Helper to format evaluation results as dictionaries for DataFrame conversion."""
    data = []
    for example, prediction, score in results:
        if hasattr(example, "toDict"):
            ex_dict = example.toDict()
        else:
            ex_dict = dict(example)

        if hasattr(prediction, "toDict"):
            pred_dict = prediction.toDict()
        else:
            pred_dict = (
                dict(prediction)
                if hasattr(prediction, "items")
                else {"prediction": prediction}
            )

        row = {}
        for key, value in ex_dict.items():
            if key in pred_dict:
                row[f"example_{key}"] = value
            else:
                row[key] = value

        for key, value in pred_dict.items():
            if key in ex_dict:
                row[f"pred_{key}"] = value
            else:
                row[key] = value

        row[metric_name] = score
        data.append(row)

    return data


# %%
# ============================================================================
# 5. MAIN - Load data, optimize prompts, save artifacts, evaluate
# ============================================================================


import pandas as pd

lm = build_lm()
configure_runtime(lm)

print("Loading training data...")
all_examples = load_data(DEFAULT_DATA_PATH)

if len(all_examples) == 0:
    raise SystemExit("No examples loaded. Check the data path.")

split_idx = int(len(all_examples) * 0.8)
train_set = all_examples[:split_idx]
val_set = all_examples[split_idx:]

print(f"Train set: {len(train_set)}, Val set: {len(val_set)}")

print("\n" + "=" * 60)
print("Optimizing 'at' classification prompt...")
print("=" * 60)
at_program = build_at_program(lm)
at_optimizer = GEPA(
    metric=at_metric,
    auto="light",
    num_threads=2,
    reflection_minibatch_size=40,
    skip_perfect_score=True,
    reflection_lm=lm,
    seed=42,
)
optimized_at = at_optimizer.compile(
    at_program,
    trainset=train_set,
    valset=val_set,
)
print("\n--- Optimized 'at' Prompt Instructions ---")
print(optimized_at.predict.signature.instructions)
print("-" * 60)


#%%
print("\n" + "=" * 60)
print("Optimizing 'isAt' classification prompt...")
print("=" * 60)
isat_program = build_isat_program(lm)
isat_optimizer = GEPA(
    metric=isat_metric,
    auto="light",
    num_threads=2,
    reflection_minibatch_size=40,
    skip_perfect_score=True,
    reflection_lm=lm,
    seed=42,
)
optimized_isat = isat_optimizer.compile(
    isat_program,
    trainset=train_set,
    valset=val_set,
)
print("\n--- Optimized 'isAt' Prompt Instructions ---")
print(optimized_isat.predict.signature.instructions)
print("-" * 60)

#%%

DISTILL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_name = lm.model if hasattr(lm, "model") else str(lm)
model_short_name = get_model_short_name(model_name)
at_gepa_path, isat_gepa_path = get_gepa_model_paths(model_short_name)

optimized_at.save(at_gepa_path)
optimized_isat.save(isat_gepa_path)

summary_path = DISTILL_OUTPUT_DIR / f"optimized_prompts_{model_short_name}.json"
summary = {
    "model_name": model_name,
    "at_instructions": optimized_at.predict.signature.instructions,
    "isat_instructions": optimized_isat.predict.signature.instructions,
    "at_gepa_model_path": str(at_gepa_path),
    "isat_gepa_model_path": str(isat_gepa_path),
}
with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, ensure_ascii=False)


#%%
print("\n" + "=" * 60)
print("Saved optimized artifacts")
print("=" * 60)
print(f"Summary saved to: {summary_path}")
print(f"AT GEPA model saved to: {at_gepa_path}")
print(f"isAt GEPA model saved to: {isat_gepa_path}")
print(f"Model used: {model_name}")

print("\n" + "=" * 60)
print("Evaluating Optimized Programs...")
print("=" * 60)

print("\n--- Evaluating 'at' classification ---")
at_evaluate = dspy.Evaluate(
    devset=all_examples,
    metric=at_metric,
    num_threads=4,
    display_table=True,
    display_progress=True,
)
at_result = at_evaluate(optimized_at)
print(f"'at' evaluation score: {at_result.score}%")

at_data = prepare_results_output(at_result.results, at_metric.__name__)
df_at = pd.DataFrame(at_data)
df_at["model_name"] = model_name
at_json_path = DISTILL_OUTPUT_DIR / f"at_evaluation_results_{model_short_name}.json"
df_at.to_json(at_json_path, orient="records", indent=2, force_ascii=False)
print(f"Results saved to: {at_json_path}")

print("\n--- Evaluating 'isAt' classification ---")
isat_evaluate = dspy.Evaluate(
    devset=all_examples,
    metric=isat_metric,
    num_threads=4,
    display_table=True,
    display_progress=True,
)
isat_result = isat_evaluate(optimized_isat)
print(f"'isAt' evaluation score: {isat_result.score}%")

isat_data = prepare_results_output(isat_result.results, isat_metric.__name__)
df_isat = pd.DataFrame(isat_data)
df_isat["model_name"] = model_name
isat_json_path = DISTILL_OUTPUT_DIR / f"isat_evaluation_results_{model_short_name}.json"
df_isat.to_json(isat_json_path, orient="records", indent=2, force_ascii=False)
print(f"Results saved to: {isat_json_path}")

print("\n" + "=" * 60)
print("Evaluation Complete!")
print("=" * 60)



# %%
    
# prediction = optimized_at(
# text="DIXIELAND JAZZ GETS\nNATION’S EAR AGAIN\nNEW YORK.—All over the na\ntion. Dixieland Jazz is hot again,\nis the disclosure of Joseph Roddy\nin the June fl issue of Look Mag\nazine. The great new temples,\nsuch as Birdland and Bop City,\ndedicated to the new jazz, are\nlosing their congregations, he\nfinds.\nAs a native American art form,\n\"Dixieland is a lot of things to a\nlot of people.” it is explained. To\nthe jazz purist, his definition\nstates, \"it is a ritually stylized\nway of playing a hallowed set of\nhollers, work songs, spirituals,\nblues and marches.\" Most of them\nexisted before 1920. Their idio-\nlogical point of origin is New Or\nleans. he says.\nRecord companies are reissuing\nsome of the best work* of old\nDixieland greats, many of them\nNegroes. What caused the revival\nof Dixieland jazz is a mystery.\nRoddy credits disk jockeys in va\nrious cities with helping the re\nvival.",
# person="Napoleon",
# location="Paris",
# date="1815-03-10"
# )
# # %%
# optimized_at.save("module.json")

# %%


# %%
