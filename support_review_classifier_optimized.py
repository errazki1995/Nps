# Optimized Local Support Review Classifier
#
# Install:
#   pip install pandas openpyxl torch transformers accelerate
#
# Example:
#   python support_review_classifier_optimized.py --input reviews.xlsx --comment-column Comment --output reviews_classified.xlsx
#
# Test 20 rows:
#   python support_review_classifier_optimized.py --input reviews.xlsx --comment-column Comment --output test.xlsx --limit 20
#
# Notes:
# - Outputs only essential columns for speed.
# - Saves checkpoint/output every 5 minutes by default.
# - Can resume from existing output file.

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_SAVE_EVERY_MINUTES = 5
DEFAULT_MAX_NEW_TOKENS = 80

THEMES = [
    "Slow Response",
    "Long Resolution Time",
    "Complex Process",
    "Lack of Communication",
    "Lack of Ownership",
    "Multiple Follow-ups Required",
    "Repeated Information Requests",
    "Unresolved Issue",
    "Poor Quality Resolution",
    "Knowledge Gap",
    "Escalation Required",
    "Ticket Closed Too Early",
    "Poor User Experience",
    "Positive Experience",
    "Other",
]

VALID_SENTIMENTS = ["positive", "neutral", "negative", "mixed"]

OUTPUT_COLUMNS = [
    "themes",
    "primary_theme",
    "sentiment",
    "confidence",
    "needs_review",
]


def load_model(model_name: str):
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.eval()
    return tokenizer, model


def build_prompt(comment: str) -> str:
    return f'''
Classify this customer support review.

Allowed themes:
{THEMES}

Rules:
- Use only allowed themes.
- Use one or more themes.
- Choose primary_theme from themes.
- sentiment must be: positive, neutral, negative, or mixed.
- confidence must be between 0 and 1.
- Return JSON only.
- No explanation.
- No markdown.

JSON schema:
{{
  "themes": [],
  "primary_theme": "",
  "sentiment": "",
  "confidence": 0.0
}}

Examples:
Comment: "The response was very slow and nobody updated me."
JSON: {{"themes":["Slow Response","Lack of Communication"],"primary_theme":"Slow Response","sentiment":"negative","confidence":0.92}}

Comment: "The process was too complicated."
JSON: {{"themes":["Complex Process"],"primary_theme":"Complex Process","sentiment":"negative","confidence":0.90}}

Comment: "The engineer was helpful and solved it quickly."
JSON: {{"themes":["Positive Experience"],"primary_theme":"Positive Experience","sentiment":"positive","confidence":0.95}}

Comment: "{comment}"
JSON:
'''


def extract_json(text: str) -> Dict[str, Any]:
    matches = re.findall(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", text, re.DOTALL)
    if not matches:
        raise ValueError("No JSON object found.")
    return json.loads(matches[-1])


def validate_output(data: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    themes = data.get("themes", [])
    primary_theme = data.get("primary_theme", "Other")
    sentiment = data.get("sentiment", "neutral")
    confidence = data.get("confidence", 0.0)

    if not isinstance(themes, list):
        themes = ["Other"]

    clean_themes = []
    for theme in themes:
        if isinstance(theme, str) and theme in THEMES and theme not in clean_themes:
            clean_themes.append(theme)

    if not clean_themes:
        clean_themes = ["Other"]

    if primary_theme not in clean_themes:
        primary_theme = clean_themes[0]

    if sentiment not in VALID_SENTIMENTS:
        sentiment = "neutral"

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    return {
        "themes": ", ".join(clean_themes),
        "primary_theme": primary_theme,
        "sentiment": sentiment,
        "confidence": confidence,
        "needs_review": confidence < threshold,
    }


def failed_result() -> Dict[str, Any]:
    return {
        "themes": "Other",
        "primary_theme": "Other",
        "sentiment": "neutral",
        "confidence": 0.0,
        "needs_review": True,
    }


def classify_comment(
    tokenizer,
    model,
    comment: Any,
    threshold: float,
    max_new_tokens: int,
    max_retries: int = 1,
) -> Dict[str, Any]:
    comment = "" if pd.isna(comment) else str(comment).strip()

    if not comment:
        return failed_result()

    prompt = build_prompt(comment)

    for attempt in range(max_retries + 1):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):]

            parsed = extract_json(decoded)
            return validate_output(parsed, threshold)

        except Exception:
            if attempt >= max_retries:
                return failed_result()
            time.sleep(0.5)


def row_already_done(row: pd.Series) -> bool:
    value = row.get("primary_theme")
    return isinstance(value, str) and value.strip() != "" and value.strip().lower() != "nan"


def load_or_create_output_df(
    input_file: str,
    output_file: str,
    comment_column: str,
    limit: Optional[int],
) -> pd.DataFrame:
    output_path = Path(output_file)

    if output_path.exists():
        print(f"Existing output found. Resuming from: {output_file}")
        df = pd.read_excel(output_path)
    else:
        df = pd.read_excel(input_file)

        if limit is not None:
            df = df.head(limit).copy()

        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None

    if comment_column not in df.columns:
        raise ValueError(
            f"Column '{comment_column}' not found. Available columns: {list(df.columns)}"
        )

    return df


def save_output(df: pd.DataFrame, output_file: str, reason: str):
    df.to_excel(output_file, index=False)
    print(f"Saved output: {output_file} ({reason})")


def process_excel(
    input_file: str,
    output_file: str,
    comment_column: str,
    model_name: str,
    threshold: float,
    save_every_minutes: int,
    max_new_tokens: int,
    limit: Optional[int],
):
    if not Path(input_file).exists() and not Path(output_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = load_or_create_output_df(input_file, output_file, comment_column, limit)
    tokenizer, model = load_model(model_name)

    pending_indexes = [idx for idx, row in df.iterrows() if not row_already_done(row)]

    print(f"Total rows: {len(df)}")
    print(f"Pending rows: {len(pending_indexes)}")
    print(f"Saving every {save_every_minutes} minute(s)")
    print(f"max_new_tokens={max_new_tokens}")

    start_time = time.time()
    last_save_time = time.time()
    save_interval_seconds = save_every_minutes * 60
    processed_since_save = 0

    for count, idx in enumerate(pending_indexes, start=1):
        row_start = time.time()

        output = classify_comment(
            tokenizer=tokenizer,
            model=model,
            comment=df.at[idx, comment_column],
            threshold=threshold,
            max_new_tokens=max_new_tokens,
        )

        for col in OUTPUT_COLUMNS:
            df.at[idx, col] = output[col]

        processed_since_save += 1
        row_seconds = time.time() - row_start

        elapsed = time.time() - start_time
        avg_seconds = elapsed / count
        remaining = len(pending_indexes) - count
        eta_minutes = (avg_seconds * remaining) / 60 if remaining else 0

        print(
            f"[{count}/{len(pending_indexes)}] "
            f"row={idx + 1} "
            f"theme={output['primary_theme']} "
            f"time={row_seconds:.1f}s "
            f"ETA={eta_minutes:.1f}min"
        )

        if time.time() - last_save_time >= save_interval_seconds:
            save_output(df, output_file, reason=f"checkpoint after {processed_since_save} rows")
            last_save_time = time.time()
            processed_since_save = 0

    save_output(df, output_file, reason="final")

    print("\nDone.")
    print("\nPrimary theme distribution:")
    print(df["primary_theme"].value_counts(dropna=False))

    print("\nRows needing review:")
    print(int(df["needs_review"].fillna(False).sum()))


def main():
    parser = argparse.ArgumentParser(
        description="Optimized local support review classifier with 5-minute checkpoint saves."
    )

    parser.add_argument("--input", default="reviews.xlsx", help="Input Excel file path.")
    parser.add_argument("--output", default="reviews_classified.xlsx", help="Output Excel file path.")
    parser.add_argument("--comment-column", default="Comment", help="Column containing comments.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--save-every-minutes", type=int, default=DEFAULT_SAVE_EVERY_MINUTES)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows for testing.")

    args = parser.parse_args()

    process_excel(
        input_file=args.input,
        output_file=args.output,
        comment_column=args.comment_column,
        model_name=args.model,
        threshold=args.confidence_threshold,
        save_every_minutes=args.save_every_minutes,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
