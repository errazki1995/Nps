"""
Local Support Review Classifier

Purpose:
- Reads an Excel file
- Takes one comment column
- Uses a local Hugging Face Qwen model
- Classifies each comment into support pain-point themes
- Outputs a new Excel file with extra classification columns

Install dependencies:
    pip install pandas openpyxl torch transformers accelerate

Example usage:
    python support_review_classifier.py --input reviews.xlsx --comment-column Comment --output reviews_classified.xlsx

Test mode:
    python support_review_classifier.py --test
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_CONFIDENCE_THRESHOLD = 0.75

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


def load_generator(model_name: str):
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )


def build_prompt(comment: str) -> str:
    return f"""
You are a senior support analytics classification agent.

Classify the customer support review comment into high-level support pain-point themes.

Allowed themes only:
{THEMES}

Rules:
- Use one or more themes.
- Do not invent new themes.
- If the comment is clearly positive, use "Positive Experience".
- If the comment is unclear or not related to support experience, use "Other".
- Choose one primary_theme from the selected themes.
- Sentiment must be one of: positive, neutral, negative, mixed.
- Confidence must be a number between 0 and 1.
- Return JSON only.
- Do not return markdown.
- Do not write any explanation outside the JSON.

Examples:

Comment: "The response was very slow and nobody kept me updated."
JSON:
{{
  "themes": ["Slow Response", "Lack of Communication"],
  "primary_theme": "Slow Response",
  "sentiment": "negative",
  "confidence": 0.92,
  "short_summary": "The customer experienced slow response and poor updates."
}}

Comment: "The process was too complicated and required too many approvals."
JSON:
{{
  "themes": ["Complex Process"],
  "primary_theme": "Complex Process",
  "sentiment": "negative",
  "confidence": 0.90,
  "short_summary": "The customer found the support process too complicated."
}}

Comment: "The engineer was helpful and solved the issue quickly."
JSON:
{{
  "themes": ["Positive Experience"],
  "primary_theme": "Positive Experience",
  "sentiment": "positive",
  "confidence": 0.95,
  "short_summary": "The customer had a positive and quick support experience."
}}

Now classify this comment:

Comment: "{comment}"

JSON:
"""


def extract_json(text: str) -> Dict[str, Any]:
    matches = re.findall(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", text, re.DOTALL)

    if not matches:
        raise ValueError(f"No JSON object found in model output: {text[:300]}")

    return json.loads(matches[-1])


def validate_output(data: Dict[str, Any], confidence_threshold: float) -> Dict[str, Any]:
    themes = data.get("themes", [])
    primary_theme = data.get("primary_theme", "Other")
    sentiment = data.get("sentiment", "neutral")
    confidence = data.get("confidence", 0.0)
    short_summary = data.get("short_summary", "")

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
        "themes": clean_themes,
        "primary_theme": primary_theme,
        "sentiment": sentiment,
        "confidence": confidence,
        "needs_review": confidence < confidence_threshold,
        "short_summary": str(short_summary).strip(),
    }


def classify_comment(
    generator,
    comment: str,
    confidence_threshold: float,
    max_retries: int = 2
) -> Dict[str, Any]:
    comment = "" if pd.isna(comment) else str(comment).strip()

    if not comment:
        return {
            "themes": ["Other"],
            "primary_theme": "Other",
            "sentiment": "neutral",
            "confidence": 0.0,
            "needs_review": True,
            "short_summary": "Empty comment.",
        }

    prompt = build_prompt(comment)

    for attempt in range(max_retries + 1):
        try:
            result = generator(
                prompt,
                max_new_tokens=220,
                do_sample=False,
                temperature=0.0,
                return_full_text=False,
            )

            raw_output = result[0]["generated_text"]
            parsed = extract_json(raw_output)
            return validate_output(parsed, confidence_threshold)

        except Exception as exc:
            if attempt >= max_retries:
                return {
                    "themes": ["Other"],
                    "primary_theme": "Other",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "needs_review": True,
                    "short_summary": f"Classification failed: {exc}",
                }

            time.sleep(1)


def process_excel(
    input_file: str,
    output_file: str,
    comment_column: str,
    model_name: str,
    confidence_threshold: float,
    limit: Optional[int] = None,
):
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_excel(input_path)

    if comment_column not in df.columns:
        raise ValueError(
            f"Column '{comment_column}' not found. Available columns: {list(df.columns)}"
        )

    if limit is not None:
        df = df.head(limit).copy()

    generator = load_generator(model_name)

    results: List[Dict[str, Any]] = []
    total = len(df)

    print(f"Processing {total} rows...")
    print(f"Comment column: {comment_column}")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        comment = row[comment_column]
        print(f"[{i}/{total}] Classifying...")

        output = classify_comment(
            generator=generator,
            comment=comment,
            confidence_threshold=confidence_threshold,
        )

        results.append({
            "themes": ", ".join(output["themes"]),
            "primary_theme": output["primary_theme"],
            "sentiment": output["sentiment"],
            "confidence": output["confidence"],
            "needs_review": output["needs_review"],
            "short_summary": output["short_summary"],
        })

    output_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    output_path = Path(output_file)
    output_df.to_excel(output_path, index=False)

    print("\nDone.")
    print(f"Output saved to: {output_path.resolve()}")

    print("\nPrimary theme distribution:")
    print(output_df["primary_theme"].value_counts(dropna=False))

    print("\nRows needing review:")
    print(int(output_df["needs_review"].sum()))


def run_test(model_name: str, confidence_threshold: float):
    generator = load_generator(model_name)

    test_comments = [
        "The response was slow and nobody gave me updates.",
        "The process was too complicated and required too many approvals.",
        "The issue is still not fixed after several calls.",
        "I had to repeat the same information to three different people.",
        "The engineer was very helpful and solved everything quickly.",
        "My ticket was closed even though the issue still exists.",
        "Support kept transferring me between teams.",
        "The solution worked for one day and then the problem came back.",
    ]

    for comment in test_comments:
        print("\nCOMMENT:")
        print(comment)

        output = classify_comment(
            generator=generator,
            comment=comment,
            confidence_threshold=confidence_threshold,
        )

        print("OUTPUT:")
        print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Classify support review comments from Excel using a local Qwen model."
    )

    parser.add_argument("--input", default="reviews.xlsx", help="Input Excel file path.")
    parser.add_argument("--output", default="reviews_classified.xlsx", help="Output Excel file path.")
    parser.add_argument("--comment-column", default="Comment", help="Name of the Excel column containing comments.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Local Hugging Face model name.")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for testing.")
    parser.add_argument("--test", action="store_true", help="Run built-in test comments instead of Excel.")

    args = parser.parse_args()

    if args.test:
        run_test(
            model_name=args.model,
            confidence_threshold=args.confidence_threshold,
        )
    else:
        process_excel(
            input_file=args.input,
            output_file=args.output,
            comment_column=args.comment_column,
            model_name=args.model,
            confidence_threshold=args.confidence_threshold,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
