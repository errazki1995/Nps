# Ultimate Hybrid Support Review Classifier
#
# Purpose:
# - Classify support review comments with maximum practical accuracy and speed.
# - Uses high-precision regex rules first.
# - Uses local Qwen LLM only when rules do not confidently classify.
# - Applies post-LLM safety rules to prevent obvious wrong outputs.
# - Uses caching for duplicate comments.
# - Saves checkpoints every 5 minutes.
#
# Install:
#   pip install pandas openpyxl torch transformers accelerate
#
# Test first:
#   python support_review_classifier_ultimate.py --input reviews.xlsx --comment-column Comments --output test.xlsx --limit 30
#
# Full run:
#   python support_review_classifier_ultimate.py --input reviews.xlsx --comment-column Comments --output reviews_classified.xlsx
#
# Faster model:
#   python support_review_classifier_ultimate.py --model Qwen/Qwen2.5-1.5B-Instruct

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MAX_NEW_TOKENS = 45
DEFAULT_SAVE_EVERY_MINUTES = 5

CATEGORIES = [
    "Unresolved Issue",
    "Resolved",
    "Slow Response",
    "Complex Process",
    "Lack of Communication",
    "Lack of Ownership",
    "Knowledge Gap",
    "Other",
    "Unknown",
]

SENTIMENTS = ["positive", "neutral", "negative", "mixed"]

OUTPUT_COLUMNS = [
    "primary_theme",
    "sentiment",
    "decision_source",
    "matched_rule",
    "llm_theme",
    "confidence",
    "needs_review",
]

PRIORITY = {
    "Unresolved Issue": 100,
    "Resolved": 90,
    "Lack of Ownership": 80,
    "Complex Process": 70,
    "Lack of Communication": 60,
    "Slow Response": 50,
    "Knowledge Gap": 40,
    "Other": 10,
    "Unknown": 0,
}


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[’‘]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text


def cache_key(value: Any) -> str:
    return hashlib.md5(normalize_text(value).encode("utf-8")).hexdigest()


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


# =========================
# HIGH-PRECISION REGEX RULES
# =========================
# Principle:
# - Rules must be precise enough to avoid false positives.
# - Vague words alone are avoided.
# - Unresolved beats Resolved.
# - Resolved beats service-experience issues only when resolution is explicit.


UNRESOLVED_PATTERNS = [
    r"\bnot\s+(fixed|resolved|solved|sorted|working|completed)\b",
    r"\bwasn'?t\s+(fixed|resolved|solved|sorted|completed)\b",
    r"\bwere\s+not\s+(able\s+to\s+)?(fix|resolve|solve|complete)\b",
    r"\bunable\s+to\s+(fix|resolve|solve|complete)\b",
    r"\bcouldn'?t\s+(fix|resolve|solve|complete)\b",
    r"\bcould\s+not\s+(fix|resolve|solve|complete)\b",
    r"\bfailed\s+to\s+(fix|resolve|solve|complete)\b",
    r"\bdidn'?t\s+(fix|resolve|solve|complete)\b",
    r"\bdid\s+not\s+(fix|resolve|solve|complete)\b",
    r"\bdoesn'?t\s+(fix|resolve|solve|work)\b",
    r"\bdoes\s+not\s+(fix|resolve|solve|work)\b",
    r"\bstill\s+(not\s+)?(unresolved|broken|not working|failing|an issue|a problem|open)\b",
    r"\b(issue|problem|case)\s+(still\s+)?(remains|persists|continues|exists|ongoing)\b",
    r"\bno\s+(fix|solution|resolution|answer)\b",
    r"\bunresolved\b",
    r"\bongoing\s+(issue|problem|case)\b",
    r"\bhad\s+to\s+reopen\b",
    r"\breopened\b",
    r"\bre-opened\b",
    r"\bticket\s+(was\s+)?closed.*(without|before|still|not)",
    r"\bcase\s+(was\s+)?closed.*(without|before|still|not)",
    r"\bclosed\s+(without|before).*(fix|resolv|solv|complete)",
    r"\bworkaround\s+only\b",
    r"\btemporary\s+(fix|solution)\b",
    r"\bissue\s+came\s+back\b",
    r"\bproblem\s+came\s+back\b",
    r"\b(issue|problem)\s+returned\b",
    r"\bsame\s+(issue|problem)\s+(again|returned)\b",
]


RESOLVED_PATTERNS = [
    r"\b(issue|problem|case|request)\s+(was\s+)?(fully\s+)?(resolved|fixed|solved|sorted|completed)\b",
    r"\b(resolved|fixed|solved|sorted|completed)\s+(my|the)\s+(issue|problem|case|request)\b",
    r"\bfully\s+(resolved|fixed|solved|completed)\b",
    r"\bquick\s+resolution\b",
    r"\bquickly\s+(resolved|fixed|solved|sorted)\b",
    r"\b(resolved|fixed|solved|sorted)\s+quickly\b",
    r"\bgot\s+(it|this|the issue|the problem)\s+(resolved|fixed|solved|sorted)\b",
    r"\ball\s+(resolved|fixed|sorted)\b",
    r"\bworks?\s+now\b",
    r"\bworking\s+now\b",
]


OWNERSHIP_PATTERNS = [
    r"\btransferred\s+(me\s+)?(between|to)\b",
    r"\bkept\s+transferring\b",
    r"\bbounced\s+(around|between)\b",
    r"\bpassed\s+(around|between)\b",
    r"\bpassed\s+from\s+one\s+team\s+to\s+another\b",
    r"\bbetween\s+(different\s+)?teams\b",
    r"\bmultiple\s+teams\b",
    r"\bno\s+ownership\b",
    r"\bnobody\s+owned\b",
    r"\bno\s+one\s+owned\b",
    r"\bunclear\s+owner(ship)?\b",
    r"\bwho\s+owns\s+(the\s+)?(case|ticket|issue)\b",
    r"\bhand\s?off(s)?\b",
    r"\bhanded\s+off\b",
]


PROCESS_PATTERNS = [
    r"\btoo\s+many\s+(steps|approvals|forms|processes|stages)\b",
    r"\bprocess\s+(was\s+)?(too\s+)?(complicated|complex|cumbersome|painful|difficult|hard|long-winded)\b",
    r"\bcomplicated\s+process\b",
    r"\bcomplex\s+process\b",
    r"\bcumbersome\s+process\b",
    r"\brepeat(ed)?\s+(my\s+)?(information|details)\b",
    r"\bsame\s+(information|details)\s+(again|multiple times|several times)\b",
    r"\basked\s+for\s+the\s+same\s+(information|details)\b",
    r"\bduplicate\s+(information|details|request)\b",
    r"\btoo\s+much\s+paperwork\b",
    r"\bapproval\s+(delay|delays|process)\b",
]


COMMUNICATION_PATTERNS = [
    r"\bno\s+updates?\b",
    r"\bwithout\s+updates?\b",
    r"\bnobody\s+updated\b",
    r"\bno\s+one\s+updated\b",
    r"\bnot\s+kept\s+informed\b",
    r"\bwasn'?t\s+kept\s+informed\b",
    r"\bpoor\s+communication\b",
    r"\black\s+of\s+communication\b",
    r"\bno\s+communication\b",
    r"\bno\s+explanation\b",
    r"\bunclear\s+(explanation|communication|instructions|guidance|response)\b",
    r"\bconfusing\s+(explanation|communication|instructions|guidance|response)\b",
    r"\bpoor\s+follow[-\s]?up\b",
    r"\bno\s+follow[-\s]?up\b",
    r"\bkept\s+me\s+in\s+the\s+dark\b",
]


RESPONSE_PATTERNS = [
    r"\bslow\s+response\b",
    r"\bresponse\s+(was\s+)?(very\s+)?slow\b",
    r"\bslow\s+to\s+(reply|respond|get back)\b",
    r"\bno\s+response\b",
    r"\bno\s+reply\b",
    r"\btook\s+too\s+long\s+to\s+(reply|respond|get back)\b",
    r"\blong\s+wait\s+(for\s+)?(response|reply)\b",
    r"\bwaited\s+(too\s+)?long\s+for\s+(a\s+)?(response|reply)\b",
    r"\bdelayed\s+(response|reply)\b",
    r"\bresponse\s+took\s+(too\s+)?long\b",
    r"\bwaited\s+(days|weeks|months)\s+for\s+(a\s+)?(response|reply)\b",
]


KNOWLEDGE_PATTERNS = [
    r"\bdidn'?t\s+know\s+(how|what)\b",
    r"\bdid\s+not\s+know\s+(how|what)\b",
    r"\bwrong\s+advice\b",
    r"\bincorrect\s+advice\b",
    r"\binaccurate\s+advice\b",
    r"\black\s+of\s+knowledge\b",
    r"\bnot\s+knowledgeable\b",
    r"\binexperienced\b",
    r"\bpoor\s+expertise\b",
    r"\b(agent|engineer|support)\s+(didn'?t|did not)\s+understand\b",
]


RULE_SETS = [
    ("Unresolved Issue", UNRESOLVED_PATTERNS, 0.99),
    ("Resolved", RESOLVED_PATTERNS, 0.96),
    ("Lack of Ownership", OWNERSHIP_PATTERNS, 0.94),
    ("Complex Process", PROCESS_PATTERNS, 0.93),
    ("Lack of Communication", COMMUNICATION_PATTERNS, 0.92),
    ("Slow Response", RESPONSE_PATTERNS, 0.91),
    ("Knowledge Gap", KNOWLEDGE_PATTERNS, 0.90),
]


POSITIVE_TERMS = [
    "good", "great", "excellent", "helpful", "professional", "quick", "fast",
    "smooth", "easy", "impressed", "thank", "thanks", "perfect", "amazing",
    "brilliant", "satisfied", "happy"
]

NEGATIVE_TERMS = [
    "not", "no", "slow", "delay", "delayed", "poor", "bad", "unable",
    "couldn't", "could not", "failed", "frustrating", "difficult",
    "confusing", "complicated", "unresolved", "still", "closed",
    "wrong", "incorrect", "painful"
]


def infer_sentiment(text: str, category: str) -> str:
    has_positive = any(term in text for term in POSITIVE_TERMS)
    has_negative = any(term in text for term in NEGATIVE_TERMS)

    if category == "Unresolved Issue":
        return "mixed" if has_positive else "negative"

    if category == "Resolved":
        return "mixed" if has_negative else "positive"

    if has_positive and has_negative:
        return "mixed"
    if has_negative:
        return "negative"
    if has_positive:
        return "positive"
    return "neutral"


def apply_high_precision_rules(comment: Any) -> Optional[Dict[str, Any]]:
    text = normalize_text(comment)

    if not text:
        return {
            "primary_theme": "Unknown",
            "sentiment": "neutral",
            "decision_source": "empty",
            "matched_rule": "",
            "llm_theme": "",
            "confidence": 0.0,
            "needs_review": True,
        }

    matches = []

    for category, patterns, confidence in RULE_SETS:
        for pattern in patterns:
            if has(pattern, text):
                matches.append({
                    "category": category,
                    "pattern": pattern,
                    "confidence": confidence,
                    "priority": PRIORITY[category],
                })

    if not matches:
        return None

    matches.sort(key=lambda x: (x["priority"], x["confidence"]), reverse=True)
    best = matches[0]
    category = best["category"]

    return {
        "primary_theme": category,
        "sentiment": infer_sentiment(text, category),
        "decision_source": "rule_high_precision",
        "matched_rule": best["pattern"],
        "llm_theme": "",
        "confidence": best["confidence"],
        "needs_review": False,
    }


# =========================
# LLM FALLBACK
# =========================

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
    return f"""
Classify this support comment into ONE primary_theme.

Allowed labels:
Unresolved Issue
Resolved
Slow Response
Complex Process
Lack of Communication
Lack of Ownership
Knowledge Gap
Other
Unknown

Priority:
1. Unresolved Issue
2. Resolved
3. Lack of Ownership
4. Complex Process
5. Lack of Communication
6. Slow Response
7. Knowledge Gap
8. Other
9. Unknown

Definitions:
- Unresolved Issue: issue not fixed, still exists, no resolution, reopened, ticket closed too early.
- Resolved: issue fixed, solved, resolved, completed successfully.
- Slow Response: slow reply, no response, delayed response, long wait for reply.
- Complex Process: too many steps, approvals, repeated information, process complicated.
- Lack of Communication: no updates, unclear explanation, confusing instructions, poor follow-up.
- Lack of Ownership: bounced between teams, transfers, no clear owner.
- Knowledge Gap: wrong advice, agent lacked knowledge or expertise.
- Other: general positive/negative comment that does not fit above.
- Unknown: impossible to classify.

Rules:
- Return JSON only.
- Do not invent labels.
- If helpful/quick BUT not fixed => Unresolved Issue.
- If fixed/resolved BUT slow => Resolved.
- If unclear but meaningful => Other.
- If impossible to classify => Unknown.

JSON:
{{"primary_theme":"","sentiment":"","confidence":0.0}}

Examples:
Comment: "Helpful agent but issue still not fixed."
JSON: {{"primary_theme":"Unresolved Issue","sentiment":"mixed","confidence":0.96}}

Comment: "The issue was resolved but it took a long time."
JSON: {{"primary_theme":"Resolved","sentiment":"mixed","confidence":0.90}}

Comment: "Nobody updated me."
JSON: {{"primary_theme":"Lack of Communication","sentiment":"negative","confidence":0.90}}

Comment: "I was transferred between teams."
JSON: {{"primary_theme":"Lack of Ownership","sentiment":"negative","confidence":0.90}}

Comment: "The process was too complicated."
JSON: {{"primary_theme":"Complex Process","sentiment":"negative","confidence":0.90}}

Comment: "{comment}"
JSON:
"""


def extract_json(text: str) -> Dict[str, Any]:
    matches = re.findall(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", text, re.DOTALL)
    if not matches:
        raise ValueError("No JSON found")
    return json.loads(matches[-1])


def call_llm(tokenizer, model, comment: str, max_new_tokens: int) -> Dict[str, Any]:
    prompt = build_prompt(comment)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
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

    data = extract_json(decoded)

    theme = data.get("primary_theme", "Unknown")
    sentiment = data.get("sentiment", "neutral")
    confidence = data.get("confidence", 0.0)

    if theme not in CATEGORIES:
        theme = "Unknown"

    if sentiment not in SENTIMENTS:
        sentiment = "neutral"

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    return {
        "primary_theme": theme,
        "sentiment": sentiment,
        "decision_source": "llm",
        "matched_rule": "",
        "llm_theme": theme,
        "confidence": confidence,
        "needs_review": confidence < 0.70 or theme == "Unknown",
    }


def classify(tokenizer, model, comment: Any, max_new_tokens: int, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    text = normalize_text(comment)
    key = cache_key(text)

    if key in cache:
        output = cache[key].copy()
        output["decision_source"] = output["decision_source"] + "_cached"
        return output

    rule_output = apply_high_precision_rules(text)

    if rule_output is not None:
        cache[key] = rule_output.copy()
        return rule_output

    try:
        llm_output = call_llm(tokenizer, model, text, max_new_tokens)
    except Exception:
        llm_output = {
            "primary_theme": "Unknown",
            "sentiment": "neutral",
            "decision_source": "llm_failed",
            "matched_rule": "",
            "llm_theme": "",
            "confidence": 0.0,
            "needs_review": True,
        }

    final_rule = apply_high_precision_rules(text)
    if final_rule is not None and final_rule["primary_theme"] != "Unknown":
        final_rule["decision_source"] = "post_rule_override"
        final_rule["llm_theme"] = llm_output.get("primary_theme", "")
        cache[key] = final_rule.copy()
        return final_rule

    cache[key] = llm_output.copy()
    return llm_output


# =========================
# EXCEL PROCESSING
# =========================

def row_done(row: pd.Series) -> bool:
    value = row.get("primary_theme")
    return isinstance(value, str) and value.strip() != "" and value.strip().lower() != "nan"


def load_or_create_df(input_file: str, output_file: str, comment_column: str, limit: Optional[int]) -> pd.DataFrame:
    if Path(output_file).exists():
        print(f"Resuming from existing output: {output_file}")
        df = pd.read_excel(output_file)
    else:
        df = pd.read_excel(input_file)

        if limit is not None:
            df = df.head(limit).copy()

        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None

    if comment_column not in df.columns:
        raise ValueError(f"Column '{comment_column}' not found. Available columns: {list(df.columns)}")

    return df


def save_output(df: pd.DataFrame, output_file: str, reason: str):
    df.to_excel(output_file, index=False)
    print(f"Saved: {output_file} ({reason})")


def process_excel(input_file: str, output_file: str, comment_column: str, model_name: str,
                  max_new_tokens: int, save_every_minutes: int, limit: Optional[int]):
    if not Path(input_file).exists() and not Path(output_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = load_or_create_df(input_file, output_file, comment_column, limit)

    tokenizer, model = load_model(model_name)

    pending = [idx for idx, row in df.iterrows() if not row_done(row)]
    cache: Dict[str, Dict[str, Any]] = {}

    print(f"Rows total: {len(df)}")
    print(f"Rows pending: {len(pending)}")
    print(f"Max new tokens: {max_new_tokens}")

    start = time.time()
    last_save = time.time()
    save_interval = save_every_minutes * 60

    for n, idx in enumerate(pending, start=1):
        row_start = time.time()

        output = classify(
            tokenizer=tokenizer,
            model=model,
            comment=df.at[idx, comment_column],
            max_new_tokens=max_new_tokens,
            cache=cache,
        )

        for col in OUTPUT_COLUMNS:
            df.at[idx, col] = output[col]

        row_time = time.time() - row_start
        elapsed = time.time() - start
        avg = elapsed / n
        remaining = len(pending) - n
        eta_min = (avg * remaining) / 60 if remaining else 0

        print(
            f"[{n}/{len(pending)}] row={idx + 1} "
            f"theme={output['primary_theme']} "
            f"source={output['decision_source']} "
            f"conf={output['confidence']} "
            f"time={row_time:.1f}s ETA={eta_min:.1f}m"
        )

        if time.time() - last_save >= save_interval:
            save_output(df, output_file, "checkpoint")
            last_save = time.time()

    save_output(df, output_file, "final")

    print("\nPrimary theme distribution:")
    print(df["primary_theme"].value_counts(dropna=False))

    print("\nDecision source distribution:")
    print(df["decision_source"].value_counts(dropna=False))

    print("\nNeeds review:")
    print(df["needs_review"].value_counts(dropna=False))


def main():
    parser = argparse.ArgumentParser(description="Ultimate hybrid support review classifier.")
    parser.add_argument("--input", default="reviews.xlsx")
    parser.add_argument("--output", default="reviews_classified.xlsx")
    parser.add_argument("--comment-column", default="Comments")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--save-every-minutes", type=int, default=DEFAULT_SAVE_EVERY_MINUTES)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    process_excel(
        input_file=args.input,
        output_file=args.output,
        comment_column=args.comment_column,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        save_every_minutes=args.save_every_minutes,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
