"""
Ultra-stable local support review classifier.

Goal: maximise consistency by using:
1) deterministic rules first
2) strict priority order
3) local Qwen only when rules are unclear
4) audit columns so you can see why a label was chosen

Install:
  pip install pandas openpyxl torch transformers accelerate

Run:
  python support_review_classifier_ultra.py --input reviews.xlsx --comment-column Comments --output reviews_classified.xlsx

Test:
  python support_review_classifier_ultra.py --input reviews.xlsx --comment-column Comments --output test.xlsx --limit 20
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DEFAULT = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW_TOKENS_DEFAULT = 70
SAVE_EVERY_MINUTES_DEFAULT = 5

PRIMARY_THEMES = [
    "Resolution quality",
    "Responsiveness / speed",
    "Communication / updates",
    "Process experience",
    "Ownership / follow-through",
    "Knowledge / competence",
    "Professionalism / helpfulness",
    "Generic / low-signal feedback",
]

SENTIMENTS = ["Positive", "Neutral", "Negative", "Mixed"]
PRIORITIES = ["Low", "Medium", "High"]
FOLLOW_UP = ["Yes", "No"]
DRIVER_TYPES = ["People", "Process", "Technology", "Communication", "Experience", "People / Process", "Process / Technology", "Communication / People"]

OUTPUT_COLUMNS = [
    "primary_theme",
    "secondary_theme",
    "sentiment",
    "driver_type",
    "recommended_action",
    "priority",
    "follow_up_required",
    "decision_source",
    "matched_rule",
    "llm_primary_theme",
    "confidence",
]

# Higher number wins. This fixes overlap.
RULES = [
    # Resolution beats everything if issue outcome is bad.
    (100, "Resolution quality", "Unresolved / not fixed", "Process / Technology", "Review resolution path and verify closure criteria", "High", "Yes", [
        r"\bnot\s+(fixed|resolved|solved|sorted|working)\b",
        r"\bstill\s+(not\s+)?(broken|unresolved|not working|an issue|a problem|failing)\b",
        r"\b(issue|problem)\s+(still\s+)?(remains|persists|continues|ongoing)\b",
        r"\b(didn'?t|did not|doesn'?t|does not)\s+(fix|resolve|solve)\b",
        r"\bno\s+resolution\b",
        r"\bclosed\s+(without|before).*(resolv|fix|solv)",
        r"\bticket\s+(was\s+)?closed.*(still|not|without)",
        r"\bhad\s+to\s+reopen\b",
        r"\breopened\b",
        r"\bunable\s+to\s+fix\b",
    ]),
    (95, "Resolution quality", "Temporary or incomplete fix", "Process / Technology", "Check fix durability and root-cause resolution", "High", "Yes", [
        r"\bcame\s+back\b",
        r"\breturned\b",
        r"\btemporary\s+fix\b",
        r"\bworkaround\b",
        r"\bnot\s+permanent\b",
        r"\bsame\s+issue\s+again\b",
        r"\bkeeps\s+happening\b",
    ]),
    # Positive resolved.
    (90, "Resolution quality", "Resolved successfully", "People / Process", "Capture good resolution practice", "Low", "No", [
        r"\b(resolved|solved|fixed|sorted|completed)\b",
        r"\bquick\s+resolution\b",
        r"\bissue\s+(was\s+)?(resolved|fixed|solved)\b",
        r"\bproblem\s+(was\s+)?(resolved|fixed|solved)\b",
        r"\bfully\s+resolved\b",
    ]),
    # Ownership.
    (80, "Ownership / follow-through", "Ownership / follow-through", "People / Process", "Reduce handoffs and clarify case ownership", "Medium", "Yes", [
        r"\btransferred\b",
        r"\bbounced\b",
        r"\bpassed\s+(around|between)\b",
        r"\bbetween\s+(different\s+)?teams\b",
        r"\bmultiple\s+teams\b",
        r"\bno\s+ownership\b",
        r"\bnobody\s+owned\b",
        r"\bunclear\s+owner(ship)?\b",
        r"\bescalat(ed|ion)\b",
        r"\bfollow\s+through\b",
    ]),
    # Process.
    (70, "Process experience", "Process friction", "Process", "Simplify process and remove unnecessary friction", "Medium", "No", [
        r"\bcomplicated\b",
        r"\bcomplex\b",
        r"\bcumbersome\b",
        r"\btoo\s+many\s+(steps|approvals|forms|processes)\b",
        r"\bapproval(s)?\b",
        r"\brepeat(ed)?\s+(my\s+)?(information|details)\b",
        r"\bsame\s+information\b",
        r"\bprocess\s+(was\s+)?(hard|difficult|painful|confusing)\b",
        r"\bstuck\s+with\s+it\b",
    ]),
    # Communication.
    (60, "Communication / updates", "Communication / updates", "Communication / People", "Improve update cadence and clarity", "Medium", "No", [
        r"\bno\s+updates?\b",
        r"\bwithout\s+updates?\b",
        r"\bnobody\s+updated\b",
        r"\bnot\s+kept\s+informed\b",
        r"\bpoor\s+communication\b",
        r"\black\s+of\s+communication\b",
        r"\bunclear\b",
        r"\bconfusing\b",
        r"\bno\s+explanation\b",
        r"\bpoor\s+follow[-\s]?up\b",
        r"\bunderstand\s+well\b",
    ]),
    # Response speed.
    (50, "Responsiveness / speed", "Response speed", "People / Process", "Investigate queue/routing delay", "Medium", "Yes", [
        r"\bslow\s+response\b",
        r"\bresponse\s+(was\s+)?slow\b",
        r"\bno\s+response\b",
        r"\btook\s+too\s+long\b",
        r"\blong\s+wait\b",
        r"\bwait(ed|ing)\b",
        r"\bdelay(ed|s)?\b",
        r"\bresponse\s+took\b",
        r"\bslow\s+to\s+get\s+resolved\b",
        r"\bquick\s+response\b",
        r"\bresponsive\b",
        r"\bgot\s+back\s+to\s+me\s+quickly\b",
    ]),
    # Knowledge.
    (40, "Knowledge / competence", "Knowledge competence", "People", "Improve knowledge guidance and troubleshooting clarity", "High", "Yes", [
        r"\bdidn'?t\s+know\b",
        r"\bdid not\s+know\b",
        r"\bwrong\s+advice\b",
        r"\bincorrect\s+advice\b",
        r"\black\s+of\s+knowledge\b",
        r"\bnot\s+knowledgeable\b",
        r"\bpoor\s+expertise\b",
        r"\binexperienced\b",
        r"\bcommands?\b",
        r"\bguidance\b",
        r"\btroubleshooting\b",
    ]),
    # Professionalism/helpfulness.
    (30, "Professionalism / helpfulness", "Professionalism / helpfulness", "People", "Capture agent behaviour as good practice", "Low", "No", [
        r"\bvery\s+helpful\b",
        r"\bhelpful\b",
        r"\bprofessional\b",
        r"\bfriendly\b",
        r"\bcourteous\b",
        r"\bexcellent\s+support\b",
        r"\bgreat\s+support\b",
        r"\bamazing\b",
    ]),
]

NEGATIVE_MARKERS = ["not fixed", "not resolved", "unresolved", "slow", "delay", "no update", "no response", "complicated", "confusing", "poor", "bad", "frustrating", "unable", "closed", "reopen", "stuck", "problem"]
POSITIVE_MARKERS = ["good", "great", "excellent", "helpful", "professional", "quick", "fast", "resolved", "fixed", "solved", "smooth", "easy", "impressed", "amazing", "thank"]


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def rule_match(comment: Any) -> Optional[Dict[str, Any]]:
    text = clean_text(comment)
    if not text:
        return None

    best = None
    for priority, primary, secondary, driver, action, pri, follow, patterns in RULES:
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                if best is None or priority > best["priority_score"]:
                    best = {
                        "primary_theme": primary,
                        "secondary_theme": secondary,
                        "driver_type": driver,
                        "recommended_action": action,
                        "priority": pri,
                        "follow_up_required": follow,
                        "matched_rule": pattern,
                        "priority_score": priority,
                    }

    # Absolute override: negated resolution beats positive resolved.
    unresolved_phrases = ["not resolved", "not fixed", "not solved", "still unresolved", "still not", "issue remains", "problem remains", "persists", "no resolution", "closed without", "unable to fix"]
    if any(x in text for x in unresolved_phrases):
        return {
            "primary_theme": "Resolution quality",
            "secondary_theme": "Unresolved / not fixed",
            "driver_type": "Process / Technology",
            "recommended_action": "Review why issue was not resolved and verify closure process",
            "priority": "High",
            "follow_up_required": "Yes",
            "matched_rule": "absolute_unresolved_override",
            "priority_score": 1000,
        }

    return best


def infer_sentiment(comment: Any, primary_theme: str, secondary_theme: str) -> str:
    text = clean_text(comment)
    has_neg = any(x in text for x in NEGATIVE_MARKERS)
    has_pos = any(x in text for x in POSITIVE_MARKERS)

    if secondary_theme in ["Unresolved / not fixed", "Temporary or incomplete fix"]:
        return "Mixed" if has_pos else "Negative"
    if secondary_theme == "Resolved successfully":
        return "Mixed" if has_neg else "Positive"
    if has_pos and has_neg:
        return "Mixed"
    if has_neg:
        return "Negative"
    if has_pos:
        return "Positive"
    return "Neutral"


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
Classify this support survey comment.
Return JSON only.

Primary themes:
{PRIMARY_THEMES}

Rules:
- Pick exactly one primary_theme.
- Pick a short secondary_theme.
- Helpful/quick but not fixed = Resolution quality.
- Resolved but slow = Resolution quality, secondary_theme = Resolved successfully.
- No updates/unclear = Communication / updates.
- Bounced/escalated/transferred = Ownership / follow-through.
- Complex/repeated info/approvals = Process experience.
- If only emoji or no detail = Generic / low-signal feedback.

JSON schema:
{{"primary_theme":"","secondary_theme":"","sentiment":"","driver_type":"","recommended_action":"","priority":"","follow_up_required":"","confidence":0.0}}

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
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1400)
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
    return extract_json(decoded)


def normalize_output(data: Dict[str, Any], comment: Any, source: str, matched_rule: str = "") -> Dict[str, Any]:
    primary = data.get("primary_theme", "Generic / low-signal feedback")
    if primary not in PRIMARY_THEMES:
        primary = "Generic / low-signal feedback"

    secondary = str(data.get("secondary_theme", "")).strip() or "-"
    sentiment = data.get("sentiment", infer_sentiment(comment, primary, secondary))
    if sentiment not in SENTIMENTS:
        sentiment = infer_sentiment(comment, primary, secondary)

    driver = data.get("driver_type", "Experience")
    if driver not in DRIVER_TYPES:
        driver = "Experience"

    priority = data.get("priority", "Medium")
    if priority not in PRIORITIES:
        priority = "Medium"

    follow = data.get("follow_up_required", "No")
    if follow not in FOLLOW_UP:
        follow = "No"

    action = str(data.get("recommended_action", "Review comment for service improvement")).strip()
    try:
        conf = float(data.get("confidence", 0.80))
    except Exception:
        conf = 0.80
    conf = max(0.0, min(1.0, conf))

    return {
        "primary_theme": primary,
        "secondary_theme": secondary,
        "sentiment": sentiment,
        "driver_type": driver,
        "recommended_action": action,
        "priority": priority,
        "follow_up_required": follow,
        "decision_source": source,
        "matched_rule": matched_rule,
        "llm_primary_theme": data.get("llm_primary_theme", ""),
        "confidence": round(conf, 3),
    }


def classify(tokenizer, model, comment: Any, max_new_tokens: int) -> Dict[str, Any]:
    text = clean_text(comment)
    if not text or len(text) <= 2:
        return normalize_output({
            "primary_theme": "Generic / low-signal feedback",
            "secondary_theme": "Emoji / low-signal feedback",
            "sentiment": "Neutral",
            "driver_type": "Experience",
            "recommended_action": "Flag as low-signal feedback; insufficient specificity for root-cause classification",
            "priority": "Medium",
            "follow_up_required": "No",
            "confidence": 0.98,
        }, comment, "empty_or_low_signal")

    rule = rule_match(text)
    if rule and rule["priority_score"] >= 70:
        return normalize_output({
            "primary_theme": rule["primary_theme"],
            "secondary_theme": rule["secondary_theme"],
            "sentiment": infer_sentiment(text, rule["primary_theme"], rule["secondary_theme"]),
            "driver_type": rule["driver_type"],
            "recommended_action": rule["recommended_action"],
            "priority": rule["priority"],
            "follow_up_required": rule["follow_up_required"],
            "confidence": 0.97,
        }, comment, "rule_high_confidence", rule["matched_rule"])

    try:
        llm = call_llm(tokenizer, model, text, max_new_tokens)
        llm_primary = llm.get("primary_theme", "")
    except Exception:
        llm = {}
        llm_primary = ""

    # If a softer rule exists, let it beat uncertain LLM drift.
    if rule:
        return normalize_output({
            "primary_theme": rule["primary_theme"],
            "secondary_theme": rule["secondary_theme"],
            "sentiment": infer_sentiment(text, rule["primary_theme"], rule["secondary_theme"]),
            "driver_type": rule["driver_type"],
            "recommended_action": rule["recommended_action"],
            "priority": rule["priority"],
            "follow_up_required": rule["follow_up_required"],
            "confidence": 0.90,
            "llm_primary_theme": llm_primary,
        }, comment, "rule_over_llm", rule["matched_rule"])

    if llm:
        llm["llm_primary_theme"] = llm_primary
        return normalize_output(llm, comment, "llm")

    return normalize_output({
        "primary_theme": "Generic / low-signal feedback",
        "secondary_theme": "Unclassified",
        "sentiment": "Neutral",
        "driver_type": "Experience",
        "recommended_action": "Review comment manually if business critical",
        "priority": "Medium",
        "follow_up_required": "No",
        "confidence": 0.50,
    }, comment, "fallback")


def load_or_create_df(input_file: str, output_file: str, comment_column: str, limit: Optional[int]) -> pd.DataFrame:
    if Path(output_file).exists():
        print(f"Resuming existing output: {output_file}")
        df = pd.read_excel(output_file)
    else:
        df = pd.read_excel(input_file)
        if limit:
            df = df.head(limit).copy()
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None
    if comment_column not in df.columns:
        raise ValueError(f"Column '{comment_column}' not found. Available: {list(df.columns)}")
    return df


def is_done(row: pd.Series) -> bool:
    v = row.get("primary_theme")
    return isinstance(v, str) and v.strip() and v.strip().lower() != "nan"


def save(df: pd.DataFrame, output_file: str, reason: str):
    df.to_excel(output_file, index=False)
    print(f"Saved {output_file} ({reason})")


def process(input_file: str, output_file: str, comment_column: str, model_name: str, max_new_tokens: int, save_every_minutes: int, limit: Optional[int]):
    if not Path(input_file).exists() and not Path(output_file).exists():
        raise FileNotFoundError(input_file)
    df = load_or_create_df(input_file, output_file, comment_column, limit)
    tokenizer, model = load_model(model_name)
    pending = [i for i, row in df.iterrows() if not is_done(row)]
    print(f"Rows total: {len(df)}")
    print(f"Rows pending: {len(pending)}")

    start = time.time()
    last_save = time.time()
    interval = save_every_minutes * 60
    for n, idx in enumerate(pending, 1):
        t0 = time.time()
        out = classify(tokenizer, model, df.at[idx, comment_column], max_new_tokens)
        for col in OUTPUT_COLUMNS:
            df.at[idx, col] = out[col]
        elapsed = time.time() - start
        eta = ((elapsed / n) * (len(pending) - n)) / 60 if n else 0
        print(f"[{n}/{len(pending)}] row={idx+1} theme={out['primary_theme']} sentiment={out['sentiment']} source={out['decision_source']} {time.time()-t0:.1f}s ETA={eta:.1f}m")
        if time.time() - last_save >= interval:
            save(df, output_file, "checkpoint")
            last_save = time.time()
    save(df, output_file, "final")
    print("\nPrimary theme distribution:")
    print(df["primary_theme"].value_counts(dropna=False))
    print("\nDecision source distribution:")
    print(df["decision_source"].value_counts(dropna=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="reviews.xlsx")
    p.add_argument("--output", default="reviews_classified.xlsx")
    p.add_argument("--comment-column", default="Comments")
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--save-every-minutes", type=int, default=SAVE_EVERY_MINUTES_DEFAULT)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    process(args.input, args.output, args.comment_column, args.model, args.max_new_tokens, args.save_every_minutes, args.limit)


if __name__ == "__main__":
    main()
