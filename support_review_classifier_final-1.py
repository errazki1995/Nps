# Final Hybrid Support Review Classifier
#
# Taxonomy:
# - Response Quality
# - Responsiveness
# - Communication
# - Knowledge / Competence
# - Ownership
# - Access / Provisioning
# - System / Tool Reliability
# - Professionalism
# - Process
# - Generic / Low Signal
#
# Features:
# - Excel input/output
# - Keyword regex engine
# - Local Qwen LLM fallback
# - keyword_theme + llm_theme + preferred_theme
# - Deterministic preferred-theme selection
# - Duplicate-comment cache
# - Checkpoint saving every 5 minutes
#
# Install:
#   pip install pandas openpyxl torch transformers accelerate
#
# Test first:
#   python support_review_classifier_final.py --input reviews.xlsx --comment-column Comments --output test.xlsx --limit 30
#
# Full run:
#   python support_review_classifier_final.py --input reviews.xlsx --comment-column Comments --output reviews_classified.xlsx
#
# Faster/lighter model:
#   python support_review_classifier_final.py --model Qwen/Qwen2.5-1.5B-Instruct

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 45
DEFAULT_SAVE_EVERY_MINUTES = 5

THEMES = [
    "Response Quality",
    "Responsiveness",
    "Communication",
    "Knowledge / Competence",
    "Ownership",
    "Access / Provisioning",
    "System / Tool Reliability",
    "Professionalism",
    "Process",
    "Generic / Low Signal",
    "Unknown",
]

SENTIMENTS = ["positive", "neutral", "negative", "mixed"]

OUTPUT_COLUMNS = [
    "keyword_theme",
    "keyword_confidence",
    "keyword_matched_rule",
    "llm_theme",
    "llm_confidence",
    "preferred_theme",
    "sentiment",
    "decision_source",
    "needs_review",
]

THEME_PRIORITY = {
    "Response Quality": 100,
    "Access / Provisioning": 90,
    "System / Tool Reliability": 85,
    "Knowledge / Competence": 80,
    "Ownership": 75,
    "Communication": 70,
    "Responsiveness": 60,
    "Professionalism": 50,
    "Process": 40,
    "Generic / Low Signal": 10,
    "Unknown": 0,
}


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[’‘`]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def cache_key(value: Any) -> str:
    return hashlib.md5(normalize_text(value).encode("utf-8")).hexdigest()


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


# ==========================================================
# REGEX KEYWORD RULES
# ==========================================================
# Important:
# - Rules are phrase-based, not vague single-word based.
# - "Response Quality" means solved/not solved/fix quality.
# - "Responsiveness" means speed of reply or turnaround.
# - "Communication" means clarity, updates, explanations.
# - "Knowledge / Competence" means ability/skill/advice quality.


KEYWORD_RULES = {
    "Response Quality": {
        "confidence": 0.98,
        "patterns": [
            # solved / fixed / resolved
            r"\b(issue|problem|case|request|ticket)\s+(was\s+)?(fully\s+)?(resolved|fixed|solved|sorted|completed|addressed|handled)\b",
            r"\b(my|the)\s+(issue|problem|case|request|ticket)\s+(was\s+)?(fully\s+)?(resolved|fixed|solved|sorted|completed|addressed|handled)\b",
            r"\b(resolved|fixed|solved|sorted|completed|addressed|handled)\s+(my|the)\s+(issue|problem|case|request|ticket)\b",
            r"\bquick\s+resolution\b",
            r"\bquickly\s+(resolved|fixed|solved|sorted|completed|addressed|handled)\b",
            r"\b(resolved|fixed|solved|sorted|completed|addressed|handled)\s+quickly\b",
            r"\bgot\s+(it|this|the issue|the problem|the ticket|the case)\s+(resolved|fixed|solved|sorted|completed)\b",
            r"\bworking\s+now\b",
            r"\bworks?\s+now\b",
            r"\bback\s+up\s+and\s+running\b",

            # not solved / poor fix
            r"\bnot\s+(fixed|resolved|solved|sorted|working|completed|addressed|handled)\b",
            r"\bnever\s+(fixed|resolved|solved|sorted|addressed|handled)\b",
            r"\bwasn'?t\s+(fixed|resolved|solved|sorted|completed|addressed|handled)\b",
            r"\bdidn'?t\s+(fix|resolve|solve|complete|address|handle)\b",
            r"\bdid\s+not\s+(fix|resolve|solve|complete|address|handle)\b",
            r"\bdoesn'?t\s+(fix|resolve|solve|work|address)\b",
            r"\bdoes\s+not\s+(fix|resolve|solve|work|address)\b",
            r"\bstill\s+(not\s+)?(unresolved|broken|not working|failing|open|pending|outstanding)\b",
            r"\bstill\s+(an\s+)?(issue|problem|fault|error|case)\b",
            r"\b(issue|problem|fault|error|case|ticket)\s+(still\s+)?(remains|persists|continues|exists|ongoing|open|pending|outstanding)\b",
            r"\bno\s+(fix|solution|resolution|answer|outcome)\b",
            r"\bwithout\s+(a\s+)?(fix|solution|resolution|answer|outcome)\b",
            r"\bunresolved\b",
            r"\bnot\s+addressed\b",
            r"\bnot\s+handled\b",

            # reopened / closed too early
            r"\bhad\s+to\s+reopen\b",
            r"\breopened\b",
            r"\bre-opened\b",
            r"\bticket\s+(was\s+)?closed.*(without|before|still|not)",
            r"\bcase\s+(was\s+)?closed.*(without|before|still|not)",
            r"\bclosed\s+(without|before).*(fix|resolv|solv|complete|address)",
            r"\bmarked\s+(as\s+)?resolved.*(but|however|although|still|not)",
            r"\bmarked\s+(as\s+)?closed.*(but|however|although|still|not)",

            # bad resolution quality
            r"\bworkaround\s+only\b",
            r"\bonly\s+a\s+workaround\b",
            r"\btemporary\s+(fix|solution|workaround)\b",
            r"\bnot\s+a\s+permanent\s+(fix|solution)\b",
            r"\b(issue|problem|fault|error)\s+(came\s+back|returned)\b",
            r"\bsame\s+(issue|problem|fault|error)\s+(again|returned|came back)\b",
            r"\bkeeps?\s+(happening|failing|breaking|returning)\b",
            r"\bkept\s+(happening|failing|breaking|returning)\b",
            r"\bwrong\s+(solution|resolution|fix)\b",
            r"\bpoor\s+(solution|resolution|fix)\b",
            r"\bincomplete\s+(solution|resolution|fix)\b",
            r"\broot\s+cause\s+(was\s+)?not\s+addressed\b",
            r"\bfix\s+(did\s+not|didn'?t)\s+work\b",
        ],
    },

    "Responsiveness": {
        "confidence": 0.91,
        "patterns": [
            r"\bslow\s+(response|reply|turnaround)\b",
            r"\bresponse\s+(was\s+)?(very\s+)?slow\b",
            r"\breply\s+(was\s+)?(very\s+)?slow\b",
            r"\bslow\s+to\s+(reply|respond|get back)\b",
            r"\bno\s+(response|reply|answer|acknowledgement)\b",
            r"\btook\s+too\s+long\s+to\s+(reply|respond|get back)\b",
            r"\blong\s+(wait|delay|turnaround)\b",
            r"\blong\s+wait\s+(for\s+)?(response|reply)\b",
            r"\bwaited\s+(days|weeks|months)\s+for\s+(a\s+)?(response|reply|answer)\b",
            r"\bwaited\s+(too\s+)?long\s+for\s+(a\s+)?(response|reply|answer)\b",
            r"\bdelayed\s+(response|reply|turnaround)\b",
            r"\bresponse\s+took\s+(too\s+)?long\b",
            r"\breply\s+took\s+(too\s+)?long\b",
            r"\bchased\s+(several|multiple|many)?\s*times\s+for\s+(a\s+)?(response|reply|answer)\b",
            r"\bneeded\s+(to\s+)?chase\b",
            r"\bhad\s+to\s+chase\b",
        ],
    },

    "Communication": {
        "confidence": 0.92,
        "patterns": [
            r"\bno\s+updates?\b",
            r"\bwithout\s+updates?\b",
            r"\bnobody\s+updated\b",
            r"\bno\s+one\s+updated\b",
            r"\bnot\s+kept\s+informed\b",
            r"\bwasn'?t\s+kept\s+informed\b",
            r"\bwasn'?t\s+informed\b",
            r"\bpoor\s+communication\b",
            r"\black\s+of\s+communication\b",
            r"\bno\s+communication\b",
            r"\bno\s+explanation\b",
            r"\bdidn'?t\s+explain\b",
            r"\bdid\s+not\s+explain\b",
            r"\bunclear\s+(explanation|communication|instructions|guidance|response|answer)\b",
            r"\bconfusing\s+(explanation|communication|instructions|guidance|response|answer)\b",
            r"\bpoor\s+follow[-\s]?up\b",
            r"\bno\s+follow[-\s]?up\b",
            r"\bkept\s+me\s+in\s+the\s+dark\b",
            r"\bconflicting\s+(information|updates|advice|messages)\b",
            r"\bmixed\s+messages\b",
        ],
    },

    "Knowledge / Competence": {
        "confidence": 0.91,
        "patterns": [
            r"\bdidn'?t\s+know\s+(how|what|why)\b",
            r"\bdid\s+not\s+know\s+(how|what|why)\b",
            r"\bdidn'?t\s+understand\b",
            r"\bdid\s+not\s+understand\b",
            r"\b(agent|engineer|support|advisor)\s+(didn'?t|did not)\s+understand\b",
            r"\bwrong\s+advice\b",
            r"\bincorrect\s+(advice|information|guidance|instructions)\b",
            r"\binaccurate\s+(advice|information|guidance|instructions)\b",
            r"\black\s+of\s+knowledge\b",
            r"\bnot\s+knowledgeable\b",
            r"\binexperienced\b",
            r"\bpoor\s+expertise\b",
            r"\black(ed)?\s+expertise\b",
            r"\bunable\s+to\s+diagnose\b",
            r"\bcouldn'?t\s+diagnose\b",
            r"\bcould\s+not\s+diagnose\b",
            r"\bfailed\s+to\s+diagnose\b",
            r"\bunable\s+to\s+identify\s+(the\s+)?(root\s+)?cause\b",
            r"\bcouldn'?t\s+identify\s+(the\s+)?(root\s+)?cause\b",
            r"\bcould\s+not\s+identify\s+(the\s+)?(root\s+)?cause\b",
            r"\bunable\s+to\s+(fix|resolve|solve)\b",
            r"\bcouldn'?t\s+(fix|resolve|solve)\b",
            r"\bcould\s+not\s+(fix|resolve|solve)\b",
            r"\bfailed\s+to\s+(fix|resolve|solve)\b",
        ],
    },

    "Ownership": {
        "confidence": 0.93,
        "patterns": [
            r"\btransferred\s+(me\s+)?(between|to|from)\b",
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
            r"\bescalated\s+between\s+teams\b",
            r"\bwrong\s+team\b",
            r"\bwrong\s+queue\b",
        ],
    },

    "Access / Provisioning": {
        "confidence": 0.94,
        "patterns": [
            r"\baccess\s+(denied|issue|problem|request|not granted|missing|needed|required)\b",
            r"\bno\s+access\b",
            r"\bwithout\s+access\b",
            r"\bmissing\s+access\b",
            r"\bpermission\s+(denied|issue|problem|missing|required|not granted)\b",
            r"\bmissing\s+permission(s)?\b",
            r"\bunable\s+to\s+log\s*in\b",
            r"\bcannot\s+log\s*in\b",
            r"\bcan'?t\s+log\s*in\b",
            r"\blogin\s+(failed|issue|problem|not working)\b",
            r"\blog\s*in\s+(failed|issue|problem|not working)\b",
            r"\baccount\s+(creation|setup|provisioning|not created|not setup|not set up)\b",
            r"\bprovisioning\s+(delay|issue|problem|failed)\b",
            r"\bpassword\s+(reset|issue|problem|expired)\b",
            r"\bshared\s+mailbox\s+access\b",
            r"\blicen[cs]e\s+(missing|not assigned|issue|required)\b",
            r"\brole\s+(missing|not assigned|required)\b",
            r"\buser\s+(not\s+)?provisioned\b",
            r"\baccess\s+was\s+not\s+provisioned\b",
            r"\bgroup\s+membership\b",
            r"\bvpn\s+access\b",
            r"\bmailbox\s+access\b",
        ],
    },

    "System / Tool Reliability": {
        "confidence": 0.93,
        "patterns": [
            r"\bsystem\s+(down|failed|not working|unavailable|crashed|slow)\b",
            r"\bapplication\s+(crashed|failed|not working|unavailable|slow|freezing)\b",
            r"\bapp\s+(crashed|failed|not working|slow|freezing)\b",
            r"\btool\s+(crashed|failed|not working|unavailable|slow|freezing)\b",
            r"\bportal\s+(crashed|failed|not working|unavailable|slow|freezing)\b",
            r"\berror\s+(message|code)\b",
            r"\bbug\b",
            r"\boutage\b",
            r"\bperformance\s+(issue|problem)\b",
            r"\bslow\s+(system|application|app|tool|portal)\b",
            r"\bkeeps?\s+(crashing|freezing|failing)\b",
            r"\bkept\s+(crashing|freezing|failing)\b",
            r"\bcrash(ed|es|ing)?\b",
            r"\bfreez(e|es|ing)\b",
            r"\bnot\s+loading\b",
            r"\bpage\s+not\s+loading\b",
            r"\bservice\s+unavailable\b",
        ],
    },

    "Professionalism": {
        "confidence": 0.90,
        "patterns": [
            r"\brude\b",
            r"\bunprofessional\b",
            r"\bdismissive\b",
            r"\bimpolite\b",
            r"\bpatroni[sz]ing\b",
            r"\bcondescending\b",
            r"\bunhelpful\s+(agent|engineer|person|support|staff)\b",
            r"\bnot\s+(friendly|polite|professional|helpful)\b",
            r"\bfriendly\b",
            r"\bcourteous\b",
            r"\bpolite\b",
            r"\bprofessional\b",
            r"\brespectful\b",
            r"\bhelpful\s+(agent|engineer|person|support|staff)\b",
        ],
    },

    "Process": {
        "confidence": 0.91,
        "patterns": [
            r"\btoo\s+many\s+(steps|approvals|forms|processes|stages)\b",
            r"\bprocess\s+(was\s+)?(too\s+)?(complicated|complex|cumbersome|painful|difficult|hard|long-winded)\b",
            r"\bcomplicated\s+process\b",
            r"\bcomplex\s+process\b",
            r"\bcumbersome\s+process\b",
            r"\bbureaucratic\b",
            r"\bred\s+tape\b",
            r"\brepeat(ed)?\s+(my\s+)?(information|details)\b",
            r"\bsame\s+(information|details)\s+(again|multiple times|several times)\b",
            r"\basked\s+for\s+the\s+same\s+(information|details)\b",
            r"\bduplicate\s+(information|details|request)\b",
            r"\btoo\s+much\s+paperwork\b",
            r"\bapproval\s+(delay|delays|process|took too long)\b",
        ],
    },

    "Generic / Low Signal": {
        "confidence": 0.75,
        "patterns": [
            r"^\s*(good|great|excellent|ok|okay|fine|thanks|thank you|no comment|none|n/a|na)\s*[.!]*\s*$",
            r"\bgood\s+service\b",
            r"\bgreat\s+support\b",
            r"\bexcellent\s+support\b",
            r"\bno\s+comment\b",
            r"\bn/?a\b",
        ],
    },
}


POSITIVE_TERMS = [
    "good", "great", "excellent", "helpful", "professional", "quick", "fast",
    "smooth", "easy", "impressed", "thank", "thanks", "perfect", "amazing",
    "brilliant", "satisfied", "happy", "resolved", "fixed", "solved",
]

NEGATIVE_TERMS = [
    "not", "no", "slow", "delay", "delayed", "poor", "bad", "unable",
    "couldn't", "could not", "failed", "frustrating", "difficult",
    "confusing", "complicated", "unresolved", "still", "closed",
    "wrong", "incorrect", "painful", "rude", "unprofessional",
]


def infer_sentiment(text: str, theme: str) -> str:
    has_positive = any(term in text for term in POSITIVE_TERMS)
    has_negative = any(term in text for term in NEGATIVE_TERMS)

    if has_positive and has_negative:
        return "mixed"
    if has_negative:
        return "negative"
    if has_positive:
        return "positive"
    return "neutral"


def apply_keyword_rules(comment: Any) -> Optional[Dict[str, Any]]:
    text = normalize_text(comment)

    if not text:
        return {
            "theme": "Unknown",
            "confidence": 0.0,
            "matched_rule": "",
            "sentiment": "neutral",
        }

    matches = []

    for theme, config in KEYWORD_RULES.items():
        for pattern in config["patterns"]:
            if has(pattern, text):
                matches.append({
                    "theme": theme,
                    "confidence": config["confidence"],
                    "matched_rule": pattern,
                    "priority": THEME_PRIORITY[theme],
                })

    if not matches:
        return None

    matches.sort(key=lambda x: (x["priority"], x["confidence"]), reverse=True)
    best = matches[0]

    return {
        "theme": best["theme"],
        "confidence": best["confidence"],
        "matched_rule": best["matched_rule"],
        "sentiment": infer_sentiment(text, best["theme"]),
    }


def load_model(model_name: str):
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

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
Classify this support review into exactly ONE theme.

Allowed themes:
Response Quality
Responsiveness
Communication
Knowledge / Competence
Ownership
Access / Provisioning
System / Tool Reliability
Professionalism
Process
Generic / Low Signal
Unknown

Priority if multiple themes apply:
1 Response Quality
2 Access / Provisioning
3 System / Tool Reliability
4 Knowledge / Competence
5 Ownership
6 Communication
7 Responsiveness
8 Professionalism
9 Process
10 Generic / Low Signal
11 Unknown

Definitions:
- Response Quality: issue solved or not solved, fix quality, workaround, temporary fix, issue returned, ticket closed too early.
- Responsiveness: speed of reply, no response, delayed reply, long wait for response.
- Communication: no updates, unclear explanation, confusing guidance, poor follow-up.
- Knowledge / Competence: wrong advice, lack of knowledge, agent could not diagnose or did not understand.
- Ownership: transferred/bounced between teams, unclear owner, handoffs, wrong team.
- Access / Provisioning: access, permissions, login, account creation, provisioning, password, license, role.
- System / Tool Reliability: bug, outage, crash, freezing, error, system/app/tool down or slow.
- Professionalism: rude, dismissive, unprofessional, polite, friendly, courteous, helpful attitude.
- Process: too many approvals/steps, repeated information, bureaucracy, paperwork.
- Generic / Low Signal: vague praise/complaint without a specific theme.
- Unknown: impossible to classify.

Rules:
- Return JSON only.
- Do not invent themes.
- Choose exactly one theme.
- If the comment is about whether the issue was solved or not, choose Response Quality.
- If the comment says support could not diagnose or did not know how, choose Knowledge / Competence.
- If access/login/provisioning is mentioned, choose Access / Provisioning unless Response Quality is clearly the main point.

JSON:
{{"theme":"","sentiment":"","confidence":0.0}}

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

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1600,
    )

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

    theme = data.get("theme", "Unknown")
    sentiment = data.get("sentiment", "neutral")
    confidence = data.get("confidence", 0.0)

    if theme not in THEMES:
        theme = "Unknown"

    if sentiment not in SENTIMENTS:
        sentiment = "neutral"

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    return {
        "theme": theme,
        "sentiment": sentiment,
        "confidence": confidence,
    }


def choose_preferred_theme(
    keyword_result: Optional[Dict[str, Any]],
    llm_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    keyword_theme = keyword_result["theme"] if keyword_result else "Unknown"
    keyword_conf = keyword_result["confidence"] if keyword_result else 0.0
    keyword_rule = keyword_result["matched_rule"] if keyword_result else ""
    keyword_sentiment = keyword_result["sentiment"] if keyword_result else "neutral"

    llm_theme = llm_result["theme"] if llm_result else "Unknown"
    llm_conf = llm_result["confidence"] if llm_result else 0.0
    llm_sentiment = llm_result["sentiment"] if llm_result else "neutral"

    if keyword_theme == "Response Quality" and keyword_conf >= 0.90:
        preferred = "Response Quality"
        source = "response_quality_keyword_override"
        sentiment = keyword_sentiment

    elif keyword_theme != "Unknown" and llm_theme == "Unknown":
        preferred = keyword_theme
        source = "keyword_only"
        sentiment = keyword_sentiment

    elif keyword_theme == "Unknown" and llm_theme != "Unknown":
        preferred = llm_theme
        source = "llm_only"
        sentiment = llm_sentiment

    elif keyword_theme == llm_theme and keyword_theme != "Unknown":
        preferred = keyword_theme
        source = "keyword_llm_agree"
        sentiment = keyword_sentiment if keyword_conf >= llm_conf else llm_sentiment

    elif keyword_conf >= 0.90 and THEME_PRIORITY.get(keyword_theme, 0) >= THEME_PRIORITY.get(llm_theme, 0):
        preferred = keyword_theme
        source = "keyword_priority"
        sentiment = keyword_sentiment

    elif llm_conf >= 0.90 and THEME_PRIORITY.get(llm_theme, 0) > THEME_PRIORITY.get(keyword_theme, 0):
        preferred = llm_theme
        source = "llm_higher_priority"
        sentiment = llm_sentiment

    else:
        if THEME_PRIORITY.get(keyword_theme, 0) >= THEME_PRIORITY.get(llm_theme, 0):
            preferred = keyword_theme
            source = "priority_keyword"
            sentiment = keyword_sentiment
        else:
            preferred = llm_theme
            source = "priority_llm"
            sentiment = llm_sentiment

    confidence = max(keyword_conf, llm_conf)

    needs_review = (
        preferred == "Unknown"
        or confidence < 0.70
        or (
            keyword_theme != "Unknown"
            and llm_theme != "Unknown"
            and keyword_theme != llm_theme
            and confidence < 0.85
        )
    )

    return {
        "keyword_theme": keyword_theme,
        "keyword_confidence": round(keyword_conf, 3),
        "keyword_matched_rule": keyword_rule,
        "llm_theme": llm_theme,
        "llm_confidence": round(llm_conf, 3),
        "preferred_theme": preferred,
        "sentiment": sentiment,
        "decision_source": source,
        "needs_review": needs_review,
    }


def classify(
    tokenizer,
    model,
    comment: Any,
    max_new_tokens: int,
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    text = normalize_text(comment)
    key = cache_key(text)

    if key in cache:
        cached = cache[key].copy()
        cached["decision_source"] = cached["decision_source"] + "_cached"
        return cached

    keyword_result = apply_keyword_rules(text)

    skip_llm_themes = {
        "Response Quality",
        "Access / Provisioning",
        "System / Tool Reliability",
        "Knowledge / Competence",
        "Ownership",
    }

    if (
        keyword_result
        and keyword_result["theme"] in skip_llm_themes
        and keyword_result["confidence"] >= 0.91
    ):
        final = choose_preferred_theme(keyword_result, None)
        final["decision_source"] = "keyword_only_high_confidence"
        cache[key] = final.copy()
        return final

    try:
        llm_result = call_llm(tokenizer, model, text, max_new_tokens)
    except Exception:
        llm_result = {
            "theme": "Unknown",
            "sentiment": "neutral",
            "confidence": 0.0,
        }

    final = choose_preferred_theme(keyword_result, llm_result)
    cache[key] = final.copy()
    return final


def row_done(row: pd.Series) -> bool:
    value = row.get("preferred_theme")
    return isinstance(value, str) and value.strip() != "" and value.strip().lower() != "nan"


def load_or_create_df(
    input_file: str,
    output_file: str,
    comment_column: str,
    limit: Optional[int],
) -> pd.DataFrame:
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
        raise ValueError(
            f"Column '{comment_column}' not found. Available columns: {list(df.columns)}"
        )

    return df


def save_output(df: pd.DataFrame, output_file: str, reason: str):
    df.to_excel(output_file, index=False)
    print(f"Saved: {output_file} ({reason})")


def process_excel(
    input_file: str,
    output_file: str,
    comment_column: str,
    model_name: str,
    max_new_tokens: int,
    save_every_minutes: int,
    limit: Optional[int],
):
    if not Path(input_file).exists() and not Path(output_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = load_or_create_df(
        input_file=input_file,
        output_file=output_file,
        comment_column=comment_column,
        limit=limit,
    )

    tokenizer, model = load_model(model_name)

    pending = [idx for idx, row in df.iterrows() if not row_done(row)]
    cache: Dict[str, Dict[str, Any]] = {}

    print(f"Rows total: {len(df)}")
    print(f"Rows pending: {len(pending)}")
    print(f"Model: {model_name}")
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
            f"preferred={output['preferred_theme']} "
            f"keyword={output['keyword_theme']} "
            f"llm={output['llm_theme']} "
            f"source={output['decision_source']} "
            f"time={row_time:.1f}s ETA={eta_min:.1f}m"
        )

        if time.time() - last_save >= save_interval:
            save_output(df, output_file, "checkpoint")
            last_save = time.time()

    save_output(df, output_file, "final")

    print("\nPreferred theme distribution:")
    print(df["preferred_theme"].value_counts(dropna=False))

    print("\nDecision source distribution:")
    print(df["decision_source"].value_counts(dropna=False))

    print("\nNeeds review:")
    print(df["needs_review"].value_counts(dropna=False))


def main():
    parser = argparse.ArgumentParser(
        description="Final hybrid support review classifier."
    )

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
