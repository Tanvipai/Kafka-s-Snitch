import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)


CATEGORIES = ["hr_data", "medical_phi", "credentials", "financial"]


COMPLIANCE_MAP = {
    "hr_data": ["GDPR"],
    "medical_phi": ["HIPAA"],
    "financial": ["GDPR"],
   
    "credentials": [],
}

RISK_TIER_MAP = {
    "credentials": "Critical",   # leaked secrets are immediately exploitable
    "medical_phi": "High",       # HIPAA exposure
    "financial": "High",         # payment data exposure
    "hr_data": "Medium",         # PII, but less immediately exploitable
}


KEYWORD_SIGNALS = {
    "hr_data": [
        "employee_id", "department", "salary", "hire_date", "manager",
        "compensation", "full_name",
    ],
    "medical_phi": [
        "patient", "diagnosis", "physician", "medication", "clinical",
        "mrn", "insurance id", "prescri",
    ],
    "credentials": [
        "password", "api_key", "secret_access_key", "aws_access_key",
        "token", "db_config", "connection", "secret_key",
    ],
    "financial": [
        "invoice", "amount due", "payment method", "transaction",
        "bill to", "card number",
    ],
}


FINDING_WEIGHTS = {
    "AWS_KEY": {"credentials": 2.0},
    "STRIPE_KEY": {"credentials": 2.0},
    "GITHUB_TOKEN": {"credentials": 2.0},
    "SECRET_ASSIGNMENT": {"credentials": 1.5},
    "API_KEY": {"credentials": 1.0},
    # SSN shows up in both our HR csvs and medical notes -- it can't
    # disambiguate alone, so it nudges both and lets keywords decide.
    "SSN": {"hr_data": 1.0, "medical_phi": 1.0},
    "CREDIT_CARD": {"financial": 2.0},
    "IBAN": {"financial": 1.5},
    # IPs show up in our config_script files (DB host) -- weak nudge
    # toward credentials, not strong enough to matter alone.
    "IP_ADDRESS": {"credentials": 0.3},
}


GENERIC_FINDING_BOOST = 0.2
GENERIC_FINDING_TYPES = {"EMAIL", "PHONE"}


TIER1_MARGIN_THRESHOLD = 1.0
TIER1_FLOOR_THRESHOLD = 1.5


TIER2_MARGIN_THRESHOLD = 0.20
TIER2_FLOOR_THRESHOLD = 0.45

TIER2_LABELS = {
    "hr_data": "an HR document containing employee personal information",
    "medical_phi": "a medical document containing patient health information",
    "credentials": "a document containing exposed passwords or API credentials",
    "financial": "a financial document containing payment or invoice information",
}

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


@dataclass
class ClassificationResult:
    category: str
    risk_tier: str
    decided_by_tier: int             # 1, 2, or 3
    confidence: float
    compliance_flags: list = field(default_factory=list)
    tier1_scores: dict = field(default_factory=dict)
    notes: Optional[str] = None


def score_categories(text: str, findings: list) -> dict:
    """
    Tier 1 scoring. Pure function -- no model calls -- so it's the part
    we can unit test without spaCy/BART/Ollama installed at all.
    """
    scores = {c: 0.0 for c in CATEGORIES}
    lowered = text.lower()

    for category, keywords in KEYWORD_SIGNALS.items():
        for kw in keywords:
            count = lowered.count(kw)
            if count:
                
                scores[category] += min(count, 3) * 0.5

    for finding in findings:
        weights = FINDING_WEIGHTS.get(finding.entity_type, {})
        for category, weight in weights.items():
            scores[category] += weight * finding.confidence

        if finding.entity_type in GENERIC_FINDING_TYPES:
            if any(scores.values()):
                leading_category = max(scores, key=scores.get)
                scores[leading_category] += GENERIC_FINDING_BOOST * finding.confidence

    return scores


def decide_tier1(scores: dict) -> tuple:
    
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_category, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - second_score

    confident = (
        margin >= TIER1_MARGIN_THRESHOLD
        and top_score >= TIER1_FLOOR_THRESHOLD
    )
    return top_category, top_score, confident


class CascadeClassifier:

    def classify(self, text: str, findings: list) -> ClassificationResult:
        scores = score_categories(text, findings)
        category, top_score, confident = decide_tier1(scores)

        if confident:
            return self._build_result(
                category=category,
                confidence=min(top_score / (top_score + 1.0), 0.99),
                tier=1,
                tier1_scores=scores,
                notes=None,
            )

        logger.info(
            f"tier 1 not confident (scores={scores}) -- escalating to tier 2"
        )
        return self._tier2_zero_shot(text, scores)

    def _tier2_zero_shot(self, text: str, tier1_scores: dict) -> ClassificationResult:
        from transformers import pipeline  

        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        label_text = list(TIER2_LABELS.values())
        label_keys = list(TIER2_LABELS.keys())

        result = classifier(text, candidate_labels=label_text)
        # result["labels"]/["scores"] are sorted descending already
        top_label_text, top_score = result["labels"][0], result["scores"][0]
        second_score = result["scores"][1] if len(result["scores"]) > 1 else 0.0

        top_category = label_keys[label_text.index(top_label_text)]
        margin = top_score - second_score

        confident = (
            margin >= TIER2_MARGIN_THRESHOLD
            and top_score >= TIER2_FLOOR_THRESHOLD
        )

        if confident:
            return self._build_result(
                category=top_category,
                confidence=top_score,
                tier=2,
                tier1_scores=tier1_scores,
                notes=f"tier1 scores: {tier1_scores}",
            )

        logger.info(
            f"tier 2 not confident (top={top_label_text}:{top_score:.2f}, "
            f"second={second_score:.2f}) -- escalating to tier 3"
        )
        return self._tier3_llm(text, tier1_scores)

    def _tier3_llm(self, text: str, tier1_scores: dict) -> ClassificationResult:
        prompt = (
            "Classify this document into exactly one category: "
            "hr_data, medical_phi, credentials, or financial.\n"
            "Then list every type of sensitive information present and "
            "explain briefly why it matters under GDPR Article 4 or HIPAA "
            "where relevant.\n"
            "Respond ONLY as JSON with keys: category, reasoning.\n\n"
            f"Document:\n{text[:3000]}"
        )

        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=120,  
            )
            response.raise_for_status()
            raw_output = response.json().get("response", "")
            parsed = json.loads(raw_output)
            category = parsed.get("category", "hr_data")
            if category not in CATEGORIES:
                category = "hr_data"
            reasoning = parsed.get("reasoning", "")
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"tier 3 LLM call failed: {e}")
            #Tier 3 
            category = max(tier1_scores, key=tier1_scores.get)
            reasoning = f"LLM call failed ({e}); fell back to tier1 best guess"

       
        return self._build_result(
            category=category,
            confidence=0.6,
            tier=3,
            tier1_scores=tier1_scores,
            notes=reasoning,
        )

    def _build_result(
        self, category: str, confidence: float, tier: int,
        tier1_scores: dict, notes: Optional[str],
    ) -> ClassificationResult:
        return ClassificationResult(
            category=category,
            risk_tier=RISK_TIER_MAP.get(category, "Medium"),
            decided_by_tier=tier,
            confidence=round(confidence, 2),
            compliance_flags=COMPLIANCE_MAP.get(category, []),
            tier1_scores=tier1_scores,
            notes=notes,
        )