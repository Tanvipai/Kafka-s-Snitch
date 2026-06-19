import math
import re
import logging
from dataclasses import dataclass

import phonenumbers

logger = logging.getLogger(__name__)


@dataclass
class Finding:

    entity_type: str
    masked_value: str
    char_position: int
    confidence: float
    context_snippet: str
    validation_passed: bool


def mask_ssn(value: str) -> str:
    digits = re.sub(r'\D', '', value)
    return f"XXX-XX-{digits[-4:]}"


def mask_credit_card(value: str) -> str:
    digits = re.sub(r'\D', '', value)
    return f"{digits[:4]}-XXXX-XXXX-{digits[-4:]}"


def mask_default(value: str) -> str:
    if len(value) <= 4:
        return "****"
    return value[:4] + "*" * (len(value) - 4)


# validation functions

def luhn_check(card_number: str) -> bool:
    digits = [int(d) for d in re.sub(r'\D', '', card_number)]
    if len(digits) < 13 or len(digits) > 19:
        return False

    for i in range(len(digits) - 2, -1, -2):
        digits[i] *= 2
        if digits[i] > 9:
            digits[i] -= 9

    return sum(digits) % 10 == 0


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    length = len(text)
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    return -sum((count / length) * math.log2(count / length)
                for count in freq.values())


def validate_phone(number: str, region: str = "US") -> bool:
    try:
        parsed = phonenumbers.parse(number, region)
        return phonenumbers.is_valid_number(parsed)
    except Exception:
        return False


def validate_iban(value: str) -> bool:
    """ real IBAN checksum (MOD-97 / ISO 7064), same role as Luhn for cards.
    Without this, any random 2-letters+2-digits+alnum token was being reported as a
    validated IBAN - pure false positive.
    """
    value = value.replace(' ', '').upper()
    if not (15 <= len(value) <= 34):
        return False
    if not re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]+$', value):
        return False

    rearranged = value[4:] + value[:4]
    try:
        numeric_str = ''.join(str(int(ch, 36)) for ch in rearranged)
        return int(numeric_str) % 97 == 1
    except ValueError:
        return False


# context scoring

SENSITIVE_CONTEXT_WORDS = [
    "ssn", "social security", "patient", "medical", "diagnosis",
    "credit card", "card number", "payment", "api key", "secret",
    "password", "token", "confidential", "hipaa", "gdpr", "private",
    "account", "routing", "iban", "passport", "license",
]


def context_score_boost(text: str, position: int, window: int = 50) -> float:
    start = max(0, position - window)
    end = min(len(text), position + window)
    surrounding = text[start:end].lower()

    for word in SENSITIVE_CONTEXT_WORDS:
        if word in surrounding:
            return 0.15
    return 0.0


class PatternScanner:

    # order now matters. Specific/structured patterns run first;
    
    # from being reported twice -- once under their specific entity type
    
    PATTERNS = {
        "AWS_KEY": re.compile(r'\b(AKIA[0-9A-Z]{16})\b'),
        "STRIPE_KEY": re.compile(r'\b(sk_live_[0-9a-zA-Z]{24,})\b'),
        "GITHUB_TOKEN": re.compile(r'\b(ghp_[0-9a-zA-Z]{36})\b'),
        "SSN": re.compile(r'\b(\d{3}-\d{2}-\d{4})\b'),
        "CREDIT_CARD": re.compile(r'\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b'),
        "EMAIL": re.compile(r'\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b'),
        "PHONE": re.compile(r'\b(\+?1?[\s\-.]?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4})\b'),
        "IP_ADDRESS": re.compile(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'),
        "IBAN": re.compile(r'\b([A-Z]{2}\d{2}[A-Z0-9]{4,})\b'),
        #new pattern. Catches secrets by ASSIGNMENT CONTEXT
        
        # with punctuation was previously invisible to the scanner.
        "SECRET_ASSIGNMENT": re.compile(
            r'(?:password|secret|api[_-]?key|token|pwd)[\'"]?\s*[:=]\s*'
            r'[\'"]([^\'"]{6,})[\'"]',
            re.IGNORECASE,
        ),
        "API_KEY": re.compile(r'\b([A-Za-z0-9_\-]{20,64})\b'),
    }

    def scan(self, text: str) -> list[Finding]:
        if not text:
            return []

        findings = []
        claimed_spans = []  

        for entity_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group(1)
                position = match.start(1)
                span = (position, position + len(value))

                if self._overlaps(span, claimed_spans):
                    continue

                confidence, validation_passed, masked = self._validate(
                    entity_type, value, text, position
                )

                
                confidence = min(confidence, 1.0)

                if confidence < 0.3:
                    continue

                start = max(0, position - 30)
                end = min(len(text), position + len(value) + 30)
                snippet = text[start:end].replace('\n', ' ')

                findings.append(Finding(
                    entity_type=entity_type,
                    masked_value=masked,
                    char_position=position,
                    confidence=round(confidence, 2),
                    context_snippet=snippet,
                    validation_passed=validation_passed,
                ))
                claimed_spans.append(span)

        logger.info(f"scan complete -- {len(findings)} findings")
        return findings

    @staticmethod
    def _overlaps(span: tuple, claimed_spans: list) -> bool:
        s, e = span
        for cs, ce in claimed_spans:
            if s < ce and e > cs:
                return True
        return False

    def _validate(
        self, entity_type: str, value: str, text: str, position: int
    ) -> tuple[float, bool, str]:
        """
        Returns (confidence, validation_passed, masked_value) for a match.

        NOTE on validation_passed: for SSN, EMAIL, STRIPE_KEY, GITHUB_TOKEN
        this is always True -- there's no independent checksum for these,
        so "validated" really means "the format itself is specific enough
        that a regex match IS the validation" (a Stripe/GitHub prefix is
        about as good as a checksum). For CREDIT_CARD, PHONE, AWS_KEY,
        API_KEY, and IBAN, validation_passed reflects a real, independent
        check (Luhn / phonenumbers / entropy / MOD-97).
        """
        boost = context_score_boost(text, position)

        if entity_type == "SSN":
            return (0.75 + boost, True, mask_ssn(value))

        elif entity_type == "CREDIT_CARD":
            passed = luhn_check(value)
            confidence = (0.90 + boost) if passed else 0.1
            return (confidence, passed, mask_credit_card(value))

        elif entity_type == "EMAIL":
            return (0.85 + boost, True, mask_default(value))

        elif entity_type == "PHONE":
            passed = validate_phone(value)
            confidence = (0.80 + boost) if passed else 0.25
            return (confidence, passed, mask_default(value))

        elif entity_type == "AWS_KEY":
            entropy = shannon_entropy(value)
            passed = entropy > 3.5
            confidence = (0.95 + boost) if passed else 0.4
            return (confidence, passed, mask_default(value))

        elif entity_type in ("STRIPE_KEY", "GITHUB_TOKEN"):
            return (0.95 + boost, True, mask_default(value))

        elif entity_type == "SECRET_ASSIGNMENT":
            entropy = shannon_entropy(value)
            passed = entropy > 2.5
            confidence = (0.75 + boost) if passed else 0.35
            return (confidence, passed, mask_default(value))

        elif entity_type == "API_KEY":
            entropy = shannon_entropy(value)
            passed = entropy > 3.5
            confidence = (0.65 + boost) if passed else 0.1
            return (confidence, passed, mask_default(value))

        elif entity_type == "IP_ADDRESS":
            parts = value.split('.')
            passed = all(0 <= int(p) <= 255 for p in parts)
            confidence = (0.60 + boost) if passed else 0.1
            return (confidence, passed, mask_default(value))

        elif entity_type == "IBAN":
            passed = validate_iban(value)
            confidence = (0.80 + boost) if passed else 0.15
            return (confidence, passed, mask_default(value))

        return (0.5 + boost, False, mask_default(value))