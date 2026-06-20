from pathlib import Path
from discovery.ingestion.file_ingester import FileIngester
from discovery.scanner.pattern_scanner import PatternScanner
from discovery.classifier.cascade_classifier import CascadeClassifier


EXPECTED_BY_PREFIX = {
    "hr_employees": "hr_data",
    "patient_note": "medical_phi",
    "config_script": "credentials",
    "invoice": "financial",
}


AMBIGUOUS_PREFIXES = ("ambiguous_dual", "ambiguous_vague")


def expected_category(filename: str) -> str:
    for prefix, category in EXPECTED_BY_PREFIX.items():
        if filename.startswith(prefix):
            return category
    return "unknown"


def is_ambiguous(filename: str) -> bool:
    return filename.startswith(AMBIGUOUS_PREFIXES)


def run():
    ingester = FileIngester()
    scanner = PatternScanner()
    classifier = CascadeClassifier()
    samples = list(Path("data/synthetic/samples").iterdir())

    print(f"\nclassifying {len(samples)} files (tier 1 only -- no spaCy/BART/Ollama wired in yet)\n")

    correct = 0
    wrong = []
    skipped = []
    escalated = 0
    ambiguous_escalated = 0
    ambiguous_total = 0

    for file_path in samples:
        result = ingester.ingest(str(file_path))
        if result["error"] or not result["text"]:
            skipped.append((file_path.name, result["error"]))
            continue

        findings = scanner.scan(result["text"])
        try:
            classification = classifier.classify(result["text"], findings)
        except ModuleNotFoundError as e:
            
            if is_ambiguous(file_path.name):
                ambiguous_total += 1
                ambiguous_escalated += 1
                print(f"  ESCALATED (tier2 unavailable here) {file_path.name}")
            else:
                print(f"  UNEXPECTED ESCALATION {file_path.name}: {e}")
            continue

        if is_ambiguous(file_path.name):
            ambiguous_total += 1
            escalated_ok = classification.decided_by_tier > 1
            ambiguous_escalated += escalated_ok
            marker = "ESCALATED" if escalated_ok else "DECIDED-AT-T1 (unexpected)"
            print(
                f"  {marker:<24} {file_path.name:<25} "
                f"-> {classification.category:<12} "
                f"(tier={classification.decided_by_tier} conf={classification.confidence} "
                f"scores={classification.tier1_scores})"
            )
            continue

        expected = expected_category(file_path.name)
        is_correct = classification.category == expected
        correct += is_correct
        if not is_correct:
            wrong.append((file_path.name, expected, classification))
        if classification.decided_by_tier > 1:
            escalated += 1

        marker = "OK" if is_correct else "MISS"
        print(
            f"  {marker:<4} {file_path.name:<25} "
            f"-> {classification.category:<12} "
            f"(expected={expected:<12} tier={classification.decided_by_tier} "
            f"conf={classification.confidence} risk={classification.risk_tier})"
        )

    total_scored = len(samples) - len(skipped) - ambiguous_total
    print(f"\n{correct}/{total_scored} correct (tier 1 decided {total_scored - escalated}, escalated {escalated})")
    if ambiguous_total:
        print(f"ambiguous files: {ambiguous_escalated}/{ambiguous_total} correctly escalated past tier 1")

    if wrong:
        print(f"\n{len(wrong)} misclassified:")
        for name, expected, classification in wrong:
            print(f"  - {name}: expected {expected}, got {classification.category} "
                  f"(scores={classification.tier1_scores})")

    if skipped:
        print(f"\n{len(skipped)} file(s) skipped during ingestion:")
        for name, error in skipped:
            print(f"  - {name}: {error}")


if __name__ == "__main__":
    run()