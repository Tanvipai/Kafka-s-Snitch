from pathlib import Path
from discovery.ingestion.file_ingester import FileIngester
from discovery.scanner.pattern_scanner import PatternScanner


def run():
    ingester = FileIngester()
    scanner = PatternScanner()
    samples = list(Path("data/synthetic/samples").iterdir())

    print(f"\nscanning {len(samples)} files\n")

    total_findings = 0
    skipped = []  #track and report skipped files instead of silently continuing
                 

    for file_path in samples[:10]:  
        result = ingester.ingest(str(file_path))
        if result["error"] or not result["text"]:
            skipped.append((result["file_name"], result["error"]))
            continue

        findings = scanner.scan(result["text"])
        total_findings += len(findings)

        print(f"{result['file_name']} — {len(findings)} findings")
        for f in findings:
            print(f"  [{f.entity_type}] {f.masked_value} "
                  f"(confidence={f.confidence}, validated={f.validation_passed})")

    print(f"\ntotal findings across 10 files: {total_findings}")

    if skipped:
        print(f"\n{len(skipped)} file(s) skipped during ingestion:")
        for name, error in skipped:
            print(f"  - {name}: {error}")


if __name__ == "__main__":
    run()