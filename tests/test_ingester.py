from pathlib import Path
from discovery.ingestion.file_ingester import FileIngester


def run_tests():
    ingester = FileIngester()
    samples_dir = Path("data/synthetic/samples")

    files = list(samples_dir.iterdir())
    print(f"\ntesting ingester on {len(files)} files\n")

    passed = 0
    failed = 0

    for file_path in files:
        result = ingester.ingest(str(file_path))

        if result["error"]:
            print(f"  FAIL  {result['file_name']}: {result['error']}")
            failed += 1
        else:
            print(f"  OK    {result['file_name']} | type={result['file_type']} | chars={result['char_count']}")
            passed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(files)} files")


if __name__ == "__main__":
    run_tests()