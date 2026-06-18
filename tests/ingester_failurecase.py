import os
from pathlib import Path
from discovery.ingestion.file_ingester import FileIngester


def create_fake_files(tmp_dir: Path):
    tmp_dir.mkdir(parents=True, exist_ok=True)

    #1.file that doesn't exist at all
    #2. .txt file that is secretly a PDF inside
    fake_txt = tmp_dir / "totally_normal.txt"
    fake_txt.write_bytes(b"%PDF-1.4 fake pdf content that is malformed")

    #3.completely empty file
    empty_file = tmp_dir / "empty.txt"
    empty_file.write_bytes(b"")

    #4.file with non-UTF-8 encoding (Latin-1)
    latin1_file = tmp_dir / "latin1_encoded.txt"
    latin1_file.write_bytes("Ae Oe Ue - these are German characters in Latin-1".encode("latin-1"))

    #file type we don't support
    fake_zip = tmp_dir / "archive.zip"
    fake_zip.write_bytes(b"PK\x03\x04 this is a zip file")

    #PDF that is completely corrupted
    corrupt_pdf = tmp_dir / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4 \x00\x00\x00 totally broken content !!!")

    #very large text file (stress test)
    large_file = tmp_dir / "large_file.txt"
    large_file.write_text("sensitive data SSN 123-45-6789\n" * 10000, encoding="utf-8")

    return {
        "missing": str(tmp_dir / "ghost_file.txt"),  # never created
        "fake_txt": str(fake_txt),
        "empty": str(empty_file),
        "latin1": str(latin1_file),
        "fake_zip": str(fake_zip),
        "corrupt_pdf": str(corrupt_pdf),
        "large_file": str(large_file),
    }


def run_failure_tests():
    ingester = FileIngester()
    tmp_dir = Path("data/synthetic/failure_samples")

    print("\ncreating intentionally broken files...")
    files = create_fake_files(tmp_dir)

    print(f"\nrunning ingester on {len(files)} edge case files\n")
    print(f"{'file':<25} {'outcome':<10} {'detail'}")
    print("-" * 70)

    for label, path in files.items():
        result = ingester.ingest(path)

        if result["error"]:
            outcome = "HANDLED"
            detail = result["error"]
        else:
            outcome = "OK"
            detail = f"type={result['file_type']} | chars={result['char_count']}"

        print(f"{label:<25} {outcome:<10} {detail}")

    print("\nif every row shows HANDLED or OK, the pipeline is robust.")
    print("a crash (exception not caught) would be the real failure.\n")


if __name__ == "__main__":
    run_failure_tests()