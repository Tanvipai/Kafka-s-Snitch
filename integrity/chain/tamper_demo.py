
#Builds five snapshots of a directory, then runs two different attacks and shows that each one is caught by a different check.

import json
import shutil
from pathlib import Path

from integrity.chain.signer import SnapshotSigner
from integrity.chain.snapshot_builder import SnapshotBuilder
from integrity.chain.verifier import ChainVerifier

DEMO_DIR = Path("data/demo_snapshots")
KEY_DIR = Path("keys")
CHAIN_PATH = Path("data/snapshot_chain.json")


def banner(text):
    print(f"\n{text}\n{'-' * len(text)}")


def seed_files():
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)
    for name in ["payroll.csv", "patients.txt", "config.py"]:
        (DEMO_DIR / name).write_text(f"original contents of {name}\n", encoding="utf-8")


def build_five_snapshots(builder):
    banner("building 5 snapshots")
    for i in range(5):
        if i > 0:
            (DEMO_DIR / f"report_{i}.txt").write_text(f"quarterly report {i}\n", encoding="utf-8")
        record = builder.create_snapshot(DEMO_DIR)
        print(f"  snapshot {record.snapshot_id}: {len(record.files)} files, root {record.merkle_root[:16]}...")


def verify(verifier, chain, label):
    banner(label)
    report = verifier.verify_chain(chain)
    for r in report["records"]:
        status = "OK" if r["ok"] else "BROKEN"
        detail = "" if r["ok"] else (
            f"  [sig={r['signature_valid']} link={r['link_valid']} merkle={r['merkle_valid']}]"
        )
        print(f"  snapshot {r['snapshot_id']}: {status}{detail}")

    if report["clean"]:
        print("\n  chain intact")
    else:
        print(f"\n  first break at snapshot {report['first_break']}")
    return report


def attack_the_disk(builder, chain):
    banner("attack 1: a file is edited on disk after the snapshot")
    target = DEMO_DIR / "patients.txt"
    target.write_text("MODIFIED — records exfiltrated\n", encoding="utf-8")

    current = {f["path"]: f["hash"] for f in builder.scan(DEMO_DIR)}
    changes = ChainVerifier.changed_files(chain[-1], current)

    print("  chain itself is untouched, so signatures still verify.")
    print("  comparing the latest snapshot against a fresh scan:")
    for c in changes:
        print(f"    {c['status']:>8}  {c['path']}")


def attack_the_chain(chain_path):
    banner("attack 2: the attacker edits the stored chain to cover their tracks")
    raw = json.loads(chain_path.read_text(encoding="utf-8"))

    victim = raw[2]["files"][0]
    print(f"  rewriting the recorded hash for {victim['path']} in snapshot 2")
    victim["hash"] = "0" * 64

    chain_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def main():
    seed_files()
    if CHAIN_PATH.exists():
        CHAIN_PATH.unlink()

    signer = SnapshotSigner.generate(KEY_DIR)
    builder = SnapshotBuilder(signer, CHAIN_PATH)
    verifier = ChainVerifier(SnapshotSigner.load(KEY_DIR, private=False))

    build_five_snapshots(builder)
    verify(verifier, builder.load_chain(), "baseline verification")

    attack_the_disk(builder, builder.load_chain())

    attack_the_chain(CHAIN_PATH)
    report = verify(verifier, builder.load_chain(), "verification after chain tampering")

    banner("what this proves")
    print("  attack 1 changed data but not the record — caught by comparing a fresh")
    print("  scan against the signed file list.")
    print("  attack 2 changed the record itself — caught three ways at snapshot 2:")
    print("  the merkle root no longer matches its file list, the signature no longer")
    print("  covers the edited payload, and every later record's link is broken.")
    print(f"\n  first break reported at snapshot {report['first_break']}")


if __name__ == "__main__":
    main()