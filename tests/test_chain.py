import hashlib
import tempfile

from integrity.chain.snapshot_record import SnapshotRecord
from integrity.chain.signer import SnapshotSigner
from integrity.chain.verifier import ChainVerifier
from integrity.merkle.merkle_tree import MerkleTree


def fake_hash(label):
    return hashlib.sha256(label.encode()).hexdigest()


def make_files(labels):
    return sorted(
        [{"path": f"/data/{l}.txt", "hash": fake_hash(l)} for l in labels],
        key=lambda f: f["path"],
    )


def build_chain(signer, snapshots):
    records = []
    prev = None
    for i, labels in enumerate(snapshots):
        files = make_files(labels)
        tree = MerkleTree().build([f["hash"] for f in files])
        record = SnapshotRecord(
            snapshot_id=i,
            merkle_root=tree.root_hash,
            files=files,
            prev_record_hash=prev.record_hash() if prev else None,
        )
        signer.sign(record)
        records.append(record)
        prev = record
    return records


def fresh_signer():
    return SnapshotSigner.generate(tempfile.mkdtemp())


SNAPS = [["a", "b"], ["a", "b", "c"], ["a", "b", "c", "d"], ["a", "c", "d"], ["a", "c", "d", "e"]]


def test_clean_chain_verifies():
    signer = fresh_signer()
    report = ChainVerifier(signer).verify_chain(build_chain(signer, SNAPS))
    assert report["clean"], report
    assert report["first_break"] is None


def test_signature_is_not_deterministic_but_still_verifies():
    signer = fresh_signer()
    files = make_files(["a", "b"])
    root = MerkleTree().build([f["hash"] for f in files]).root_hash

    r1 = SnapshotRecord(0, root, files)
    r2 = SnapshotRecord(0, root, files, created_at=r1.created_at)
    signer.sign(r1)
    signer.sign(r2)

    assert r1.signature != r2.signature
    assert signer.verify(r1) and signer.verify(r2)


def test_tampered_file_list_breaks_merkle_and_signature():
    signer = fresh_signer()
    records = build_chain(signer, SNAPS)
    records[2].files[0]["hash"] = fake_hash("evil")

    report = ChainVerifier(signer).verify_chain(records)
    assert report["first_break"] == 2
    assert not report["records"][2]["merkle_valid"]
    assert not report["records"][2]["signature_valid"]


def test_tampered_timestamp_breaks_signature_only():
    signer = fresh_signer()
    records = build_chain(signer, SNAPS)
    records[1].created_at = "2020-01-01T00:00:00+00:00"

    report = ChainVerifier(signer).verify_chain(records)
    assert not report["records"][1]["signature_valid"]
    assert report["records"][1]["merkle_valid"]


def test_stripped_signature_breaks_the_link():
    signer = fresh_signer()
    records = build_chain(signer, SNAPS)
    records[1].signature = None

    report = ChainVerifier(signer).verify_chain(records)
    assert not report["records"][1]["signature_valid"]
    assert not report["records"][2]["link_valid"]


def test_deleted_record_breaks_the_link():
    signer = fresh_signer()
    records = build_chain(signer, SNAPS)
    del records[2]

    report = ChainVerifier(signer).verify_chain(records)
    assert not report["clean"]
    assert report["first_break"] == 3


def test_wrong_key_fails_everything():
    signer = fresh_signer()
    records = build_chain(signer, SNAPS)

    report = ChainVerifier(fresh_signer()).verify_chain(records)
    assert all(not r["signature_valid"] for r in report["records"])


def test_verifier_needs_only_public_key():
    key_dir = tempfile.mkdtemp()
    signer = SnapshotSigner.generate(key_dir)
    records = build_chain(signer, SNAPS)

    public_only = SnapshotSigner.load(key_dir, private=False)
    assert public_only.private_key is None
    assert ChainVerifier(public_only).verify_chain(records)["clean"]


def test_canonical_serialization_is_stable():
    files = make_files(["a", "b"])
    root = MerkleTree().build([f["hash"] for f in files]).root_hash
    r = SnapshotRecord(0, root, files)
    assert r.record_hash() == SnapshotRecord.from_dict(r.to_dict()).record_hash()


def test_changed_files_pinpoints_drift():
    files = make_files(["a", "b", "c"])
    root = MerkleTree().build([f["hash"] for f in files]).root_hash
    record = SnapshotRecord(0, root, files)

    current = {f["path"]: f["hash"] for f in files}
    current["/data/b.txt"] = fake_hash("b-edited")
    del current["/data/c.txt"]
    current["/data/z.txt"] = fake_hash("z")

    changes = {c["path"]: c["status"] for c in ChainVerifier.changed_files(record, current)}
    assert changes == {
        "/data/b.txt": "modified",
        "/data/c.txt": "missing",
        "/data/z.txt": "added",
    }


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")