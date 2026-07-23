import hashlib
from integrity.merkle.merkle_tree import MerkleTree


def fake_hash(label):
    return hashlib.sha256(label.encode()).hexdigest()


def test_root_is_deterministic():
    files = [fake_hash(c) for c in "ABCD"]
    assert MerkleTree().build(files).root_hash == MerkleTree().build(files).root_hash


def test_tampering_changes_root():
    clean = [fake_hash(c) for c in "ABCD"]
    tampered = clean[:2] + [fake_hash("C-modified")] + clean[3:]
    assert MerkleTree().build(clean).root_hash != MerkleTree().build(tampered).root_hash


def test_order_changes_root():
    files = [fake_hash(c) for c in "ABCD"]
    assert MerkleTree().build(files).root_hash != MerkleTree().build(list(reversed(files))).root_hash


def test_valid_proof_verifies():
    files = [fake_hash(c) for c in "ABCDEFG"]
    tree = MerkleTree().build(files)
    for i, f in enumerate(files):
        assert MerkleTree.verify_proof(f, tree.get_proof(i), tree.root_hash), f"leaf {i} failed"


def test_proof_for_absent_file_fails():
    files = [fake_hash(c) for c in "ABCD"]
    tree = MerkleTree().build(files)
    assert not MerkleTree.verify_proof(fake_hash("Z"), tree.get_proof(2), tree.root_hash)


def test_corrupted_proof_fails():
    files = [fake_hash(c) for c in "ABCD"]
    tree = MerkleTree().build(files)
    proof = tree.get_proof(1)
    proof[0]["hash"] = fake_hash("garbage")
    assert not MerkleTree.verify_proof(files[1], proof, tree.root_hash)


def test_flipped_sibling_side_fails():
    files = [fake_hash(c) for c in "ABCD"]
    tree = MerkleTree().build(files)
    proof = tree.get_proof(1)
    proof[0]["side"] = "right" if proof[0]["side"] == "left" else "left"
    assert not MerkleTree.verify_proof(files[1], proof, tree.root_hash)


def test_odd_leaf_count_is_not_forgeable():
    three = [fake_hash(c) for c in "ABC"]
    four = three + [fake_hash("C")]
    assert MerkleTree().build(three).root_hash != MerkleTree().build(four).root_hash


def test_single_leaf():
    only = [fake_hash("A")]
    tree = MerkleTree().build(only)
    assert tree.get_proof(0) == []
    assert MerkleTree.verify_proof(only[0], [], tree.root_hash)


def test_empty_tree():
    tree = MerkleTree().build([])
    assert tree.root_hash == hashlib.sha256(b"").hexdigest()
    assert tree.leaf_count == 0


def test_proof_length_is_logarithmic():
    files = [fake_hash(str(i)) for i in range(1000)]
    tree = MerkleTree().build(files)
    assert len(tree.get_proof(500)) == 10


def test_export_roundtrip():
    files = [fake_hash(c) for c in "ABCDE"]
    tree = MerkleTree().build(files)
    restored = MerkleTree.from_json(tree.export())
    assert restored.root_hash == tree.root_hash
    assert MerkleTree.verify_proof(files[3], restored.get_proof(3), restored.root_hash)


def test_index_out_of_range():
    tree = MerkleTree().build([fake_hash(c) for c in "ABC"])
    for bad in (-1, 3, 99):
        try:
            tree.get_proof(bad)
            assert False, f"expected IndexError for {bad}"
        except IndexError:
            pass


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