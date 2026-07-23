import hashlib
import json

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"

EMPTY_ROOT = hashlib.sha256(b"").hexdigest()


class MerkleTree:

    def __init__(self):
        self.leaves = []
        self._cache = {}

    def build(self, file_hashes):
        self.leaves = [self._hash_leaf(h) for h in file_hashes]
        self._cache = {}
        return self

    @property
    def leaf_count(self):
        return len(self.leaves)

    @property
    def root_hash(self):
        return self._subtree_root(0, len(self.leaves))

    def get_proof(self, index):
        
        if not 0 <= index < len(self.leaves):
            raise IndexError(f"leaf index {index} out of range (have {len(self.leaves)} leaves)")

        proof = []
        start, end = 0, len(self.leaves)

        while end - start > 1:
            split = start + self._split_point(end - start)
            if index < split:
                proof.append({"hash": self._subtree_root(split, end), "side": "right"})
                end = split
            else:
                proof.append({"hash": self._subtree_root(start, split), "side": "left"})
                start = split

        return list(reversed(proof))

    @staticmethod
    def verify_proof(file_hash, proof, root_hash):
        
        computed = MerkleTree._hash_leaf(file_hash)

        for step in proof:
            if step["side"] == "right":
                computed = MerkleTree._hash_node(computed, step["hash"])
            else:
                computed = MerkleTree._hash_node(step["hash"], computed)

        return computed == root_hash

    def export(self):
        return json.dumps({"root": self.root_hash, "leaves": self.leaves})

    @classmethod
    def from_json(cls, blob):
        data = json.loads(blob)
        tree = cls()
        tree.leaves = data["leaves"]
        return tree

    def _subtree_root(self, start, end):
        if start >= end:
            return EMPTY_ROOT
        if end - start == 1:
            return self.leaves[start]

        if (start, end) in self._cache:
            return self._cache[(start, end)]

        split = start + self._split_point(end - start)
        node = self._hash_node(
            self._subtree_root(start, split),
            self._subtree_root(split, end),
        )
        self._cache[(start, end)] = node
        return node

    @staticmethod
    def _split_point(n):
        return 1 << ((n - 1).bit_length() - 1)

    @staticmethod
    def _hash_leaf(file_hash):
        return hashlib.sha256(LEAF_PREFIX + bytes.fromhex(file_hash)).hexdigest()

    @staticmethod
    def _hash_node(left, right):
        return hashlib.sha256(
            NODE_PREFIX + bytes.fromhex(left) + bytes.fromhex(right)
        ).hexdigest()