import hashlib
import json
from pathlib import Path

from integrity.chain.snapshot_record import SnapshotRecord
from integrity.merkle.merkle_tree import MerkleTree


class SnapshotBuilder:

    def __init__(self, signer, chain_path):
        self.signer = signer
        self.chain_path = Path(chain_path)

    def scan(self, directory):
        files = []
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file():
                files.append({"path": str(path).replace("\\", "/"), "hash": self._hash_file(path)})
        return files

    def create_snapshot(self, directory):
        chain = self.load_chain()
        files = self.scan(directory)
        tree = MerkleTree().build([f["hash"] for f in files])

        record = SnapshotRecord(
            snapshot_id=len(chain),
            merkle_root=tree.root_hash,
            files=files,
            prev_record_hash=chain[-1].record_hash() if chain else None,
        )
        self.signer.sign(record)

        chain.append(record)
        self.save_chain(chain)
        return record

    def load_chain(self):
        if not self.chain_path.exists():
            return []
        raw = json.loads(self.chain_path.read_text(encoding="utf-8"))
        return [SnapshotRecord.from_dict(r) for r in raw]

    def save_chain(self, chain):
        self.chain_path.parent.mkdir(parents=True, exist_ok=True)
        self.chain_path.write_text(
            json.dumps([r.to_dict() for r in chain], indent=2), encoding="utf-8"
        )

    @staticmethod
    def _hash_file(path):
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()