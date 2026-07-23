import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


def canonical(payload):

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class SnapshotRecord:
    snapshot_id: int
    merkle_root: str
    files: list                     
    prev_record_hash: str = None     
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signature: str = None            
    def signing_payload(self):
        
        payload = asdict(self)
        payload.pop("signature")
        return canonical(payload)

    def record_hash(self):
        
        return hashlib.sha256(canonical(asdict(self))).hexdigest()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)