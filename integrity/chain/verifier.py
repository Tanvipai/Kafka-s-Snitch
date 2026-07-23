from integrity.merkle.merkle_tree import MerkleTree


class ChainVerifier:
    
    def __init__(self, signer):
        self.signer = signer

    def verify_chain(self, records):
        report = {"clean": True, "records": [], "first_break": None}
        previous = None

        for record in records:
            checks = {
                "snapshot_id": record.snapshot_id,
                "signature_valid": self.signer.verify(record),
                "link_valid": self._check_link(record, previous),
                "merkle_valid": self._check_merkle(record),
            }
            checks["ok"] = all(
                checks[k] for k in ("signature_valid", "link_valid", "merkle_valid")
            )

            if not checks["ok"]:
                report["clean"] = False
                if report["first_break"] is None:
                    report["first_break"] = record.snapshot_id

            report["records"].append(checks)
            previous = record

        return report

    def _check_link(self, record, previous):
        if previous is None:
            return record.prev_record_hash is None
        return record.prev_record_hash == previous.record_hash()

    def _check_merkle(self, record):
        rebuilt = MerkleTree().build([f["hash"] for f in record.files])
        return rebuilt.root_hash == record.merkle_root

    @staticmethod
    def changed_files(record, current_hashes):
        
        changes = []
        for entry in record.files:
            now = current_hashes.get(entry["path"])
            if now is None:
                changes.append({"path": entry["path"], "status": "missing"})
            elif now != entry["hash"]:
                changes.append({"path": entry["path"], "status": "modified"})

        known = {f["path"] for f in record.files}
        for path in current_hashes:
            if path not in known:
                changes.append({"path": path, "status": "added"})

        return changes