from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature


PSS_PADDING = padding.PSS(
    mgf=padding.MGF1(hashes.SHA256()),
    salt_length=padding.PSS.MAX_LENGTH,
)


class SnapshotSigner:
    
    def __init__(self, private_key=None, public_key=None):
        self.private_key = private_key
        self.public_key = public_key

    @classmethod
    def generate(cls, key_dir, key_size=2048):
        key_dir = Path(key_dir)
        key_dir.mkdir(parents=True, exist_ok=True)

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)

        (key_dir / "snapshot_private.pem").write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        (key_dir / "snapshot_public.pem").write_bytes(
            private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

        return cls(private_key, private_key.public_key())

    @classmethod
    def load(cls, key_dir, private=True):
        key_dir = Path(key_dir)

        public_key = serialization.load_pem_public_key(
            (key_dir / "snapshot_public.pem").read_bytes()
        )
        private_key = None
        if private:
            private_key = serialization.load_pem_private_key(
                (key_dir / "snapshot_private.pem").read_bytes(), password=None
            )

        return cls(private_key, public_key)

    def sign(self, record):
        if self.private_key is None:
            raise ValueError("no private key loaded — this signer can only verify")

        signature = self.private_key.sign(
            record.signing_payload(), PSS_PADDING, hashes.SHA256()
        )
        record.signature = signature.hex()
        return record

    def verify(self, record):
        if record.signature is None:
            return False
        try:
            self.public_key.verify(
                bytes.fromhex(record.signature),
                record.signing_payload(),
                PSS_PADDING,
                hashes.SHA256(),
            )
            return True
        except (InvalidSignature, ValueError):
            return False