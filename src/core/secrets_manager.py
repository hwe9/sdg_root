# src/core/secrets_manager.py

import os
import base64
import logging
from typing import Optional
from pathlib import Path
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

class SecretsManager:
    def __init__(self):
        self._fernet = self._build_fernet()
        self._store_dir = Path(os.environ.get("SECRET_STORE_DIR", "/data/secrets"))
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Secret store dir not writable: {e}")

    def _build_fernet(self) -> Optional[Fernet]:
        try:
            secret = os.environ.get("SECRET_KEY")
            salt = os.environ.get("ENCRYPTION_SALT", "default_salt").encode("utf-8")
            if not secret:
                logger.warning("SECRET_KEY not set; encrypted secrets will not be usable")
                return None
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390000)
            key = base64.urlsafe_b64encode(kdf.derive(secret.encode("utf-8")))
            return Fernet(key)
        except Exception as e:
            logger.error(f"Failed to init Fernet: {e}")
            return None

    def encrypt_secret(self, plaintext: str) -> str:
        if not self._fernet:
            raise RuntimeError("Fernet not initialized; cannot encrypt")
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt_secret(self, token: str) -> str:
        if not self._fernet:
            raise RuntimeError("Fernet not initialized; cannot decrypt")
        return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")

    def _file_path(self, key: str) -> Path:
        # Store encrypted payloads; filename normalized
        name = f"{key.upper()}_ENCRYPTED.secret"
        return self._store_dir / name

    def set_secret(self, key: str, value: str, encrypt: bool = True) -> None:
        """Persist secret to file-backed store (encrypted by default) and export to ENV_ENCRYPTED."""
        try:
            payload = self.encrypt_secret(value) if encrypt else value
            # Write to file
            fp = self._file_path(key)
            fp.write_text(payload)
            # Export to env for in-process consumers
            os.environ[f"{key.upper()}_ENCRYPTED"] = payload
            logger.info(f"Persisted secret: {key.upper()} to {fp}")
        except Exception as e:
            logger.error(f"Failed to persist secret {key}: {e}")
            raise

    def get_secret(self, key: str) -> Optional[str]:
        """Read priority: file (encrypted) -> ENV_ENCRYPTED -> ENV (plaintext)."""
        # 1) File-backed encrypted secret
        try:
            fp = self._file_path(key)
            if fp.exists():
                enc = fp.read_text().strip()
                return self.decrypt_secret(enc)
        except Exception as e:
            logger.error(f"Decrypt/read failed for file secret {key}: {e}")

        # 2) Encrypted env
        enc = os.environ.get(f"{key}_ENCRYPTED")
        if enc:
            try:
                return self.decrypt_secret(enc)
            except Exception as e:
                logger.error(f"Decrypt failed for {key}_ENCRYPTED: {e}")

        # 3) Plain env (dev)
        return os.environ.get(key)

# global instance
secrets_manager = SecretsManager()
