# src/core/secrets_manager.py
import os
import base64
import logging
from typing import Optional
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

class SecretsManager:
    def __init__(self):
        self._fernet = self._build_fernet()

    def _build_fernet(self) -> Optional[Fernet]:
        try:
            secret = os.environ.get("SECRET_KEY")
            salt = os.environ.get("ENCRYPTION_SALT", "default_salt").encode("utf-8")
            if not secret:
                logger.warning("SECRET_KEY not set; encrypted secrets will not be usable")
                return None
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=390000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(secret.encode("utf-8")))
            return Fernet(key)
        except Exception as e:
            logger.error(f"Failed to init Fernet: {e}")
            return None

    def encrypt_secret(self, plaintext: str) -> str:
        if not self._fernet:
            raise RuntimeError("Fernet not initialized; cannot encrypt")
        token = self._fernet.encrypt(plaintext.encode("utf-8"))
        return token.decode("utf-8")

    def decrypt_secret(self, token: str) -> str:
        if not self._fernet:
            raise RuntimeError("Fernet not initialized; cannot decrypt")
        plaintext = self._fernet.decrypt(token.encode("utf-8"))
        return plaintext.decode("utf-8")

    def get_secret(self, key: str) -> Optional[str]:
        enc = os.environ.get(f"{key}_ENCRYPTED")
        if enc:
            try:
                return self.decrypt_secret(enc)
            except Exception as e:
                logger.error(f"Decrypt failed for {key}_ENCRYPTED: {e}")
                return None
        return os.environ.get(key)

# global instance
secrets_manager = SecretsManager()
