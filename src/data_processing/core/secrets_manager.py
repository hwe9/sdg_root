# /sdg_root/src/core/secrets_manager.py
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
import logging

logger = logging.getLogger(__name__)

class SecretsManager:
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self):
        """Get encryption key from secure storage or create new one"""
        try:
            # Try to get key from system keyring
            key = keyring.get_password("sdg_pipeline", "encryption_key")
            if key:
                return key.encode()
            
            # Generate new key if none exists
            password = os.environ.get('MASTER_PASSWORD', 'default_dev_password').encode()
            salt = os.environ.get('ENCRYPTION_SALT', 'default_salt').encode()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            
            # Store in system keyring for production
            if os.environ.get('ENVIRONMENT') == 'production':
                keyring.set_password("sdg_pipeline", "encryption_key", key.decode())
            
            return key
        except Exception as e:
            logger.error(f"Error managing encryption key: {e}")
            raise
    
    def encrypt_secret(self, plaintext: str) -> str:
        """Encrypt a secret value"""
        return self.cipher.encrypt(plaintext.encode()).decode()
    
    def decrypt_secret(self, encrypted: str) -> str:
        """Decrypt a secret value"""
        return self.cipher.decrypt(encrypted.encode()).decode()
    
    def get_secret(self, key: str) -> str:
        """Get decrypted secret from environment"""
        encrypted_value = os.environ.get(f"{key}_ENCRYPTED")
        if encrypted_value:
            return self.decrypt_secret(encrypted_value)
        
        # Fallback to plaintext for development (with warning)
        plaintext_value = os.environ.get(key)
        if plaintext_value and os.environ.get('ENVIRONMENT') != 'production':
            logger.warning(f"Using plaintext secret for {key} in development")
            return plaintext_value
        
        raise ValueError(f"Secret {key} not found or not properly encrypted")

# Initialize global secrets manager
secrets_manager = SecretsManager()
