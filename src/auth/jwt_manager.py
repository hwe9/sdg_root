# /sdg_root/src/auth/jwt_manager.py
import os
import jwt
import secrets
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from src.core.secrets_manager import secrets_manager

logger = logging.getLogger(__name__)

class JWTManager:
    def __init__(self):
        self.algorithm = "RS256"  # Use RSA instead of HMAC for better security
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        self.refresh_token_family = {}
        self.private_key, self.public_key = self._load_or_generate_keys()
        self.issuer = os.getenv("AUTH_ISSUER", "sdg-auth")
        self.audience = os.getenv("AUTH_AUDIENCE", "sdg-clients")
    
    def _load_or_generate_keys(self):
        """Load existing RSA key pair or generate new ones"""
        try:
            # Try to load existing keys from secrets manager
            private_key_pem = secrets_manager.get_secret("JWT_PRIVATE_KEY")
            public_key_pem = secrets_manager.get_secret("JWT_PUBLIC_KEY")
            
            private_key = serialization.load_pem_private_key(
                private_key_pem.encode(),
                password=None
            )
            public_key = serialization.load_pem_public_key(public_key_pem.encode())
            
            logger.info("Loaded existing JWT key pair")
            return private_key, public_key
            
        except Exception as e:
            logger.warning(f"Could not load existing keys: {e}. Generating new ones.")
            return self._generate_new_keys()
    
    def _generate_new_keys(self):
        """Generate new RSA key pair"""
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Get public key
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode()
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
            
            # Store in environment variables (encrypted)
            if os.environ.get('ENVIRONMENT') == 'production':
                # In production, these should be stored securely
                pub_fp = hashlib.sha256(public_pem.encode()).hexdigest()[:16]
                logger.info(f"Generated new JWT key pair (public fp={pub_fp}); store securely via secrets manager")
            
            logger.info("Generated new JWT key pair")
            return private_key, public_key
            
        except Exception as e:
            logger.error(f"Error generating JWT keys: {e}")
            raise
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        try:
            now = datetime.utcnow()
            exp = now + timedelta(minutes=self.access_token_expire_minutes)
            payload = {
                **data,
                "iss": self.issuer,
                "aud": self.audience,
                "nbf": now,
                "iat": now,
                "exp": exp,
                "type": "access",
                "jti": secrets.token_urlsafe(16)
            }
            token = jwt.encode(payload, self.private_key, algorithm=self.algorithm)
            return token if isinstance(token, str) else token.decode()
            
        except Exception as e:
            logger.error(f"Error creating access token: {e}")
            raise
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        try:
            now = datetime.utcnow()
            exp = now + timedelta(days=self.refresh_token_expire_days)
            payload = {
                **data,
                "iss": self.issuer,
                "aud": self.audience,
                "nbf": now,
                "iat": now,
                "exp": exp,
                "type": "refresh",
                "jti": secrets.token_urlsafe(24)
            }
            return jwt.encode(payload, self.private_key, algorithm=self.algorithm)
            
        except Exception as e:
            logger.error(f"Error creating refresh token: {e}")
            raise
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat", "nbf", "type", "jti", "iss", "aud"]}
            )
            if payload.get("type") != token_type:
                raise jwt.InvalidTokenError(f"Expected {token_type} token")
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired"); raise
        except (jwt.InvalidIssuerError, jwt.InvalidAudienceError, jwt.InvalidTokenError) as e:
            logger.warning(f"Invalid token: {e}"); raise

# Global JWT manager
jwt_manager = JWTManager()
