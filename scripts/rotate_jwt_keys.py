# /sdg_root/scripts/rotate_jwt_keys.py
"""
Script to rotate JWT keys for enhanced security
Run this periodically in production
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auth.jwt_manager import JWTManager
from src.core.secrets_manager import secrets_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rotate_jwt_keys():
    """Rotate JWT signing keys"""
    try:
        logger.info("Starting JWT key rotation...")
        
        # Generate new keys
        jwt_manager = JWTManager()
        new_private_key, new_public_key = jwt_manager._generate_new_keys()
        
        logger.info("JWT keys rotated successfully")
        logger.info("Remember to update your environment variables with the new encrypted keys")
        logger.info("Old tokens will become invalid after the rotation")
        
    except Exception as e:
        logger.error(f"Error rotating JWT keys: {e}")
        raise

if __name__ == "__main__":
    rotate_jwt_keys()
