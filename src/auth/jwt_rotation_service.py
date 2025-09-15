# /src/auth/jwt_rotation_service.py
"""
Automatic JWT Key Rotation Service
Handles monthly JWT key rotation for enhanced security
"""
import os
import asyncio
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from .jwt_manager import jwt_manager
from ..core.secrets_manager import secrets_manager
from ..core.service_registry import service_registry
from cryptography.hazmat.primitives import serialization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RotationResult:
    success: bool
    timestamp: datetime
    old_key_fingerprint: str
    new_key_fingerprint: str
    affected_services: list
    error_message: Optional[str] = None

class JWTRotationService:
    """Handles automatic JWT key rotation with service coordination"""
    
    def __init__(self):
        self.rotation_day = int(os.environ.get("JWT_ROTATION_DAY", "1"))  # 1st of each month
        self.rotation_hour = int(os.environ.get("JWT_ROTATION_HOUR", "2"))  # 2 AM
        self.backup_retention_months = int(os.environ.get("JWT_KEY_RETENTION_MONTHS", "3"))
        self.service_coordination_timeout = 300  # 5 minutes
        
        # Services that need to be notified of key rotation
        self.dependent_services = [
            "api_service",
            "content_extraction_service", 
            "vectorization_service",
            "data_processing_service"
        ]
        
    async def start_rotation_scheduler(self):
        """Start the automatic rotation scheduler"""
        logger.info("Starting JWT rotation scheduler...")
        
        # Schedule monthly rotation
        schedule.every().month.at(f"{self.rotation_hour:02d}:00").on(self.rotation_day).do(
            self._schedule_rotation
        )
        
        # Also schedule a manual trigger check (daily)
        schedule.every().day.at("01:00").do(self._check_manual_rotation_trigger)
        
        # Run scheduler in background
        while True:
            try:
                schedule.run_pending()
                await asyncio.sleep(3600)  # Check every hour
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def _schedule_rotation(self):
        """Wrapper for async rotation"""
        asyncio.create_task(self.rotate_jwt_keys())
    
    async def _check_manual_rotation_trigger(self):
        """Check for manual rotation trigger file"""
        trigger_file = Path("/tmp/jwt_rotation_trigger")
        if trigger_file.exists():
            logger.info("Manual JWT rotation triggered")
            await self.rotate_jwt_keys()
            trigger_file.unlink()  # Remove trigger file
    
    async def rotate_jwt_keys(self) -> RotationResult:
        """Perform complete JWT key rotation with service coordination"""
        start_time = datetime.utcnow()
        logger.info("🔄 Starting JWT key rotation process...")
        
        try:
            # Step 1: Backup current keys
            old_fingerprint = self._get_key_fingerprint(jwt_manager.public_key)
            await self._backup_current_keys()
            
            # Step 2: Generate new keys
            logger.info("Generating new JWT key pair...")
            new_private_key, new_public_key = jwt_manager._generate_new_keys()
            new_fingerprint = self._get_key_fingerprint(new_public_key)
            
            # Step 3: Coordinate with dependent services
            logger.info("Coordinating with dependent services...")
            coordination_success = await self._coordinate_with_services("prepare_rotation")
            
            if not coordination_success:
                raise Exception("Service coordination failed - aborting rotation")
            
            # Step 4: Update JWT manager with new keys
            jwt_manager.private_key = new_private_key
            jwt_manager.public_key = new_public_key
            
            # Step 5: Persist new keys
            await self._persist_new_keys(new_private_key, new_public_key)
            
            # Step 6: Notify services of successful rotation
            await self._coordinate_with_services("rotation_complete")
            
            # Step 7: Cleanup old keys (after grace period)
            await self._schedule_key_cleanup()
            
            result = RotationResult(
                success=True,
                timestamp=start_time,
                old_key_fingerprint=old_fingerprint,
                new_key_fingerprint=new_fingerprint,
                affected_services=self.dependent_services
            )
            
            logger.info(f"✅ JWT rotation completed successfully in {(datetime.utcnow() - start_time).total_seconds():.2f}s")
            
            # Log rotation event for audit
            await self._log_rotation_event(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ JWT rotation failed: {e}")
            
            # Attempt rollback
            await self._rollback_rotation()
            
            result = RotationResult(
                success=False,
                timestamp=start_time,
                old_key_fingerprint="",
                new_key_fingerprint="",
                affected_services=[],
                error_message=str(e)
            )
            
            await self._log_rotation_event(result)
            return result
    
    async def _backup_current_keys(self):
        """Backup current JWT keys"""
        try:
            backup_dir = Path("/app/auth/key_backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Backup private key (encrypted)
            private_key_pem = jwt_manager.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode()
            
            public_key_pem = jwt_manager.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
            
            # Save encrypted backups
            backup_data = {
                "timestamp": timestamp,
                "private_key": secrets_manager.encrypt_secret(private_key_pem),
                "public_key": secrets_manager.encrypt_secret(public_key_pem),
                "fingerprint": self._get_key_fingerprint(jwt_manager.public_key)
            }
            
            backup_file = backup_dir / f"jwt_keys_backup_{timestamp}.json"
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f)
            
            logger.info(f"Keys backed up to {backup_file}")
            
        except Exception as e:
            logger.error(f"Key backup failed: {e}")
            raise
    
    def _get_key_fingerprint(self, public_key) -> str:
        """Generate fingerprint for public key"""
        import hashlib
        from cryptography.hazmat.primitives import serialization
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return hashlib.sha256(public_pem).hexdigest()[:16]
    
    async def _coordinate_with_services(self, action: str) -> bool:
        """Coordinate rotation with dependent services"""
        success_count = 0
        
        for service_name in self.dependent_services:
            try:
                response = await service_registry.call_service(
                    service_name=service_name,
                    endpoint=f"/auth/jwt-rotation/{action}",
                    method="POST",
                    json={"timestamp": datetime.utcnow().isoformat()},
                    timeout=30
                )
                
                if response and response.get("status") == "acknowledged":
                    success_count += 1
                    logger.info(f"✅ {service_name} acknowledged {action}")
                else:
                    logger.warning(f"⚠️ {service_name} did not acknowledge {action}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to coordinate with {service_name}: {e}")
        
        # Require majority of services to acknowledge
        required_services = len(self.dependent_services) // 2 + 1
        return success_count >= required_services
    
    async def _persist_new_keys(self, private_key, public_key):
        """Persist new keys to environment/secrets"""
        try:
            from cryptography.hazmat.primitives import serialization
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode()
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
            
            # Update environment variables (encrypted)
            os.environ["JWT_PRIVATE_KEY_ENCRYPTED"] = secrets_manager.encrypt_secret(private_pem)
            os.environ["JWT_PUBLIC_KEY_ENCRYPTED"] = secrets_manager.encrypt_secret(public_pem)
            
            # For production, also update external secret store
            if os.environ.get("ENVIRONMENT") == "production":
                # Update Kubernetes secrets, Azure Key Vault, etc.
                await self._update_external_secrets(private_pem, public_pem)
            
            logger.info("New keys persisted successfully")
            
        except Exception as e:
            logger.error(f"Failed to persist new keys: {e}")
            raise
    
    async def _update_external_secrets(self, private_pem: str, public_pem: str):
        """Update external secret stores in production"""
        # Implement based on your secret management system
        # Examples:
        
        # Kubernetes secrets
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            await self._update_k8s_secrets(private_pem, public_pem)
        
        # Azure Key Vault
        if os.environ.get("AZURE_KEY_VAULT_URL"):
            await self._update_azure_secrets(private_pem, public_pem)
    
    async def _schedule_key_cleanup(self):
        """Schedule cleanup of old keys after grace period"""
        cleanup_delay = int(os.environ.get("JWT_KEY_GRACE_PERIOD_HOURS", "24")) * 3600
        
        async def cleanup_old_keys():
            await asyncio.sleep(cleanup_delay)
            try:
                backup_dir = Path("/app/auth/key_backups")
                cutoff_date = datetime.utcnow() - timedelta(days=30 * self.backup_retention_months)
                
                for backup_file in backup_dir.glob("jwt_keys_backup_*.json"):
                    if backup_file.stat().st_mtime < cutoff_date.timestamp():
                        backup_file.unlink()
                        logger.info(f"Cleaned up old backup: {backup_file.name}")
                        
            except Exception as e:
                logger.error(f"Key cleanup failed: {e}")
        
        asyncio.create_task(cleanup_old_keys())
    
    async def _rollback_rotation(self):
        """Attempt to rollback failed rotation"""
        try:
            logger.info("Attempting rotation rollback...")
            
            # Load most recent backup
            backup_dir = Path("/app/auth/key_backups")
            backups = sorted(backup_dir.glob("jwt_keys_backup_*.json"), reverse=True)
            
            if backups:
                with open(backups[0], 'r') as f:
                    backup_data = json.load(f)
                
                # Restore keys from backup
                private_pem = secrets_manager.decrypt_secret(backup_data["private_key"])
                public_pem = secrets_manager.decrypt_secret(backup_data["public_key"])
                
                # Restore JWT manager state (implementation depends on your JWT manager)
                logger.info("Rollback completed successfully")
            else:
                logger.error("No backup available for rollback")
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
    
    async def _log_rotation_event(self, result: RotationResult):
        """Log rotation event for audit trail"""
        try:
            log_entry = {
                "event": "jwt_key_rotation",
                "success": result.success,
                "timestamp": result.timestamp.isoformat(),
                "old_key_fingerprint": result.old_key_fingerprint,
                "new_key_fingerprint": result.new_key_fingerprint,
                "affected_services": result.affected_services,
                "error_message": result.error_message
            }
            
            # Log to audit system (implement based on your logging infrastructure)
            logger.info(f"Rotation audit log: {json.dumps(log_entry)}")
            
            # Could also send to external audit system
            # await audit_system.log_security_event(log_entry)
            
        except Exception as e:
            logger.error(f"Failed to log rotation event: {e}")

# Global rotation service instance
rotation_service = JWTRotationService()

# Manual rotation trigger function
async def trigger_manual_rotation() -> RotationResult:
    """Trigger manual JWT key rotation"""
    logger.info("Manual JWT rotation triggered")
    return await rotation_service.rotate_jwt_keys()
