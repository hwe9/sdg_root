# /sdg_root/src/auth/main.py

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import bcrypt
import os
from datetime import datetime, timedelta
import logging
import secrets
import asyncio
import jwt

from .jwt_manager import jwt_manager
from ..core.secrets_manager import secrets_manager
from ..core.dependency_manager import dependency_manager, wait_for_dependencies, setup_sdg_dependencies, get_dependency_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global user database
users_db = None

# Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

def get_allowed_origins():
    """Get allowed CORS origins with validation"""
    origins = os.environ.get("ALLOWED_ORIGINS", "").split(",")
    origins = [origin.strip() for origin in origins if origin.strip()]
    
    if not origins:
        if os.environ.get("ENVIRONMENT") == "development":
            return ["http://localhost:3000", "http://localhost:8080"]
        else:
            raise ValueError("ALLOWED_ORIGINS must be set in production")
    
    for origin in origins:
        if not origin.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid origin format: {origin}")
    
    return origins

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan context manager replacing @app.on_event"""
    global users_db
    
    # === STARTUP ===
    logger.info("🚀 Starting SDG Auth Service...")
    
    try:
        # Setup SDG dependencies
        setup_sdg_dependencies()
        
        # Register auth-specific startup tasks
        async def initialize_auth_dependencies():
            """Initialize authentication dependencies"""
            global users_db
            
            logger.info("🔄 Initializing auth service dependencies...")
            
            # Wait for database to be ready
            await wait_for_dependencies("database")
            
            # Validate secrets manager
            try:
                admin_password = secrets_manager.get_secret("ADMIN_PASSWORD")
                if not admin_password:
                    admin_password = "admin123"  # Default for development
                    logger.warning("Using default admin password for development")
                logger.info("✅ Secrets manager validated")
            except Exception as e:
                logger.error(f"❌ Secrets manager validation failed: {e}")
                raise
            
            # Initialize user database
            users_db = UserDatabase()
            logger.info("✅ User database initialized")
            
            logger.info("✅ Auth service dependencies initialized successfully")
        
        dependency_manager.register_startup_task("auth", initialize_auth_dependencies)
        
        # Register cleanup task for graceful shutdown
        async def cleanup_auth_service():
            """Cleanup auth service resources"""
            global users_db
            logger.info("🔄 Cleaning up auth service...")
            users_db = None
            logger.info("✅ Auth service cleanup completed")
        
        dependency_manager.register_cleanup_task("auth", cleanup_auth_service)
        
        # Start dependency manager if not already started
        if not dependency_manager._startup_complete.is_set():
            await dependency_manager.start_all_services()
        else:
            # If dependency manager already started, just run our auth initialization
            await initialize_auth_dependencies()
        
        logger.info("✅ SDG Auth Service startup completed")
        
    except Exception as e:
        logger.error(f"❌ Failed to start SDG Auth Service: {e}")
        raise
    
    # === YIELD TO APPLICATION ===
    yield
    
    # === SHUTDOWN ===
    logger.info("🔄 Shutting down SDG Auth Service...")
    try:
        # Only shutdown dependency manager if we're the last service
        if dependency_manager._startup_complete.is_set():
            await dependency_manager.shutdown_all_services()
        logger.info("✅ SDG Auth Service shutdown completed")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="SDG Auth Service",
    description="Authentication and authorization service for SDG AI Pipeline",
    version="2.0.0",
    lifespan=lifespan  # Modern lifespan instead of @app.on_event
)

# CORS middleware
allowed_origins = get_allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"], 
    allow_headers=["Authorization", "Content-Type"],
)

security = HTTPBearer()

# Pydantic models
class UserLogin(BaseModel):
    username: str
    password: str

    class Config:
        str_strip_whitespace = True
        min_anystr_length = 1
        max_anystr_length = 100

class UserCreate(BaseModel):
    username: str
    password: str
    email: str
    role: str = "user"

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class UserDatabase:
    """User database with dependency-aware initialization"""
    
    def __init__(self):
        self.users = {}
        self._initialize_admin_user()
    
    def _initialize_admin_user(self):
        """Initialize admin user with proper secret management"""
        try:
            admin_password = secrets_manager.get_secret("ADMIN_PASSWORD")
            if not admin_password:
                admin_password = "admin123"  # Development fallback
                logger.warning("Using default admin password for development")
            
            self.users["hwe"] = {
                "username": "hwe",
                "hashed_password": bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt()).decode(),
                "email": "heinrich.ekam@gmail.com",
                "role": "admin",
                "is_active": True,
                "created_at": datetime.utcnow(),
                "last_login": None
            }
            logger.info("✅ Admin user initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize admin user: {e}")
            raise

# Utility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    try:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Hash a password for storage"""
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user with dependency checking"""
    if not users_db:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not ready"
        )
    
    user = users_db.users.get(username)
    if user and user.get("is_active") and verify_password(password, user["hashed_password"]):
        # Update last login
        user["last_login"] = datetime.utcnow()
        return user
    return None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Dependency injection helpers
async def ensure_auth_ready():
    """Ensure auth service dependencies are ready"""
    await wait_for_dependencies("database")
    if not users_db:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not fully initialized"
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token with dependency checking"""
    await ensure_auth_ready()
    
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt_manager.verify_token(token, "access")
        
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        user = users_db.users.get(username)
        if user is None or not user.get("is_active"):
            raise credentials_exception
        
        return user
        
    except jwt.PyJWTError:
        raise credentials_exception

async def get_admin_user(current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Ensure current user has admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

# API Endpoints
@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Secure user authentication with dependency management"""
    await ensure_auth_ready()
    
    user = authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        # Add delay to prevent timing attacks
        await asyncio.sleep(1)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Create tokens using JWT manager
    token_data = {"sub": user["username"], "role": user["role"]}
    access_token = jwt_manager.create_access_token(token_data)
    refresh_token = jwt_manager.create_refresh_token(token_data)
    
    logger.info(f"User {user['username']} logged in successfully")
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": jwt_manager.access_token_expire_minutes * 60
    }

@app.post("/auth/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token"""
    await ensure_auth_ready()
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    
    try:
        payload = jwt_manager.verify_token(refresh_token, "refresh")
        username: str = payload.get("sub")
        
        if username is None:
            raise credentials_exception
            
        user = users_db.users.get(username)
        if user is None:
            raise credentials_exception
        
        # Create new tokens
        token_data = {"sub": user["username"], "role": user["role"]}
        new_access_token = jwt_manager.create_access_token(token_data)
        new_refresh_token = jwt_manager.create_refresh_token(token_data)
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": jwt_manager.access_token_expire_minutes * 60
        }
        
    except jwt.PyJWTError:
        raise credentials_exception

@app.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "role": current_user["role"],
        "is_active": current_user["is_active"],
        "last_login": current_user.get("last_login")
    }

@app.post("/auth/register", response_model=dict)
async def register_user(user: UserCreate, current_user: dict = Depends(get_admin_user)):
    """Register new user (admin only)"""
    await ensure_auth_ready()
    
    if user.username in users_db.users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Hash password properly
    hashed_password = get_password_hash(user.password)
    
    users_db.users[user.username] = {
        "username": user.username,
        "hashed_password": hashed_password,
        "email": user.email,
        "role": user.role,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "last_login": None
    }
    
    logger.info(f"New user registered: {user.username} with role {user.role}")
    return {"message": f"User {user.username} created successfully"}

@app.get("/health")
async def health_check():
    """Enhanced health check with dependency status"""
    try:
        dependency_status = await get_dependency_status()
        
        # Check if auth service is properly initialized
        auth_ready = users_db is not None
        jwt_ready = jwt_manager is not None
        
        overall_status = "healthy" if (auth_ready and jwt_ready) else "starting"
        
        return {
            "status": overall_status,
            "service": "SDG Auth Service",
            "version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "auth_ready": auth_ready,
            "dependencies": dependency_status,
            "components": {
                "jwt_manager": "ready" if jwt_ready else "initializing",
                "secrets_manager": "configured",
                "user_database": "ready" if auth_ready else "initializing",
                "dependency_manager": "active" if dependency_manager._startup_complete.is_set() else "starting"
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "service": "SDG Auth Service", 
            "version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@app.get("/auth/status")
async def get_auth_status(current_user: dict = Depends(get_admin_user)):
    """Get detailed auth service status (admin only)"""
    dependency_status = await get_dependency_status()
    
    return {
        "service_name": "auth",
        "status": dependency_status,
        "user_count": len(users_db.users) if users_db else 0,
        "active_users": sum(1 for user in users_db.users.values() if user.get("is_active")) if users_db else 0,
        "startup_complete": dependency_manager._startup_complete.is_set(),
        "last_check": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)
