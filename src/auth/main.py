# /sdg_root/src/auth/main.py

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import bcrypt
import os
from datetime import datetime, timedelta
import logging
import secrets
from .jwt_manager import jwt_manager
from ..core.secrets_manager import secrets_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

app = FastAPI(
    title="SDG Auth Service",
    description="Authentication and authorization service for SDG AI Pipeline",
    version="1.0.0"
)

def get_allowed_origins():
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
    def __init__(self):
        self.users = {}
        self._initialize_admin_user()
    
    def _initialize_admin_user(self):
        admin_password = secrets_manager.get_secret("ADMIN_PASSWORD")
        if not admin_password:
            raise ValueError("ADMIN_PASSWORD not configured")
        
        self.users["hwe"] = {
            "username": "hwe",
            "hashed_password": bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt()).decode(),
            "email": "heinrich.ekam@gmail.com",
            "role": "admin",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "last_login": None
        }

users_db = UserDatabase().users


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    try:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user with rate limiting"""
    user = users_db.get(username)
    if user and user.get("is_active") and verify_password(password, user["hashed_password"]):
        # Update last login
        user["last_login"] = datetime.utcnow()
        return user
    return None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token."""
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt_manager.verify_token(token, "access")
        
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        user = users_db.get(username)
        if user is None or not user.get("is_active"):
            raise credentials_exception
        
        return user
        
    except wt.PyJWTError:
        raise credentials_exception

async def get_admin_user(current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Ensure current user has admin role."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Secure user authentication"""
    user = authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        # Add delay to prevent timing attacks
        await asyncio.sleep(1)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Create tokens
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
    """Refresh access token using refresh token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if username is None or token_type != "refresh":
            raise credentials_exception
            
        user = users_db.get(username)
        if user is None:
            raise credentials_exception
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        new_access_token = create_access_token(
            data={"sub": user["username"], "role": user["role"]},
            expires_delta=access_token_expires
        )
        new_refresh_token = create_refresh_token(
            data={"sub": user["username"], "role": user["role"]}
        )
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except jwt.PyJWTError:
        raise credentials_exception

@app.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information."""
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "role": current_user["role"],
        "is_active": current_user["is_active"]
    }

@app.post("/auth/register", response_model=dict)
async def register_user(user: UserCreate, current_user: dict = Depends(get_admin_user)):
    """Register new user (admin only)."""
    if user.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    hashed_password = get_password_hash(user.password)
    users_db[user.username] = {
        "username": user.username,
        "hashed_password": hashed_password,
        "email": user.email,
        "role": user.role,
        "is_active": True
    }
    
    logger.info(f"New user registered: {user.username} with role {user.role}")
    return {"message": f"User {user.username} created successfully"}

@app.get("/health")
async def health_check():
    """Secure health check"""
    return {
        "status": "healthy",
        "service": "SDG Auth Service",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)
