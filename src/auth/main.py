# /sdg_root/src/auth/main.py

import os
import re
import jwt
import json
import bcrypt
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, Body, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .jwt_manager import jwt_manager
from .config import settings
from ..core.secrets_manager import secrets_manager
from ..core.dependency_manager import (
    dependency_manager,
    wait_for_dependencies,
    setup_sdg_dependencies,
    get_dependency_status,
)
from .token_store import register_refresh_token, consume_refresh_token
from .repository import UserRepository
from .models import init_db, SessionLocal, User

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

repo = UserRepository()
security = HTTPBearer()

# ---------- Pydantic models ----------
class UserLogin(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)

class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8)
    email: str = Field(..., min_length=3, max_length=320)
    role: str = Field("user", min_length=2, max_length=50)

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="Refresh token")

# ---------- Helpers ----------
def parse_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS") or os.getenv("ALLOWEDORIGINS", "")
    s = raw.strip()
    if not s:
        if (os.getenv("ENVIRONMENT") or "").lower() == "development":
            return ["http://localhost:3000", "http://localhost:8080"]
        return []
    if s.startswith("["):
        try:
            data = json.loads(s)
            return [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]

def get_allowed_origins():
    origins = os.environ.get("ALLOWED_ORIGINS", "").split(",")
    origins = [o.strip() for o in origins if o.strip()]
    if not origins:
        if os.environ.get("ENVIRONMENT") == "development":
            return ["http://localhost:3000", "http://localhost:8080"]
        raise ValueError("ALLOWED_ORIGINS must be set in production")
    for origin in origins:
        if not origin.startswith(("http://", "https://")):
            raise ValueError(f"Invalid origin format: {origin}")
    return origins

def validate_password_policy(pw: str):
    if len(pw) < settings.password_min_length:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {settings.password_min_length} characters",
        )
    if sum(bool(re.search(pat, pw)) for pat in [r"[A-Z]", r"[a-z]"]) < 2:
        raise HTTPException(status_code=400, detail="Password must include upper and lower case letters")
    if not re.search(r"[\d\W_]", pw):
        raise HTTPException(status_code=400, detail="Password must include a number or symbol")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def authenticate_user(username: str, password: str) -> Optional[User]:
    user = repo.get_by_username(username)
    if user and user.is_active and repo.verify_password(password, user.hashed_password):
        repo.touch_login(user)
        return user
    return None

# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting SDG Auth Service...")
    try:
        setup_sdg_dependencies()

        async def initialize_auth_service():
            logger.info("🔄 Initializing auth service dependencies...")
            await wait_for_dependencies("database")
            init_db()
            admin_pw = None
            try:
                admin_pw = secrets_manager.get_secret("ADMIN_PASSWORD")
            except Exception as e:
                logger.warning(f"Secrets manager not available for ADMIN_PASSWORD: {e}")
            if not admin_pw:
                admin_pw = "admin123" if os.environ.get("ENVIRONMENT") != "production" else None
                if not admin_pw:
                    logger.warning("No ADMIN_PASSWORD configured; skipping admin seed in production")
            if admin_pw:
                if not repo.get_by_username("hwe"):
                    repo.create_user(
                        username="hwe",
                        email="heinrich.ekam@gmail.com",
                        password=admin_pw,
                        role="admin",
                    )
                    logger.info("✅ Seeded default admin user 'hwe'")
                else:
                    logger.info("ℹ️ Admin user 'hwe' already present")
            logger.info("✅ Auth service dependencies initialized")

        dependency_manager.register_startup_task("auth", initialize_auth_service)
        if not dependency_manager._startup_complete.is_set():
            await dependency_manager.start_all_services()
        else:
            await initialize_auth_service()

        logger.info("✅ SDG Auth Service startup completed")
        yield
    finally:
        logger.info("🔄 Shutting down SDG Auth Service...")
        try:
            if dependency_manager._startup_complete.is_set():
                await dependency_manager.shutdown_all_services()
            logger.info("✅ SDG Auth Service shutdown completed")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

# ---------- Routers ----------
auth_router = APIRouter(prefix="/auth", tags=["auth"])
sys_router = APIRouter(tags=["system"])

# Auth endpoints
@auth_router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin):
    await ensure_auth_ready()
    user = authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        await asyncio.sleep(1)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    claims = {"sub": user.username, "role": user.role}
    access = jwt_manager.create_access_token(claims)
    refresh = jwt_manager.create_refresh_token(claims)
    payload = jwt_manager.verify_token(refresh, "refresh")
    register_refresh_token(payload["sub"], payload["jti"], int(payload["exp"].timestamp()))
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "expires_in": jwt_manager.access_token_expire_minutes * 60,
    }

@auth_router.post("/refresh", response_model=Token)
async def refresh_token(req: RefreshRequest):
    await ensure_auth_ready()
    try:
        payload = jwt_manager.verify_token(req.refresh_token, "refresh")
        result = consume_refresh_token(payload["jti"])
        if result == "reused":
            raise HTTPException(status_code=401, detail="Refresh token reuse detected; family revoked")
        if result in ("revoked", "unknown"):
            raise HTTPException(status_code=401, detail="Refresh token invalid")
        claims = {"sub": payload["sub"], "role": payload.get("role")}
        new_access = jwt_manager.create_access_token(claims)
        new_refresh = jwt_manager.create_refresh_token(claims)
        new_p = jwt_manager.verify_token(new_refresh, "refresh")
        register_refresh_token(new_p["sub"], new_p["jti"], int(new_p["exp"].timestamp()))
        return {
            "access_token": new_access,
            "refresh_token": new_refresh,
            "token_type": "bearer",
            "expires_in": jwt_manager.access_token_expire_minutes * 60,
        }
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

@auth_router.get("/me")
async def get_current_user_info(current_user: dict = Depends(lambda c=Depends(security): c)):
    # Delegate to the shared dependency so we can mount on router
    return await _current_user()

async def _current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    return await get_current_user(credentials)

@auth_router.post("/register", response_model=dict)
async def register_user(user: UserCreate, current_user: dict = Depends(lambda: None)):
    # Enforce admin via explicit dependency
    admin = await get_admin_user(await get_current_user(Depends(security)))  # pragma: no cover
    await ensure_auth_ready()
    if repo.get_by_username(user.username):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    validate_password_policy(user.password)
    repo.create_user(username=user.username, email=user.email, password=user.password, role=user.role)
    logger.info(f"New user registered: {user.username} with role {user.role}")
    return {"message": f"User {user.username} created successfully"}

# System endpoints
@sys_router.get("/health")
async def health_check():
    try:
        dependency_status = await get_dependency_status()
        user_count = 0
        active_count = 0
        try:
            with SessionLocal() as s:
                user_count = s.query(User).count()
                active_count = s.query(User).filter(User.is_active.is_(True)).count()
        except Exception as e:
            logger.warning(f"Auth DB stats unavailable: {e}")
        overall_status = dependency_status.get("overall_status") or "starting"
        return {
            "status": overall_status,
            "service": "SDG Auth Service",
            "version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "users": {"total": user_count, "active": active_count},
            "dependencies": dependency_status,
            "components": {
                "jwt_manager": "ready",
                "secrets_manager": "configured",
                "dependency_manager": "active" if dependency_manager._startup_complete.is_set() else "starting",
            },
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "service": "SDG Auth Service",
            "version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }

@auth_router.get("/status")
async def get_auth_status(current_user: dict = Depends(lambda: None)):
    dependency_status = await get_dependency_status()
    with SessionLocal() as s:
        total = s.query(User).count()
        active = s.query(User).filter(User.is_active.is_(True)).count()
    return {
        "service_name": "auth",
        "status": dependency_status,
        "user_count": total,
        "active_users": active,
        "startup_complete": dependency_manager._startup_complete.is_set(),
        "last_check": datetime.utcnow().isoformat(),
    }

# ---------- FastAPI app ----------
app = FastAPI(
    title="SDG Auth Service",
    description="Authentication and authorization service for SDG AI Pipeline",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sys_router)
app.include_router(auth_router)

# ---------- Shared dependencies ----------
async def ensure_auth_ready():
    await wait_for_dependencies("database")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
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
        if not username:
            raise credentials_exception
        user = repo.get_by_username(username)
        if not user or not user.is_active:
            raise credentials_exception
        return {
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "is_active": user.is_active,
        }
    except jwt.PyJWTError:
        raise credentials_exception

async def get_admin_user(current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    return current_user

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)
