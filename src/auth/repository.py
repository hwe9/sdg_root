# /sdg_root/src/auth/repository.py
import bcrypt
from datetime import datetime
from typing import Optional
from .models import SessionLocal, User

class UserRepository:
    def __init__(self):
        self.Session = SessionLocal

    def get_by_username(self, username: str) -> Optional[User]:
        with self.Session() as s:
            return s.query(User).filter(User.username == username).first()

    def create_user(self, username: str, email: str, password: str, role: str = "user") -> User:
        hpw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()
        with self.Session() as s:
            user = User(username=username, email=email, hashed_password=hpw, role=role)
            s.add(user); s.commit(); s.refresh(user)
            return user

    def verify_password(self, plain: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(plain.encode(), hashed.encode())
        except Exception:
            return False

    def touch_login(self, user: User) -> None:
        with self.Session() as s:
            s.query(User).filter(User.id == user.id).update({"last_login": datetime.utcnow()})
            s.commit()
