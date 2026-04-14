from passlib.context import CryptContext
from fastapi import HTTPException

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

MAX_LEN = 72

def hash_password(password: str) -> str:
    if len(password.encode("utf-8")) > MAX_LEN:
        raise HTTPException(
            status_code=400,
            detail="Password too long (max 72 bytes)"
        )
    return _pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return _pwd_context.verify(password, hashed)
