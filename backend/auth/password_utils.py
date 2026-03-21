from passlib.context import CryptContext

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
MAX_LEN = 72  # bcrypt limit

def _normalize_password(password: str) -> str:
    return password.encode("utf-8")[:MAX_LEN].decode("utf-8", errors="ignore")

def hash_password(password: str) -> str:
    return _pwd_context.hash(_normalize_password(password))

def verify_password(password: str, hashed: str) -> bool:
    return _pwd_context.verify(_normalize_password(password), hashed)
