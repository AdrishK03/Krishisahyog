from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserDocument(BaseModel):
    id: str
    email: EmailStr
    name: str
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
