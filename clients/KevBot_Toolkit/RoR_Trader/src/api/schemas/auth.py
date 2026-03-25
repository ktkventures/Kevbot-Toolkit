"""Auth request/response schemas."""

from pydantic import BaseModel


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class UserInfo(BaseModel):
    id: str
    email: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: int
    user: UserInfo
