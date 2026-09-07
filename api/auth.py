import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=True)


def _load_tokens() -> dict[str, str]:
    #API_TOKENS="token1:alice,token2:bob" -> {"token1": "alice", "token2": "bob"}
    tokens: dict[str, str] = {}
    for pair in os.getenv("API_TOKENS", "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        token, _, user = pair.partition(":")
        if token and user:
            tokens[token] = user
    return tokens


_TOKENS = _load_tokens()


def current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    #resolve the bearer token to a user id; constant-time compare against each token
    presented = credentials.credentials
    for token, user in _TOKENS.items():
        if secrets.compare_digest(presented, token):
            return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing bearer token",
    )
