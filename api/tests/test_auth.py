import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import auth


def test_load_tokens_parses_pairs(monkeypatch):
    monkeypatch.setenv("API_TOKENS", "t1:alice, t2:bob")
    assert auth._load_tokens() == {"t1": "alice", "t2": "bob"}


def test_load_tokens_missing_env(monkeypatch):
    monkeypatch.delenv("API_TOKENS", raising=False)
    assert auth._load_tokens() == {}


def test_load_tokens_skips_blank_and_malformed(monkeypatch):
    #"" -> skipped, "malformed"/"tokenonly:" -> no user, ":nouser" -> no token
    monkeypatch.setenv("API_TOKENS", "t1:alice,,malformed,:nouser,tokenonly:")
    assert auth._load_tokens() == {"t1": "alice"}


def _creds(token):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_current_user_resolves_token(monkeypatch):
    monkeypatch.setattr(auth, "_TOKENS", {"secret": "alice"})
    assert auth.current_user(_creds("secret")) == "alice"


def test_current_user_rejects_unknown_token(monkeypatch):
    monkeypatch.setattr(auth, "_TOKENS", {"secret": "alice"})
    with pytest.raises(HTTPException) as exc:
        auth.current_user(_creds("wrong"))
    assert exc.value.status_code == 401
