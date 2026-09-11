"""API key generation and verification. Transport-agnostic.

Used by both the gRPC ingestion server and the FastAPI read API, so it depends
on neither. Backed by asyncpg (lighter than SQLAlchemy for a two-column lookup
on the hot path) with a Redis cache in front.

Credential format:

    Authorization: Basic base64(public_key ":" secret_key)

    public_key  pk-ow-<22 url-safe chars>   stored in plaintext, it is public
    secret_key  sk-ow-<43 url-safe chars>   stored ONLY as sha256(secret+salt)

WHY SHA-256 AND NOT BCRYPT — you will be asked this, so know the answer:
bcrypt and argon2 exist to make brute force expensive against LOW-ENTROPY human
passwords. These keys are 256 bits of CSPRNG output that we generate; there is
nothing to brute force. Bcrypt on the ingestion path would add ~100ms to every
request for no security gain. The real risks for an API key are leakage and
lack of rotation, which is what `expires_at`, `last_used_at` and the display
suffix address. (Langfuse stores both a bcrypt hash and a fast sha256 hash for
exactly this reason — the fast one serves the cached path.)

Set OPENWEAVE_SALT in production. A global salt is enough here: it defends
against precomputation, and per-key salts would defeat the cache lookup, which
is keyed by hash.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger("openweave.auth")

PUBLIC_PREFIX = "pk-ow-"
SECRET_PREFIX = "sk-ow-"

SALT = os.getenv("OPENWEAVE_SALT", "openweave-dev-salt")
CACHE_TTL_SECONDS = int(os.getenv("API_KEY_CACHE_TTL", "300"))
CACHE_PREFIX = "apikey:"

# Exposed for /metrics. Cache hit rate is a genuinely good README number.
STATS = {"cache_hit": 0, "cache_miss": 0, "reject": 0}


class AuthError(Exception):
    """Raised for any failed authentication. The message is safe to return."""


@dataclass(frozen=True)
class AuthContext:
    project_id: str
    api_key_id: str


# --------------------------------------------------------------------------- #
# Key generation
# --------------------------------------------------------------------------- #
def generate_key_pair() -> tuple[str, str]:
    """Return a fresh (public_key, secret_key). The secret is never stored."""
    return (
        PUBLIC_PREFIX + secrets.token_urlsafe(16),
        SECRET_PREFIX + secrets.token_urlsafe(32),
    )


def hash_secret(secret_key: str) -> str:
    return hashlib.sha256((secret_key + SALT).encode("utf-8")).hexdigest()


def display_suffix(secret_key: str) -> str:
    """What the UI shows so a human can tell two keys apart."""
    return f"{SECRET_PREFIX}...{secret_key[-4:]}"


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #
def parse_basic_auth(header: str | None) -> tuple[str, str]:
    """Split an HTTP Basic header into (public_key, secret_key)."""
    if not header:
        raise AuthError("missing authorization header")
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "basic":
        raise AuthError("expected 'Basic <base64>' authorization header")
    try:
        decoded = base64.b64decode(parts[1], validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        raise AuthError("malformed base64 in authorization header") from None
    public_key, sep, secret_key = decoded.partition(":")
    if not sep or not public_key or not secret_key:
        raise AuthError("authorization must be base64(public_key:secret_key)")
    return public_key, secret_key


def _is_expired(expires_at) -> bool:
    """Accepts a datetime from Postgres or an ISO string from the cache."""
    if expires_at is None:
        return False
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    return expires_at <= datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #
class ApiKeyVerifier:
    """Verifies credentials against Postgres, with a Redis cache in front.

    The cache is keyed by the SECRET HASH, not the public key. That matters: a
    cache hit then proves the caller knew the secret, so a stolen public key
    cannot ride the cache. Invalidate by deleting `apikey:<hash>` when a key is
    revoked.
    """

    def __init__(self, pool, redis_client, ttl: int = CACHE_TTL_SECONDS):
        self._pool = pool
        self._redis = redis_client
        self._ttl = ttl

    async def verify(self, authorization: str | None) -> AuthContext:
        public_key, secret_key = parse_basic_auth(authorization)
        key_hash = hash_secret(secret_key)

        cached = await self._cache_get(key_hash)
        # A hit must still check the public key and expiry. Returning it as-is
        # accepted ANY public key alongside a cached secret, and kept an expired
        # key working until its cache entry lapsed. A mismatch falls through to
        # Postgres, which rejects it.
        if cached is not None and cached["public_key"] == public_key:
            if _is_expired(cached["expires_at"]):
                STATS["reject"] += 1
                raise AuthError("credentials expired")
            STATS["cache_hit"] += 1
            return AuthContext(cached["project_id"], cached["api_key_id"])

        STATS["cache_miss"] += 1

        row = await self._pool.fetchrow(
            """
            SELECT id, project_id, expires_at
              FROM api_keys
             WHERE hashed_secret_key = $1 AND public_key = $2
            """,
            key_hash,
            public_key,
        )
        if row is None:
            STATS["reject"] += 1
            # Deliberately identical message for "no such key" and "wrong
            # secret" — distinguishing them tells an attacker which half of a
            # guess was right.
            raise AuthError("invalid credentials")

        if _is_expired(row["expires_at"]):
            STATS["reject"] += 1
            raise AuthError("credentials expired")

        ctx = AuthContext(project_id=row["project_id"], api_key_id=row["id"])
        await self._cache_put(key_hash, ctx, public_key, row["expires_at"])
        return ctx

    async def touch_last_used(self, api_key_id: str) -> None:
        """Fire-and-forget. Never await this inside the request path."""
        try:
            await self._pool.execute(
                "UPDATE api_keys SET last_used_at = now() WHERE id = $1", api_key_id
            )
        except Exception as exc:  # noqa: BLE001 — best effort by design
            logger.debug("last_used_at update failed: %s", exc)

    # --- cache ------------------------------------------------------------ #
    async def _cache_get(self, key_hash: str) -> dict | None:
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(CACHE_PREFIX + key_hash)
        except Exception as exc:  # noqa: BLE001
            # Cache failures must never reject a valid request. Fail OPEN to
            # Postgres, not closed to the caller.
            logger.warning("api key cache read failed, falling through: %s", exc)
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return {
                "project_id": data["project_id"],
                "api_key_id": data["api_key_id"],
                "public_key": data["public_key"],
                "expires_at": data["expires_at"],
            }
        except (json.JSONDecodeError, KeyError):
            # Also clears entries cached before public_key/expires_at were stored.
            await self._redis.delete(CACHE_PREFIX + key_hash)
            return None

    async def _cache_put(
        self, key_hash: str, ctx: AuthContext, public_key: str, expires_at
    ) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.setex(
                CACHE_PREFIX + key_hash,
                self._ttl,
                json.dumps(
                    {
                        "project_id": ctx.project_id,
                        "api_key_id": ctx.api_key_id,
                        "public_key": public_key,
                        "expires_at": expires_at.isoformat() if expires_at else None,
                    }
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("api key cache write failed: %s", exc)

    async def invalidate(self, secret_key: str) -> None:
        if self._redis is not None:
            await self._redis.delete(CACHE_PREFIX + hash_secret(secret_key))
