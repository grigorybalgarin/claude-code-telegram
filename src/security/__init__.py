"""Security framework for Claude Code Telegram Bot.

This module provides comprehensive security features including:
- Multi-layer authentication (whitelist and token-based)
- Rate limiting with token bucket algorithm
- Path traversal and injection prevention
- Input validation and sanitization
- Security audit logging

Key Components:
- AuthenticationManager: Main authentication system
- RateLimiter: Request and cost-based rate limiting
- SecurityValidator: Input validation and path security
- AuditLogger: Security event logging
"""

from .audit import AuditEvent, AuditLogger, SQLiteAuditStorage
from .auth import (
    AuthenticationManager,
    AuthProvider,
    SQLiteTokenStorage,
    TokenAuthProvider,
    UserSession,
    WhitelistAuthProvider,
)
from .rate_limiter import RateLimitBucket, RateLimiter
from .validators import SecurityValidator

__all__ = [
    "AuthProvider",
    "WhitelistAuthProvider",
    "TokenAuthProvider",
    "SQLiteTokenStorage",
    "AuthenticationManager",
    "UserSession",
    "RateLimiter",
    "RateLimitBucket",
    "SecurityValidator",
    "AuditLogger",
    "AuditEvent",
    "SQLiteAuditStorage",
]
