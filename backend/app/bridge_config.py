"""Configuration for FastAPI ↔ AWS bridge.

Local FastAPI should treat AWS (edenbot.duckdns.org) as the source of truth
for auth and balances. These settings are intentionally simple and can be
refined later if needed.
"""

AWS_API_BASE = "https://edenbot.duckdns.org:8443/api"

# HTTP client behaviour
REQUEST_TIMEOUT_SECONDS: float = 10.0
VERIFY_TLS: bool = True

# JWT-related metadata for future validation hooks (not yet enforced here)
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "eden-backend-aws"
JWT_AUDIENCE = "eden-mobile"
