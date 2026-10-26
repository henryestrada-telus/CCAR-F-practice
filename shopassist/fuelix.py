"""Build the Claude client against the Fuel iX gateway.

Fuel iX serves the native Anthropic Messages API, so the official ``anthropic``
SDK is used unchanged. Only two things differ from a direct Anthropic account:

- ``base_url``   -> https://api.fuelix.ai   (instead of api.anthropic.com)
- ``auth_token`` -> your Fuel iX key, sent as ``Authorization: Bearer ...``
  (an Anthropic key would go in ``api_key`` and travel as ``x-api-key``).

Every script calls ``make_client()`` once at import time. The key lives in
``.env`` as ``FUELIX_API_KEY`` and never in the code. Create one by making a
project in the Fuel iX Dev Portal: https://dev.fuelix.ai
"""
import os

from anthropic import Anthropic
from dotenv import load_dotenv

FUELIX_BASE_URL = "https://api.fuelix.ai"


def make_client() -> Anthropic:
    """Return an Anthropic SDK client that talks to Fuel iX."""
    load_dotenv()  # reads .env in the current folder into os.environ
    key = os.environ.get("FUELIX_API_KEY", "").strip()
    if not key or key.lower().startswith(("paste", "your", "changeme")):
        raise SystemExit(
            "FUELIX_API_KEY is not set. Copy .env.example to .env and paste the "
            "key from your Fuel iX Dev Portal project (https://dev.fuelix.ai)."
        )
    return Anthropic(
        base_url=os.environ.get("FUELIX_BASE_URL", FUELIX_BASE_URL),
        auth_token=key,
    )
