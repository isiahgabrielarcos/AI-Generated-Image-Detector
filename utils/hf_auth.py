"""
hf_auth.py
──────────
Load the Hugging Face token from .env and authenticate with HF Hub.

Call setup_hf_auth() once at the top of any script that downloads
or accesses models (cache_clip_features.py, train.py, etc.).

What this does:
  1. Reads HF_TOKEN from .env (no extra dependencies needed)
  2. Sets the HF_TOKEN and HUGGINGFACE_HUB_TOKEN environment variables
  3. Calls huggingface_hub.login() so all HF Hub requests are authenticated

Benefits:
  - Higher rate limits for model downloads (open_clip, timm, transformers)
  - Access to gated models if you have been granted permission
  - More reliable downloads on slow / restricted networks
"""

import os
from pathlib import Path


def _load_dotenv(env_path: str | Path = ".env") -> dict:
    """
    Minimal .env parser — no external dependency required.
    Reads KEY=VALUE lines, ignores comments and blank lines.
    """
    env_path = Path(env_path)
    found = {}
    if not env_path.exists():
        return found
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")   # remove optional quotes
            found[key] = value
    return found


def setup_hf_auth(env_path: str | Path = ".env", verbose: bool = True) -> bool:
    """
    Load the HF token and authenticate with Hugging Face Hub.

    Args:
        env_path : path to the .env file (default: .env in cwd)
        verbose  : print a one-line status message

    Returns:
        True if authenticated successfully, False otherwise.
    """
    env_vars = _load_dotenv(env_path)
    token = env_vars.get("HF_TOKEN") or os.environ.get("HF_TOKEN")

    if not token:
        if verbose:
            print("[hf_auth] No HF_TOKEN found in .env or environment — "
                  "model downloads may be rate-limited.")
        return False

    # Export to both variable names so every library picks it up
    os.environ["HF_TOKEN"]               = token
    os.environ["HUGGINGFACE_HUB_TOKEN"]  = token

    try:
        import huggingface_hub
        huggingface_hub.login(token=token, add_to_git_credential=False)
        if verbose:
            # Mask all but the last 4 chars for safe logging
            masked = "*" * (len(token) - 4) + token[-4:]
            print(f"[hf_auth] Authenticated with HF Hub (token: {masked})")
        return True
    except Exception as exc:
        if verbose:
            print(f"[hf_auth] huggingface_hub.login() failed: {exc}")
        return False
