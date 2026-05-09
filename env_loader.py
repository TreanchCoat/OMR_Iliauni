"""
env_loader.py — One-line .env loading for the OMR pipeline.

Usage
-----
At the top of any script that reads os.environ, add:

    import env_loader   # noqa  (loads .env into os.environ)

That's it.  All `os.environ.get(...)` calls below this import will see the
values from your .env file.

Why this and not just python-dotenv directly?
---------------------------------------------
python-dotenv is the standard library, but having one shared loader means:
  - Loading happens once per process even if multiple modules import this
  - Falls back gracefully if python-dotenv isn't installed
  - Looks for .env in a few standard locations (script dir, project root,
    cwd) so it works regardless of how scripts are launched
  - Prints a one-line confirmation in verbose mode

Install
-------
    pip install python-dotenv
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_env(verbose: bool = False) -> bool:
    """
    Load .env from the first location it's found.  Returns True if any
    file was loaded.  Search order:
        1. <script dir>/.env
        2. <cwd>/.env
        3. <script dir>/../.env
    """
    global _LOADED
    if _LOADED:
        return True

    try:
        from dotenv import load_dotenv
    except ImportError:
        if verbose:
            print('  python-dotenv not installed — skipping .env loading')
        return False

    here = Path(__file__).resolve().parent
    candidates = [
        here / '.env',
        Path.cwd() / '.env',
        here.parent / '.env',
    ]

    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)
            if verbose:
                print(f'  Loaded environment from {path}')
            _LOADED = True
            return True

    if verbose:
        print('  No .env file found — using OS environment only')
    return False


# Load automatically on import
load_env()
