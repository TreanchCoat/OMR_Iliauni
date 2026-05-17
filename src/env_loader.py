"""
env_loader.py — One-line .env loading + path bootstrap for the OMR pipeline.

Usage
-----
At the very top of any entry-point script, add::

    import env_loader   # noqa  (loads .env, registers src/ on sys.path)

That's it.  After this line, ``os.environ.get(...)`` sees ``.env`` values
and ``from xml_builder import XMLBuilder`` (and the other src/ modules)
works regardless of which folder the script was launched from.

What it does
------------
1. **Project root resolution.**  This file lives at ``<project>/src/env_loader.py``.
   The project root is the parent of its directory.  This is computed once
   and exposed as ``env_loader.PROJECT_ROOT``.

2. **.env loading.**  Searches in order:
       - ``<project root>/.env``
       - ``<cwd>/.env``
       - ``<script dir>/.env``
   The first one found is loaded into ``os.environ`` (without overriding
   existing values).  Gracefully no-ops if ``python-dotenv`` isn't
   installed.

3. **sys.path bootstrap.**  Adds ``<project>/src`` to ``sys.path`` so that
   flat imports like ``from xml_builder import XMLBuilder`` work from any
   entry point.

4. **Path helpers.**  Common project paths (models, data, sample, etc.) are
   exposed as module-level constants whose values come from environment
   variables when present and fall back to sensible defaults under the
   project root.  Use them instead of hard-coding paths::

       from env_loader import MODELS_DIR, INPUT_DIR, OUTPUT_BASE_DIR

Install
-------
    pip install python-dotenv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Project layout
# ─────────────────────────────────────────────────────────────────────────────

# env_loader.py lives at <project>/src/env_loader.py, so the project root
# is the parent of THIS file's directory.
SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SRC_DIR.parent


# ─────────────────────────────────────────────────────────────────────────────
# .env loading
# ─────────────────────────────────────────────────────────────────────────────

_LOADED = False


def load_env(verbose: bool = False) -> bool:
    """
    Load ``.env`` from the first location it's found.  Returns True if any
    file was loaded.  Search order::

        1. <project root>/.env
        2. <cwd>/.env
        3. <script dir>/.env
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

    candidates = [
        PROJECT_ROOT / '.env',
        Path.cwd() / '.env',
        SRC_DIR / '.env',
    ]

    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)
            if verbose:
                print(f'  Loaded environment from {path}')
            _LOADED = True
            break

    if not _LOADED and verbose:
        print('  No .env file found — using OS environment only')
    return _LOADED


# Load env BEFORE we compute the path helpers below (they read env vars).
load_env()


# ─────────────────────────────────────────────────────────────────────────────
# sys.path bootstrap
# ─────────────────────────────────────────────────────────────────────────────

_src_str = str(SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers (configurable via .env, with sensible defaults)
# ─────────────────────────────────────────────────────────────────────────────

def _path_env(name: str, default: Path) -> Path:
    """Read a path from ``$NAME`` or fall back to ``default``."""
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


MODELS_DIR:      Path = _path_env('MODELS_DIR',      PROJECT_ROOT / 'models')
DATA_DIR:        Path = _path_env('DATA_DIR',        PROJECT_ROOT / 'data')
INPUT_DIR:       Path = _path_env('INPUT_DIR',       DATA_DIR / 'input')
SAMPLE_DATA_DIR: Path = _path_env('SAMPLE_DATA_DIR', DATA_DIR / 'sample')
OUTPUT_BASE_DIR: Path = _path_env('OUTPUT_BASE_DIR', PROJECT_ROOT / 'output')
DOCS_DIR:        Path = _path_env('DOCS_DIR',        PROJECT_ROOT / 'docs')
ARCHIVE_DIR:     Path = _path_env('ARCHIVE_DIR',     PROJECT_ROOT / 'archive')

# Model file paths.  Defaults follow the convention
# <MODELS_DIR>/<filename> if no override is set in .env.
MODEL_PATH:      Path = _path_env('MODEL_PATH',
                                  MODELS_DIR / 'deepscores_crops_v1.pt')
OLA_MODEL_PATH:  Path = _path_env('OLA_MODEL_PATH',
                                  MODELS_DIR / 'ola_v2.pt')
UNET_MODEL_PATH: Path = _path_env('UNET_MODEL_PATH',
                                  MODELS_DIR / 'staff_unet.pth')

# Dummy-API fixtures — used by api/dummy_api.py
DUMMY_RECTIFIED_PATH: Path = _path_env(
    'DUMMY_RECTIFIED_PATH', SAMPLE_DATA_DIR / 'rectified.png')
DUMMY_XML_PATH: Path = _path_env(
    'DUMMY_XML_PATH', SAMPLE_DATA_DIR / 'score.xml')
DUMMY_DETECTIONS_PATH: Path = _path_env(
    'DUMMY_DETECTIONS_PATH', SAMPLE_DATA_DIR / 'detections.json')

# API server settings (used by api/real_api.py and api/dummy_api.py).
HOST:         str = os.environ.get('HOST', '0.0.0.0')
PORT:         int = int(os.environ.get('PORT', '5000'))
MAX_IMAGE_MB: int = int(os.environ.get('MAX_IMAGE_MB', '50'))


__all__ = [
    'load_env',
    'PROJECT_ROOT', 'SRC_DIR',
    'MODELS_DIR', 'DATA_DIR', 'INPUT_DIR', 'SAMPLE_DATA_DIR',
    'OUTPUT_BASE_DIR', 'DOCS_DIR', 'ARCHIVE_DIR',
    'MODEL_PATH', 'OLA_MODEL_PATH', 'UNET_MODEL_PATH',
    'DUMMY_RECTIFIED_PATH', 'DUMMY_XML_PATH', 'DUMMY_DETECTIONS_PATH',
    'HOST', 'PORT', 'MAX_IMAGE_MB',
]
