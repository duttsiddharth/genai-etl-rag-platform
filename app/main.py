"""Vercel entrypoint.

Vercel's Python runtime looks for a WSGI/ASGI `app` object in the file
referenced by vercel.json's `functions` config. This module re-exports
the real FastAPI app defined in src/api/main.py so the deployment entry
point stays a one-line shim and all real logic stays in src/.
"""
from src.api.main import app  # noqa: F401
