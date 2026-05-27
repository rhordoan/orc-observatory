"""Benchmark instance loaders (TSPLIB, QAPLIB)."""

from .tsplib import load_tsplib, download_tsplib
from .qaplib import load_qaplib, download_qaplib

__all__ = [
    "load_tsplib",
    "download_tsplib",
    "load_qaplib",
    "download_qaplib",
]
