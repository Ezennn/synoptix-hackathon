"""Adapter contract and concrete memory backends.

The ONLY thing a new backend must do is implement ``MemoryBackend`` from
``base`` and return a ``TurnResult`` from ``ask``. Import the contract from
here so call sites never depend on a concrete adapter module:

    from src.adapters import MemoryBackend, TurnResult
"""
from src.adapters.base import MemoryBackend, TurnResult

__all__ = ["MemoryBackend", "TurnResult"]
