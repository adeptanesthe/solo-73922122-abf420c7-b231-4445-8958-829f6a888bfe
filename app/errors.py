"""Shared exception types."""

from __future__ import annotations


class InvalidInput(ValueError):
    """The request payload is malformed or violates a documented limit."""
