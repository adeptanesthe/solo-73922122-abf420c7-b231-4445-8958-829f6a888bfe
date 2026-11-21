"""Shared helpers for the test suite: build small synthetic PNG payloads."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def png_b64(array: np.ndarray, mode: str = "RGB") -> str:
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def mask_b64(mask: np.ndarray) -> str:
    return png_b64((mask.astype(np.uint8) * 255), mode="L")


def decode_b64_png(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload)
    return np.asarray(Image.open(io.BytesIO(raw)))


def rgb(height: int, width: int, value: int = 0) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def rect_mask(height: int, width: int, value: bool = True) -> np.ndarray:
    return np.full((height, width), value, dtype=bool)


def patch_payload(
    height: int,
    width: int,
    candidates: list[tuple[int, int]],
    value: int = 200,
    mask: np.ndarray | None = None,
) -> dict:
    if mask is None:
        mask = rect_mask(height, width)
    return {
        "image_b64": png_b64(rgb(height, width, value)),
        "mask_b64": mask_b64(mask),
        "candidates": candidates,
    }
