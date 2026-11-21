"""Runtime configuration, overridable through environment variables."""

from __future__ import annotations

import os

# Hard protocol limits from the task description.
MAX_PATCHES = 4
MAX_CANDIDATES = 5

# At most 5 candidates on each of at most 4 patches.
MAX_SEARCH_NODES = MAX_CANDIDATES**MAX_PATCHES

# Every pasted product instance must keep at least half of its original mask
# area after clipping and occlusion ("每个实例都须保留原掩码面积的至少一半").
MIN_VISIBLE_NUMERATOR = 1
MIN_VISIBLE_DENOMINATOR = 2


def _int_env(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover - startup guard
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def _ratio_env(name: str, default_num: int, default_den: int) -> tuple[int, int]:
    raw = os.environ.get(name)
    if raw is None:
        return default_num, default_den
    try:
        num_text, den_text = raw.split("/")
        num, den = int(num_text), int(den_text)
    except ValueError as exc:  # pragma: no cover - startup guard
        raise RuntimeError(f"{name} must look like '1/2', got {raw!r}") from exc
    if den <= 0 or not (0 <= num <= den):
        raise RuntimeError(f"{name} must satisfy 0 <= numerator <= denominator")
    return num, den


# Per-image limits, applied to the decoded byte length and to the pixel grid
# declared in the image header.
MAX_IMAGE_BYTES = _int_env("MAX_IMAGE_BYTES", 4 * 1024 * 1024, minimum=1024)
MAX_IMAGE_PIXELS = _int_env("MAX_IMAGE_PIXELS", 4_194_304, minimum=1)  # 2048^2
MAX_BASE_INSTANCES = _int_env("MAX_BASE_INSTANCES", 64, minimum=0)
# Whole-body limit; base64 inflates by 4/3, so this admits ~24 MiB of image data.
MAX_REQUEST_BYTES = _int_env("MAX_REQUEST_BYTES", 32 * 1024 * 1024, minimum=1024)

# Feasibility floor for pasted instances. The task fixes it at one half; the
# override exists so the threshold can be exercised in tests.
PATCH_MIN_VISIBLE_NUMERATOR, PATCH_MIN_VISIBLE_DENOMINATOR = _ratio_env(
    "MIN_VISIBLE_RATIO", MIN_VISIBLE_NUMERATOR, MIN_VISIBLE_DENOMINATOR
)

