"""Compositing core: decoding, feasibility search and rendering.

Semantics
---------
* A patch is composited through its own mask only; pixels outside the patch
  mask never reach the canvas.
* Stacking order is the request order, back to front (patch 0 is bottom-most).
  A later patch hides an earlier one wherever both masks cover a pixel.
* An instance's visible mask is its mask clipped to the canvas, minus every
  later instance's mask. It must keep at least half of the instance's
  *original* mask pixel count after clipping and occlusion, otherwise the
  candidate combination is infeasible. The floor is applied with integer
  arithmetic, so exactly half is feasible.
* Among feasible combinations the search minimises the number of annotated
  base pixels covered by pasted instances (union, not sum). Ties are broken by
  maximising the smallest visible-area margin, then by candidate order, which
  makes the result deterministic.

The search is exhaustive over at most 5^4 = 625 candidate combinations. The
objective is not separable because a later patch can hide an earlier one and
thereby change both feasibility and coverage, so a per-patch greedy choice is
not guaranteed to be optimal.

Efficiency
----------
The DFS walks patches top-most first while maintaining the union of already
placed (later) masks. When patch ``i`` is placed, every patch that can hide it
is already in the union, so its final visible count and its incremental
coverage of annotated pixels are both exact and can be used to prune
immediately. Per-node work is proportional to patch area; canvas-sized arrays
are allocated once and mutated with exact backtracking.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

from .config import (
    MAX_BASE_INSTANCES,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_SEARCH_NODES,
    PATCH_MIN_VISIBLE_DENOMINATOR,
    PATCH_MIN_VISIBLE_NUMERATOR,
)
from .errors import InvalidInput

ImageFile.LOAD_TRUNCATED_IMAGES = False
# We enforce our own pixel limit from the header before decoding pixel data.
Image.MAX_IMAGE_PIXELS = None

Box = tuple[int, int, int, int]  # rows_start, rows_stop, cols_start, cols_stop


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def decode_b64_image(label: str, payload: str) -> Image.Image:
    """Decode a base64 image payload, enforcing the configured byte/pixel limits.

    The pixel limit is checked from the header before pixels are materialised,
    so decompression bombs are rejected cheaply.
    """

    if not isinstance(payload, str) or not payload.strip():
        raise InvalidInput(f"{label}: payload is empty")
    # Accept data URLs, URL-safe alphabets and missing padding for convenience.
    compact = "".join(payload.split())
    if compact.startswith("data:"):
        _, _, compact = compact.partition(",")
    compact = compact.replace("-", "+").replace("_", "/").rstrip("=")
    if len(compact) * 3 // 4 > MAX_IMAGE_BYTES:
        raise InvalidInput(f"{label}: decoded payload exceeds {MAX_IMAGE_BYTES} bytes")
    compact += "=" * ((-len(compact)) % 4)
    try:
        raw = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidInput(f"{label}: invalid base64 data ({exc})") from exc
    if not raw:
        raise InvalidInput(f"{label}: decoded payload is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise InvalidInput(f"{label}: decoded payload exceeds {MAX_IMAGE_BYTES} bytes")

    try:
        image = Image.open(io.BytesIO(raw))
        width, height = image.size
        if width <= 0 or height <= 0:
            raise InvalidInput(f"{label}: image has zero extent")
        if width * height > MAX_IMAGE_PIXELS:
            raise InvalidInput(
                f"{label}: image is {width}x{height}, above the "
                f"{MAX_IMAGE_PIXELS} pixel limit"
            )
        image.load()
    except InvalidInput:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidInput(f"{label}: unsupported or corrupt image ({exc})") from exc
    return image


def _mask_array(image: Image.Image) -> np.ndarray:
    """Convert any mask encoding to bool; any non-zero sample is instance."""

    if image.mode == "F":
        return np.asarray(image, dtype=np.float32) > 0
    if image.mode in {"I", "I;16", "I;16B", "I;16L"}:
        return np.asarray(image, dtype=np.int64) > 0
    return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def decode_mask(label: str, payload: str, size: tuple[int, int]) -> np.ndarray:
    """Decode a mask and require it to align with its image's pixel grid."""

    image = decode_b64_image(f"{label} (mask)", payload)
    width, height = size
    if image.size != (width, height):
        raise InvalidInput(
            f"{label}: mask is {image.size[0]}x{image.size[1]} but its image is "
            f"{width}x{height}; masks must align with their image"
        )
    return np.ascontiguousarray(_mask_array(image))


def decode_rgb(label: str, payload: str) -> tuple[np.ndarray, tuple[int, int]]:
    image = decode_b64_image(label, payload)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb), image.size


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Candidate:
    """One concrete placement of a patch, clipped to the canvas."""

    x: int
    y: int
    box: Box
    mask: np.ndarray  # clipped bool mask, C-contiguous
    area: int


@dataclass(frozen=True)
class PreparedPatch:
    index: int
    rgb: np.ndarray
    full_mask: np.ndarray
    original_pixels: int
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class BaseInfo:
    rgb: np.ndarray
    masks: tuple[np.ndarray, ...]
    annotated: np.ndarray  # union of the base instance masks
    original_pixels: tuple[int, ...]


@dataclass
class Solution:
    positions: list[tuple[int, int]]
    visible_masks: list[np.ndarray]  # canvas-sized bool, one per patch
    visible_pixels: list[int]
    base_visible_masks: list[np.ndarray]  # canvas-sized bool, one per base instance
    base_visible_pixels: list[int]
    overlapped_base_pixels: int
    composed: np.ndarray  # HxWx3 uint8
    evaluated_combinations: int


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def clip_box(
    canvas_shape: tuple[int, int], patch_shape: tuple[int, int], x: int, y: int
) -> Box | None:
    """Canvas box covered by a patch whose top-left is (x, y), or None."""

    canvas_h, canvas_w = canvas_shape
    patch_h, patch_w = patch_shape
    r0, c0 = max(0, y), max(0, x)
    r1, c1 = min(canvas_h, y + patch_h), min(canvas_w, x + patch_w)
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, r1, c0, c1


def usable_candidates(
    canvas_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    candidates: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Drop candidates whose patch would fall entirely outside the canvas."""

    return [
        (x, y)
        for x, y in candidates
        if clip_box(canvas_shape, patch_shape, x, y) is not None
    ]


def composite(canvas: np.ndarray, rgb: np.ndarray, mask: np.ndarray, x: int, y: int) -> None:
    """Write masked patch pixels onto the canvas in place, clipping at edges."""

    box = clip_box(canvas.shape[:2], mask.shape, x, y)
    if box is None:
        return
    r0, r1, c0, c1 = box
    src_rows = slice(r0 - y, r1 - y)
    src_cols = slice(c0 - x, c1 - x)
    source_mask = mask[src_rows, src_cols]
    if not source_mask.any():
        return
    canvas[r0:r1, c0:c1][source_mask] = rgb[src_rows, src_cols][source_mask]


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


def _prepare_patches(
    patches: Sequence[tuple[np.ndarray, np.ndarray, Sequence[tuple[int, int]]]],
    canvas_shape: tuple[int, int],
) -> list[PreparedPatch]:
    prepared: list[PreparedPatch] = []
    for index, (rgb, mask, candidates) in enumerate(patches):
        if mask.shape != rgb.shape[:2]:
            raise InvalidInput(f"patch {index}: image and mask sizes differ")
        original_pixels = int(np.count_nonzero(mask))
        if original_pixels == 0:
            raise InvalidInput(f"patch {index}: mask contains no pixels")
        allowed = usable_candidates(canvas_shape, mask.shape, candidates)
        built: list[Candidate] = []
        for x, y in allowed:
            box = clip_box(canvas_shape, mask.shape, x, y)
            assert box is not None
            r0, r1, c0, c1 = box
            clipped = np.ascontiguousarray(mask[r0 - y : r1 - y, c0 - x : c1 - x])
            area = int(np.count_nonzero(clipped))
            if area == 0:
                continue
            built.append(Candidate(x=x, y=y, box=box, mask=clipped, area=area))
        if not built:
            # Every candidate puts the whole mask outside the canvas: no
            # arrangement can satisfy the area floor for this patch.
            raise _NoFeasiblePlacement(
                f"patch {index}: no candidate keeps any mask pixel inside the canvas"
            )
        prepared.append(
            PreparedPatch(
                index=index,
                rgb=rgb,
                full_mask=mask,
                original_pixels=original_pixels,
                candidates=tuple(built),
            )
        )
    return prepared


class _NoFeasiblePlacement(Exception):
    """Input is well-formed but no arrangement can satisfy the constraints."""


class _Search:
    """Exact DFS over candidate tuples, top-most patch first."""

    def __init__(
        self,
        base: BaseInfo,
        prepared: Sequence[PreparedPatch],
        canvas_shape: tuple[int, int],
    ) -> None:
        self.base = base
        self.prepared = prepared
        self.canvas_shape = canvas_shape
        self.height, self.width = canvas_shape
        # Union of masks of patches placed so far (i.e. later layers).
        self.union = np.zeros(canvas_shape, dtype=bool)
        self.covered_annotated = 0
        self.evaluated = 0
        # Final visible count per patch for the combo currently on the stack.
        self.visible_now = [0] * len(prepared)
        self.best_key: tuple[int, int, tuple[int, ...]] | None = None
        self.best_combo: tuple[int, ...] | None = None

    # -- DFS -----------------------------------------------------------------
    def run(self) -> tuple[Solution | None, str | None]:
        total = 1
        for patch in self.prepared:
            total *= len(patch.candidates)
        if total > MAX_SEARCH_NODES:
            raise InvalidInput(
                f"placement search would evaluate {total} combinations, above the "
                f"server limit of {MAX_SEARCH_NODES}"
            )
        self._visit(len(self.prepared) - 1, [0] * len(self.prepared))
        if self.best_combo is None:
            return None, (
                "no candidate combination keeps at least half of every product "
                "instance's original mask area"
            )
        return self._render(), None

    def _visit(self, index: int, combo: list[int]) -> None:
        if index < 0:
            self.evaluated += 1
            slack = min(
                2 * self.visible_now[i] - patch.original_pixels
                for i, patch in enumerate(self.prepared)
            )
            key = (self.covered_annotated, -slack, tuple(combo))
            if self.best_key is None or key < self.best_key:
                self.best_key = key
                self.best_combo = tuple(combo)
            return

        patch = self.prepared[index]
        annotated = self.base.annotated
        for candidate_index, candidate in enumerate(patch.candidates):
            r0, r1, c0, c1 = candidate.box
            union_view = self.union[r0:r1, c0:c1]
            new = candidate.mask & ~union_view
            visible = int(np.count_nonzero(new))
            # Only later (already placed) patches can hide this one, so this
            # count is final and can prune the branch immediately.
            if (
                PATCH_MIN_VISIBLE_DENOMINATOR * visible
                < PATCH_MIN_VISIBLE_NUMERATOR * patch.original_pixels
            ):
                continue
            covered_now = int(np.count_nonzero(new & annotated[r0:r1, c0:c1]))
            union_view |= new
            self.covered_annotated += covered_now
            combo[index] = candidate_index
            self.visible_now[index] = visible
            self._visit(index - 1, combo)
            self.covered_annotated -= covered_now
            union_view &= ~new
        combo[index] = 0

    # -- rendering -----------------------------------------------------------
    def _render(self) -> Solution:
        assert self.best_combo is not None
        height, width = self.canvas_shape
        labels = np.zeros((height, width), dtype=np.uint8)
        positions: list[tuple[int, int]] = []
        for index, candidate_index in enumerate(self.best_combo):
            candidate = self.prepared[index].candidates[candidate_index]
            r0, r1, c0, c1 = candidate.box
            labels[r0:r1, c0:c1][candidate.mask] = index + 1
            positions.append((candidate.x, candidate.y))

        occupied = labels != 0
        visible_masks = [labels == index + 1 for index in range(len(self.prepared))]
        base_visible_masks = [mask & ~occupied for mask in self.base.masks]

        composed = self.base.rgb.copy()
        for patch, (x, y) in zip(self.prepared, positions):
            composite(composed, patch.rgb, patch.full_mask, x, y)

        return Solution(
            positions=positions,
            visible_masks=visible_masks,
            visible_pixels=[int(np.count_nonzero(m)) for m in visible_masks],
            base_visible_masks=base_visible_masks,
            base_visible_pixels=[int(np.count_nonzero(m)) for m in base_visible_masks],
            overlapped_base_pixels=int(np.count_nonzero(self.base.annotated & occupied)),
            composed=composed,
            evaluated_combinations=self.evaluated,
        )


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def build_base(image_b64: str, instance_masks_b64: Sequence[str]) -> BaseInfo:
    """Decode the base image and validate its non-overlapping instance masks."""

    if len(instance_masks_b64) > MAX_BASE_INSTANCES:
        raise InvalidInput(
            f"base: {len(instance_masks_b64)} instance masks provided, above the "
            f"limit of {MAX_BASE_INSTANCES}"
        )
    rgb, size = decode_rgb("base", image_b64)
    height, width = rgb.shape[:2]
    masks: list[np.ndarray] = []
    annotated = np.zeros((height, width), dtype=bool)
    for index, payload in enumerate(instance_masks_b64):
        label = f"base.instance_masks[{index}]"
        mask = decode_mask(label, payload, size)
        if not mask.any():
            raise InvalidInput(f"{label}: mask contains no pixels")
        overlap = int(np.count_nonzero(mask & annotated))
        if overlap:
            raise InvalidInput(
                f"{label}: overlaps {overlap} pixels of an earlier base instance; "
                "base instance masks must not overlap"
            )
        annotated |= mask
        masks.append(mask)
    return BaseInfo(
        rgb=rgb,
        masks=tuple(masks),
        annotated=annotated,
        original_pixels=tuple(int(np.count_nonzero(m)) for m in masks),
    )


def solve(
    base: BaseInfo,
    patches: Sequence[tuple[np.ndarray, np.ndarray, Sequence[tuple[int, int]]]],
) -> tuple[Solution | None, str | None]:
    """Run the exhaustive placement search.

    ``patches`` holds ``(rgb, mask, candidates)`` triples in request order,
    i.e. index 0 is the bottom-most layer. Returns ``(solution, None)`` on
    success or ``(None, reason)`` when no arrangement satisfies the area
    floor. Raises `InvalidInput` for malformed placements.
    """

    canvas_shape = base.rgb.shape[:2]
    try:
        prepared = _prepare_patches(patches, canvas_shape)
    except _NoFeasiblePlacement as exc:
        return None, str(exc)
    return _Search(base, prepared, canvas_shape).run()


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def encode_png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_mask(mask: np.ndarray) -> str:
    return encode_png(np.ascontiguousarray(mask, dtype=np.uint8) * np.uint8(255))
