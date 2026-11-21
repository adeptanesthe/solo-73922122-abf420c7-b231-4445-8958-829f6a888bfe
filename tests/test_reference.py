"""Differential test: the optimised solver vs. a naive reference implementation.

The reference is deliberately slow and obvious: it materialises a full canvas
label map per candidate combination and recomputes every visible mask. It is
only used on tiny canvases with small candidate sets.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from app import geometry
from app.config import (
    PATCH_MIN_VISIBLE_DENOMINATOR,
    PATCH_MIN_VISIBLE_NUMERATOR,
)


def reference_solve(base, patches):
    """Brute force: return (best positions, best visible masks, cost) or None."""

    height, width = base.rgb.shape[:2]
    options = []
    for rgb, mask, candidates in patches:
        usable = geometry.usable_candidates((height, width), mask.shape, candidates)
        if not usable:
            return None
        options.append(usable)
    if any(not option for option in options):
        return None

    best_key = None
    best = None
    for combo in itertools.product(*options):
        labels = np.zeros((height, width), dtype=np.int32)
        for index, ((rgb, mask, _), (x, y)) in enumerate(zip(patches, combo)):
            h, w = mask.shape
            box = geometry.clip_box((height, width), (h, w), x, y)
            if box is None:
                continue
            r0, r1, c0, c1 = box
            view = labels[r0:r1, c0:c1]
            view[mask[r0 - y : r1 - y, c0 - x : c1 - x]] = index + 1
        visible = [(labels == index + 1).sum() for index in range(len(patches))]
        if any(
            PATCH_MIN_VISIBLE_DENOMINATOR * int(count)
            < PATCH_MIN_VISIBLE_NUMERATOR * int(mask.sum())
            for count, (_, mask, _) in zip(visible, patches)
        ):
            continue
        cost = int(np.count_nonzero((labels != 0) & base.annotated))
        slack = min(
            PATCH_MIN_VISIBLE_DENOMINATOR * int(count)
            - PATCH_MIN_VISIBLE_NUMERATOR * int(mask.sum())
            for count, (_, mask, _) in zip(visible, patches)
        )
        # Tie-break on candidate *indices* in request order, exactly like the
        # production solver, so the differential test compares like with like.
        indices = tuple(options[i].index(position) for i, position in enumerate(combo))
        key = (cost, -slack, indices)
        if best_key is None or key < best_key:
            best_key = key
            best = combo
    if best is None:
        return None
    return best


def random_case(rng: random.Random):
    height = rng.randint(5, 9)
    width = rng.randint(5, 9)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    n_masks = rng.randint(0, 2)
    masks = []
    annotated = np.zeros((height, width), dtype=bool)
    for _ in range(n_masks):
        mask = np.zeros((height, width), dtype=bool)
        r = rng.randint(0, height - 1)
        c = rng.randint(0, width - 1)
        h = rng.randint(1, height - r)
        w = rng.randint(1, width - c)
        mask[r : r + h, c : c + w] = True
        mask &= ~annotated  # keep base masks disjoint
        if mask.any():
            masks.append(mask)
            annotated |= mask
    base = geometry.BaseInfo(
        rgb=rgb,
        masks=tuple(masks),
        annotated=annotated,
        original_pixels=tuple(int(m.sum()) for m in masks),
    )

    patches = []
    for _ in range(rng.randint(1, 3)):
        ph = rng.randint(1, 5)
        pw = rng.randint(1, 5)
        pmask = np.zeros((ph, pw), dtype=bool)
        # Random blob, possibly not rectangular.
        for _ in range(ph * pw):
            if rng.random() < 0.75:
                pmask[rng.randrange(ph), rng.randrange(pw)] = True
        if not pmask.any():
            pmask[0, 0] = True
        prgb = np.full((ph, pw, 3), rng.randint(0, 255), dtype=np.uint8)
        candidates = [
            (rng.randint(-pw, width), rng.randint(-ph, height))
            for _ in range(rng.randint(1, 4))
        ]
        candidates = list(dict.fromkeys(candidates))
        patches.append((prgb, pmask, candidates))
    return base, patches


@pytest.mark.parametrize("seed", range(300))
def test_matches_naive_reference(seed):
    rng = random.Random(seed)
    base, patches = random_case(rng)
    expected = reference_solve(base, patches)
    solution, reason = geometry.solve(base, patches)

    if expected is None:
        assert solution is None, f"seed {seed}: expected infeasible, got {solution.positions}"
        assert reason is not None
        return

    assert solution is not None, f"seed {seed}: expected {expected}, got infeasible ({reason})"
    assert solution.positions == list(expected), f"seed {seed}"

    # Independently recompute the visible masks and counters for the chosen
    # placement, using the reference's full-canvas semantics.
    height, width = base.rgb.shape[:2]
    labels = np.zeros((height, width), dtype=np.int32)
    for index, ((_, mask, _), (x, y)) in enumerate(zip(patches, solution.positions)):
        h, w = mask.shape
        r0, r1, c0, c1 = max(0, y), min(height, y + h), max(0, x), min(width, x + w)
        labels[r0:r1, c0:c1][mask[r0 - y : r1 - y, c0 - x : c1 - x]] = index + 1
    for index in range(len(patches)):
        assert np.array_equal(solution.visible_masks[index], labels == index + 1)
    assert solution.overlapped_base_pixels == int(
        np.count_nonzero((labels != 0) & base.annotated)
    )
