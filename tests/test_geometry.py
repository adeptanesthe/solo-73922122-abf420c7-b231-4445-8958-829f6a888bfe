"""Unit tests for the compositing core."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from app import geometry
from app.errors import InvalidInput
from tests.helpers import mask_b64, png_b64


def make_base(height=20, width=20, instance_masks=()):
    masks = [np.ascontiguousarray(mask) for mask in instance_masks]
    annotated = np.zeros((height, width), dtype=bool)
    for mask in masks:
        annotated |= mask
    return geometry.BaseInfo(
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        masks=tuple(masks),
        annotated=annotated,
        original_pixels=tuple(int(mask.sum()) for mask in masks),
    )


def solid(height, width, value=200):
    return (
        np.full((height, width, 3), value, dtype=np.uint8),
        np.ones((height, width), dtype=bool),
    )


# --------------------------------------------------------------------------- #
# Stacking order and occlusion
# --------------------------------------------------------------------------- #


def test_back_to_front_order_later_patch_wins():
    base = make_base()
    rgb_a, mask_a = solid(4, 4, 10)
    rgb_b, mask_b = solid(4, 4, 90)
    solution, reason = geometry.solve(
        base, [(rgb_a, mask_a, [(0, 0)]), (rgb_b, mask_b, [(2, 2)])]
    )
    assert reason is None
    # Patch 0 keeps 12 of 16 pixels; patch 1 is fully visible.
    assert solution.visible_pixels == [12, 16]
    assert solution.composed[0, 0].tolist() == [10, 10, 10]
    assert solution.composed[2, 2].tolist() == [90, 90, 90]
    assert solution.composed[3, 3].tolist() == [90, 90, 90]


def test_visible_masks_are_disjoint():
    base = make_base(30, 30)
    rgb_a, mask_a = solid(10, 10, 10)
    rgb_b, mask_b = solid(10, 10, 90)
    solution, _ = geometry.solve(
        base, [(rgb_a, mask_a, [(0, 0), (15, 15)]), (rgb_b, mask_b, [(5, 5)])]
    )
    first, second = solution.visible_masks
    assert not np.any(first & second)


def test_reverse_order_would_change_visibility():
    base = make_base()
    rgb_a, mask_a = solid(4, 4, 10)
    rgb_b, mask_b = solid(4, 4, 90)
    forward, _ = geometry.solve(
        base, [(rgb_a, mask_a, [(0, 0)]), (rgb_b, mask_b, [(2, 2)])]
    )
    backward, _ = geometry.solve(
        base, [(rgb_b, mask_b, [(2, 2)]), (rgb_a, mask_a, [(0, 0)])]
    )
    # The overlap is 4 pixels, so the bottom patch is always the occluded one.
    assert forward.visible_pixels == [12, 16]
    assert backward.visible_pixels == [12, 16]


# --------------------------------------------------------------------------- #
# Area floor: clipping and occlusion
# --------------------------------------------------------------------------- #


def test_clipping_exactly_half_is_feasible():
    base = make_base(10, 10)
    rgb, mask = solid(4, 4)
    solution, reason = geometry.solve(base, [(rgb, mask, [(-2, 0)])])
    assert reason is None
    assert solution.positions == [(-2, 0)]
    assert solution.visible_pixels == [8]


def test_clipping_below_half_is_infeasible():
    base = make_base(10, 10)
    rgb, mask = solid(4, 4)
    solution, reason = geometry.solve(base, [(rgb, mask, [(-3, 0)])])
    assert solution is None
    assert "half" in reason


def test_occlusion_below_half_is_infeasible():
    base = make_base(10, 10)
    rgb_a, mask_a = solid(4, 4, 10)
    rgb_b, mask_b = solid(4, 4, 90)
    # Patch 1 covers 12 of patch 0's 16 pixels, leaving 4 < 8.
    solution, reason = geometry.solve(
        base, [(rgb_a, mask_a, [(0, 0)]), (rgb_b, mask_b, [(0, 0)])]
    )
    assert solution is None
    assert "half" in reason


def test_occlusion_exactly_half_is_feasible():
    base = make_base(10, 10)
    rgb_a, mask_a = solid(4, 4, 10)
    rgb_b, mask_b = solid(2, 4, 90)
    # Patch 1 hides 8 of patch 0's 16 pixels: exactly half remains.
    solution, reason = geometry.solve(
        base, [(rgb_a, mask_a, [(0, 0)]), (rgb_b, mask_b, [(0, 0)])]
    )
    assert reason is None
    assert solution.visible_pixels == [8, 8]


def test_odd_area_floor_uses_integer_arithmetic():
    # 5x5 patch (25 px): 12 visible must be infeasible (12 < 12.5), 13 fine.
    base = make_base(10, 10)
    rgb, mask = solid(5, 5)
    infeasible, _ = geometry.solve(base, [(rgb, mask, [(-1, -2)])])  # 4 * 3 = 12
    assert infeasible is None
    feasible, _ = geometry.solve(base, [(rgb, mask, [(-1, -1)])])  # 4 * 4 = 16
    assert feasible is not None


def test_candidate_entirely_outside_canvas_is_reported():
    base = make_base(10, 10)
    rgb, mask = solid(4, 4)
    solution, reason = geometry.solve(base, [(rgb, mask, [(-10, 0), (50, 50)])])
    assert solution is None
    assert "no candidate keeps any mask pixel inside the canvas" in reason


def test_patch_hanging_over_edge_is_allowed():
    base = make_base(10, 10)
    rgb, mask = solid(6, 6)
    solution, reason = geometry.solve(base, [(rgb, mask, [(-1, -1)])])
    assert reason is None
    assert solution.positions == [(-1, -1)]
    assert solution.visible_pixels == [25]


# --------------------------------------------------------------------------- #
# Objective: minimise covered annotated base pixels
# --------------------------------------------------------------------------- #


def test_prefers_placement_without_covering_annotations():
    annotated = np.zeros((10, 10), dtype=bool)
    annotated[0:2, 0:2] = True
    base = make_base(10, 10, [annotated])
    rgb, mask = solid(4, 4)
    solution, reason = geometry.solve(base, [(rgb, mask, [(0, 0), (6, 6)])])
    assert reason is None
    assert solution.positions == [(6, 6)]
    assert solution.overlapped_base_pixels == 0
    assert solution.base_visible_pixels == [4]


def test_coverage_is_a_union_not_a_sum():
    annotated = np.zeros((10, 10), dtype=bool)
    annotated[0:6, 0:6] = True  # 36 annotated pixels
    base = make_base(10, 10, [annotated])
    rgb_a, mask_a = solid(4, 4, 10)
    rgb_b, mask_b = solid(4, 4, 90)
    # The two placements overlap by 4 pixels, so the covered annotation is
    # 16 + 16 - 4 = 28 rather than 32.
    solution, _ = geometry.solve(
        base, [(rgb_a, mask_a, [(0, 0)]), (rgb_b, mask_b, [(2, 2)])]
    )
    assert solution.overlapped_base_pixels == 28


def test_search_weighs_occlusion_effects_not_greedy_choices():
    annotated = np.zeros((10, 10), dtype=bool)
    annotated[0:4, 0:4] = True
    base = make_base(10, 10, [annotated])
    rgb_a, mask_a = solid(4, 4, 10)
    rgb_b, mask_b = solid(4, 4, 90)
    # Greedy: patch 0 avoids the annotation at (5,5); patch 1 then has only
    # (5,5) too, which would hide patch 0 entirely. The exhaustive search must
    # instead trade a small overlap for keeping both instances visible.
    solution, reason = geometry.solve(
        base,
        [
            (rgb_a, mask_a, [(5, 5), (0, 0)]),
            (rgb_b, mask_b, [(5, 5)]),
        ],
    )
    assert reason is None
    assert solution.visible_pixels[0] >= 8
    assert solution.positions[1] == (5, 5)


def test_tie_break_is_deterministic():
    base = make_base()
    rgb, mask = solid(4, 4)
    first, _ = geometry.solve(base, [(rgb, mask, [(0, 0), (1, 1), (6, 6)])])
    second, _ = geometry.solve(base, [(rgb, mask, [(0, 0), (1, 1), (6, 6)])])
    assert first.positions == second.positions
    assert first.positions == [(0, 0)]


# --------------------------------------------------------------------------- #
# Non-rectangular masks
# --------------------------------------------------------------------------- #


def test_non_rectangular_mask_is_respected():
    base = make_base(10, 10)
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:3, 0:3] = True
    rgb = np.full((6, 6, 3), 77, dtype=np.uint8)
    solution, reason = geometry.solve(base, [(rgb, mask, [(0, 0)])])
    assert reason is None
    assert solution.visible_pixels == [9]
    assert solution.composed[0, 0].tolist() == [77, 77, 77]
    assert solution.composed[5, 5].tolist() == [0, 0, 0]


def test_mask_and_image_grid_must_match():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.ones((5, 5), dtype=bool)
    with pytest.raises(InvalidInput, match="sizes differ"):
        geometry._prepare_patches([(rgb, mask, [(0, 0)])], (10, 10))


def test_empty_patch_mask_is_rejected():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    with pytest.raises(InvalidInput, match="no pixels"):
        geometry._prepare_patches([(rgb, mask, [(0, 0)])], (10, 10))


# --------------------------------------------------------------------------- #
# Base annotations
# --------------------------------------------------------------------------- #


def test_overlapping_base_instance_masks_are_rejected():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask_a = np.zeros((8, 8), dtype=bool)
    mask_a[0:4, 0:4] = True
    mask_b = np.zeros((8, 8), dtype=bool)
    mask_b[2:6, 2:6] = True
    with pytest.raises(InvalidInput, match="overlap"):
        geometry.build_base(png_b64(image), [mask_b64(mask_a), mask_b64(mask_b)])


def test_empty_base_mask_is_rejected():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(InvalidInput, match="no pixels"):
        geometry.build_base(png_b64(image), [mask_b64(np.zeros((8, 8), bool))])


def test_base_mask_size_mismatch_is_rejected():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(InvalidInput, match="align"):
        geometry.build_base(png_b64(image), [mask_b64(np.ones((9, 8), bool))])


def test_base_without_annotations_has_zero_cost():
    base = make_base(10, 10)
    rgb, mask = solid(4, 4)
    solution, reason = geometry.solve(base, [(rgb, mask, [(0, 0)])])
    assert reason is None
    assert solution.overlapped_base_pixels == 0


# --------------------------------------------------------------------------- #
# Decoding guard rails
# --------------------------------------------------------------------------- #


def test_invalid_base64_is_rejected():
    with pytest.raises(InvalidInput, match="invalid base64"):
        geometry.decode_b64_image("x", "!!!!not base64!!!!")


def test_non_image_payload_is_rejected():
    payload = base64.b64encode(b"definitely not an image").decode()
    with pytest.raises(InvalidInput, match="unsupported or corrupt"):
        geometry.decode_b64_image("x", payload)


def test_oversized_pixel_grid_is_rejected(monkeypatch):
    monkeypatch.setattr(geometry, "MAX_IMAGE_PIXELS", 100)
    payload = png_b64(np.zeros((60, 60, 3), dtype=np.uint8))
    with pytest.raises(InvalidInput, match="pixel limit"):
        geometry.decode_b64_image("x", payload)


def test_data_url_and_urlsafe_payloads_are_accepted():
    payload = png_b64(np.zeros((4, 4, 3), dtype=np.uint8))
    image = geometry.decode_b64_image("x", "data:image/png;base64," + payload)
    assert image.size == (4, 4)
    urlsafe = payload.replace("+", "-").replace("/", "_")
    assert geometry.decode_b64_image("x", urlsafe).size == (4, 4)
