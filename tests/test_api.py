"""API-level tests, exercised through FastAPI's TestClient."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.helpers import decode_b64_png, mask_b64, png_b64, rgb


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def base_payload(height=16, width=16, instance_masks=()):
    return {
        "image_b64": png_b64(rgb(height, width, 30)),
        "instance_masks": [mask_b64(mask) for mask in instance_masks],
    }


def patch_payload(height, width, candidates, value=200, mask=None):
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    return {
        "image_b64": png_b64(rgb(height, width, value)),
        "mask_b64": mask_b64(mask),
        "candidates": [list(candidate) for candidate in candidates],
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_compose_success_returns_image_masks_and_positions(client):
    body = {
        "base": base_payload(20, 20),
        "patches": [
            patch_payload(6, 6, [(0, 0)], value=10),
            patch_payload(4, 4, [(4, 4)], value=90),
        ],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "ok"
    assert len(data["instances"]) == 2
    assert [item["position"] for item in data["instances"]] == [[0, 0], [4, 4]]
    # The 4x4 patch at (4, 4) hides 4 pixels of the 6x6 patch at (0, 0).
    assert [item["visible_pixels"] for item in data["instances"]] == [32, 16]
    assert data["instances"][0]["original_mask_pixels"] == 36
    assert data["instances"][0]["visible_ratio"] == pytest.approx(32 / 36)
    assert data["overlapped_base_pixels"] == 0

    composed = decode_b64_png(data["composed_image_b64"])
    assert composed.shape == (20, 20, 3)
    assert composed[0, 0].tolist() == [10, 10, 10]
    assert composed[5, 5].tolist() == [90, 90, 90]
    assert composed[15, 15].tolist() == [30, 30, 30]

    masks = [decode_b64_png(item["visible_mask_b64"]) for item in data["instances"]]
    assert masks[0].shape == (20, 20)
    assert int((masks[0] > 0).sum()) == 32
    assert int((masks[1] > 0).sum()) == 16
    assert not np.any((masks[0] > 0) & (masks[1] > 0))


def test_infeasible_report_has_no_image(client):
    body = {
        "base": base_payload(8, 8),
        "patches": [patch_payload(8, 8, [(-5, -5)])],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "infeasible"
    assert "reason" in data
    assert "composed_image_b64" not in data
    assert "instances" not in data


def test_patch_may_hang_over_canvas_edge(client):
    body = {
        "base": base_payload(10, 10),
        "patches": [patch_payload(6, 6, [(-1, -1)])],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 200
    data = response.json()
    instance = data["instances"][0]
    assert instance["position"] == [-1, -1]
    assert instance["visible_pixels"] == 25
    assert instance["visible_ratio"] == pytest.approx(25 / 36)


def test_over_cropped_patch_is_infeasible(client):
    body = {
        "base": base_payload(10, 10),
        "patches": [patch_payload(6, 6, [(-2, -2)])],
    }
    data = client.post("/v1/compose", json=body).json()
    assert data["status"] == "infeasible"


def test_annotation_avoidance_through_api(client):
    annotation = np.zeros((12, 12), dtype=bool)
    annotation[0:4, 0:4] = True
    body = {
        "base": base_payload(12, 12, [annotation]),
        "patches": [patch_payload(4, 4, [(0, 0), (6, 6)])],
    }
    response = client.post("/v1/compose", json=body)
    data = response.json()
    assert data["instances"][0]["position"] == [6, 6]
    assert data["overlapped_base_pixels"] == 0


def test_too_many_patches_is_rejected(client):
    body = {
        "base": base_payload(),
        "patches": [patch_payload(4, 4, [(0, 0)]) for _ in range(5)],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_too_many_candidates_is_rejected(client):
    body = {
        "base": base_payload(),
        "patches": [patch_payload(4, 4, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 422


def test_duplicate_candidates_are_rejected(client):
    body = {
        "base": base_payload(),
        "patches": [patch_payload(4, 4, [(0, 0), (0, 0)])],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 422


def test_mask_size_mismatch_is_rejected(client):
    body = {
        "base": base_payload(10, 10),
        "patches": [
            {
                "image_b64": png_b64(rgb(4, 4)),
                "mask_b64": mask_b64(np.ones((5, 5), dtype=bool)),
                "candidates": [[0, 0]],
            }
        ],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_input"


def test_overlapping_base_masks_are_rejected(client):
    first = np.zeros((10, 10), dtype=bool)
    first[0:4, 0:4] = True
    second = np.zeros((10, 10), dtype=bool)
    second[2:6, 2:6] = True
    body = {
        "base": base_payload(10, 10, [first, second]),
        "patches": [patch_payload(4, 4, [(6, 6)])],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 422
    assert "overlap" in response.json()["detail"]


def test_empty_mask_is_rejected(client):
    body = {
        "base": base_payload(),
        "patches": [patch_payload(4, 4, [(0, 0)], mask=np.zeros((4, 4), dtype=bool))],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 422
    assert "no pixels" in response.json()["detail"]


def test_corrupt_image_is_rejected(client):
    body = {
        "base": {"image_b64": "bm90IGFuIGltYWdl", "mask_b64": "bm90IGFuIGltYWdl"},
        "patches": [patch_payload(4, 4, [(0, 0)])],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_input"


def test_missing_field_is_a_validation_error(client):
    response = client.post("/v1/compose", json={"base": {"image_b64": "x"}})
    assert response.status_code == 422


def test_oversized_body_is_rejected_by_middleware(client, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "MAX_REQUEST_BYTES", 10)
    body = {
        "base": base_payload(),
        "patches": [patch_payload(4, 4, [(0, 0)])],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"


def test_full_four_patch_composition(client):
    body = {
        "base": base_payload(40, 40),
        "patches": [
            patch_payload(8, 8, [(0, 0), (10, 10)], value=10),
            patch_payload(8, 8, [(6, 6), (20, 20)], value=40),
            patch_payload(8, 8, [(12, 12), (30, 30)], value=80),
            patch_payload(8, 8, [(18, 18), (25, 25)], value=120),
        ],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert len(data["instances"]) == 4
    for instance in data["instances"]:
        assert instance["visible_ratio"] >= 0.5


def test_search_node_budget_is_enforced(client, monkeypatch):
    import app.geometry as geometry_module

    monkeypatch.setattr(geometry_module, "MAX_SEARCH_NODES", 1)
    body = {
        "base": base_payload(20, 20),
        "patches": [
            patch_payload(4, 4, [(0, 0), (5, 5)]),
            patch_payload(4, 4, [(10, 10), (12, 12)]),
        ],
    }
    response = client.post("/v1/compose", json=body)
    assert response.status_code == 422
    assert "server limit" in response.json()["detail"]
