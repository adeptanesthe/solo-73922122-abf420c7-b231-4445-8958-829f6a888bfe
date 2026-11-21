"""Generate a synthetic request, call the running API, and save the result.

Usage
-----
1. Start the service (any of these):
     docker compose up --build                       # port 8000
     uvicorn app.main:app --port 8000                # local Python

2. Run this script:
     python examples/demo_request.py                     # calls http://localhost:8000
     python examples/demo_request.py --base-url http://host:8000 --out examples/out
     python examples/demo_request.py --scene reselect    # base-visibility demo

It writes ``request.json`` (ready to POST with curl), plus the composed image
and the per-instance visible masks, and prints the chosen placements.

Scenes
------
``shelf`` (default)
    A 320x240 shelf with two annotated regions; four product patches with
    non-rectangular masks, each offering several candidate top-left positions
    - including ones that hang over the canvas edge.

``reselect``
    A scene that demonstrates the base-instance visibility floor. The single
    patch has two attractive candidates: one completely hides a tiny annotated
    jar while touching nothing else, the other clips a corner off a large
    annotated box. The first covers fewer annotated pixels (256 vs 1152), but
    it would leave the jar 0/256 visible, so it must be rejected and the
    service must pick the second.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def b64_png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def shelf_base() -> tuple[np.ndarray, list[np.ndarray]]:
    height, width = 240, 320
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # Shelf boards and a back panel, so the composite is visually checkable.
    image[:, :] = (58, 62, 70)
    for y in (90, 180, 235):
        image[max(0, y - 4) : y, :] = (96, 92, 84)

    # Two pre-annotated shelf instances, drawn into the base image so the
    # "don't cover the annotation" objective is visible in the output.
    jar = np.zeros((height, width), dtype=bool)
    jar[30:78, 30:78] = True
    image[jar] = (150, 150, 140)

    box = np.zeros((height, width), dtype=bool)
    box[126:174, 236:300] = True
    image[box] = (128, 104, 84)
    return image, [jar, box]


def product(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Build a patch image and mask; masks are non-rectangular on purpose."""

    if kind == "can":
        image = np.full((56, 34, 3), (200, 60, 50), dtype=np.uint8)
        image[6:50, 6:28] = (228, 96, 78)  # label band
        mask = np.zeros((56, 34), dtype=bool)
        mask[4:52, 5:29] = True  # rounded-ish profile
        return image, mask

    if kind == "bottle":
        image = np.full((72, 30, 3), (46, 132, 88), dtype=np.uint8)
        image[10:64, 8:22] = (78, 176, 118)
        mask = np.zeros((72, 30), dtype=bool)
        mask[2:12, 12:18] = True  # neck
        mask[12:70, 5:25] = True  # body
        return image, mask

    if kind == "box":
        image = np.full((48, 60, 3), (190, 150, 60), dtype=np.uint8)
        image[4:44, 4:56] = (214, 176, 88)
        mask = np.zeros((48, 60), dtype=bool)
        mask[3:45, 3:57] = True
        return image, mask

    if kind == "bag":
        image = np.full((52, 44, 3), (100, 110, 190), dtype=np.uint8)
        image[8:44, 6:38] = (132, 142, 214)
        mask = np.zeros((52, 44), dtype=bool)
        mask[4:48, 4:40] = True
        mask[0:6, 14:30] = True  # crimped top
        return image, mask

    raise ValueError(kind)


def build_request() -> dict:
    base_image, base_masks = shelf_base()
    specs = [
        ("can", [(-8, 120), (120, 118), (150, 116), (180, 116), (250, 120)]),
        ("bottle", [(90, 60), (60, 56), (276, 50), (140, 58)]),
        ("box", [(120, 150), (100, 148), (160, 150), (250, 150), (300, 140)]),
        ("bag", [(200, 60), (40, 190), (150, 188), (280, 60)]),
    ]
    return _request_from(base_image, base_masks, specs)


def reselect_base() -> tuple[np.ndarray, list[np.ndarray]]:
    """A 320x240 shelf with a tiny jar and a large annotated box."""

    height, width = 240, 320
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (58, 62, 70)
    for y in (90, 180, 235):
        image[max(0, y - 4) : y, :] = (96, 92, 84)

    jar = np.zeros((height, width), dtype=bool)
    jar[16:32, 16:32] = True  # 16x16 = 256 px, top-left corner
    image[jar] = (150, 150, 140)

    box = np.zeros((height, width), dtype=bool)
    box[100:164, 224:300] = True  # 64x76 = 4864 px, middle-right
    image[box] = (128, 104, 84)
    return image, [jar, box]


def reselect_request() -> dict:
    """One patch, two candidates: hide the jar (cheapest) or clip the box.

    Candidate (0, 0) covers the 256-pixel jar completely (and no other
    annotation), so it has the lower total coverage. Candidate (228, 124)
    hides 38x44 = 1672 pixels of the 4864-pixel box, leaving 65.6 % visible.
    The jar would keep 0/256, so the cheaper placement must be excluded and
    the service must choose (228, 124) even though it covers more pixels.
    """

    image = np.full((48, 48, 3), (200, 80, 60), dtype=np.uint8)
    image[6:42, 6:42] = (228, 110, 88)
    mask = np.zeros((48, 48), dtype=bool)
    mask[2:46, 2:46] = True  # 44x44 = 1936 px

    base_image, base_masks = reselect_base()
    return _request_from(
        base_image,
        base_masks,
        [("reselect", [(0, 0), (228, 124)])],
        overrides={"reselect": (image, mask)},
    )


def _request_from(base_image, base_masks, specs, overrides=None) -> dict:
    overrides = overrides or {}
    patches = []
    for kind, candidates in specs:
        image, mask = overrides.get(kind) or product(kind)
        patches.append(
            {
                "image_b64": b64_png(image),
                "mask_b64": b64_png(mask.astype(np.uint8) * 255),
                "candidates": [list(candidate) for candidate in candidates],
            }
        )

    return {
        "base": {
            "image_b64": b64_png(base_image),
            # The API stores only instance masks; the annotated pixels are their
            # union. Non-zero = annotated pixel to avoid covering.
            "instance_masks": [b64_png(m.astype(np.uint8) * 255) for m in base_masks],
        },
        "patches": patches,
    }

def save_response(response: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if response.get("status") != "ok":
        (out_dir / "response.json").write_text(json.dumps(response, indent=2))
        print("status:", response.get("status"), "-", response.get("reason"))
        return

    def decode(payload: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(payload)))

    decode(response["composed_image_b64"]).save(out_dir / "composed.png")
    for index, instance in enumerate(response["instances"]):
        decode(instance["visible_mask_b64"]).save(out_dir / f"instance_{index}_visible.png")
    for index, item in enumerate(response.get("base_instances", [])):
        decode(item["visible_mask_b64"]).save(out_dir / f"base_{index}_visible.png")
    (out_dir / "response.json").write_text(json.dumps(response, indent=2))

    print(f"wrote {out_dir / 'composed.png'}")
    for instance in response["instances"]:
        print(
            f"  patch {instance['patch_index']}: position={tuple(instance['position'])} "
            f"visible={instance['visible_pixels']}/{instance['original_mask_pixels']} "
            f"({instance['visible_ratio']:.1%})"
        )
    for item in response.get("base_instances", []):
        ratio = item["visible_pixels"] / item["original_mask_pixels"]
        print(
            f"  base {item['base_index']}: visible="
            f"{item['visible_pixels']}/{item['original_mask_pixels']} ({ratio:.1%})"
        )
    print("  overlapped annotated base pixels:", response["overlapped_base_pixels"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument(
        "--scene",
        choices=("shelf", "reselect"),
        default="shelf",
        help="'shelf' is the four-patch scene; 'reselect' demonstrates the "
        "base-instance visibility floor changing the chosen placement.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only write request.json; do not call the API.",
    )
    args = parser.parse_args(argv)

    request = reselect_request() if args.scene == "reselect" else build_request()
    args.out.mkdir(parents=True, exist_ok=True)
    request_path = args.out / "request.json"
    request_path.write_text(json.dumps(request))
    print(f"wrote {request_path} ({request_path.stat().st_size} bytes)")
    if args.offline:
        return 0

    body = json.dumps(request).encode("utf-8")
    http_request = urllib.request.Request(
        f"{args.base_url}/v1/compose",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=120) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read()[:2000], file=sys.stderr)
        return 1
    save_response(payload, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
