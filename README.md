# Shelf Instance Compositor

FastAPI service that pastes up to four masked commodity patches onto a shelf
base image to generate occlusion-rich instance-segmentation training samples.

Given a shelf base image (with optional non-overlapping instance masks for
already-annotated products) and up to four product cut-outs, each with a mask
and up to five candidate top-left positions, the service:

1. tries every candidate combination (at most `5^4 = 625`),
2. discards combinations where any instance would keep less than **half of its
   original mask area** after clipping to the canvas and being covered by
   later patches,
3. among the survivors, picks the one covering the **fewest annotated base
   pixels** (union, not sum; ties prefer the largest visibility margin, then
   candidate order),
4. returns the composite, each instance's canvas-sized visible mask, and the
   chosen position.

If no combination is feasible, it returns `{"status": "infeasible", "reason": ...}`
and no image.

## API

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET`  | `/health` | Liveness probe (`{"status": "ok"}`). |
| `POST` | `/v1/compose` | Composite the patches. |
| `GET`  | `/docs` | Interactive OpenAPI docs. |

### Request body

```jsonc
{
  "base": {
    "image_b64": "<base64 PNG/JPEG>",
    // Optional. One mask per pre-annotated shelf instance, same pixel grid as
    // the base, pairwise non-overlapping. Their union is the annotated pixel
    // set the search avoids covering. May be empty.
    "instance_masks": ["<base64 8-bit mask PNG>", "..."]
  },
  "patches": [
    {
      "image_b64": "<base64 PNG/JPEG>",
      "mask_b64": "<base64 8-bit mask PNG, same size as the patch image>",
      // At most five top-left (x, y) positions in canvas coordinates.
      // Negative or overhanging positions are allowed: the patch may stick
      // out of any canvas edge.
      "candidates": [[0, 0], [100, 40], [-12, 300]]
    }
    // ...up to four patches, in back-to-front stacking order
  ]
}
```

Masks are non-zero-equals-instance, must align with their image (no
resampling), and are sent as base64. Data URLs, URL-safe base64 and missing
padding are accepted.

### Response: feasible

```jsonc
{
  "status": "ok",
  "composed_image_b64": "<base64 PNG>",
  "instances": [                       // one per patch, in stacking order
    {
      "patch_index": 0,
      "position": [120, 118],          // chosen top-left of the patch
      "visible_pixels": 1152,
      "original_mask_pixels": 1152,
      "visible_ratio": 1.0,            // always >= 0.5
      "visible_mask_b64": "<base64 canvas-sized 8-bit PNG, 255 = visible>"
    }
  ],
  "base_instances": [                  // visibility of each base annotation
    {"base_index": 0, "visible_pixels": 2304, "original_mask_pixels": 2304,
     "visible_mask_b64": "<base64 PNG>"}
  ],
  "overlapped_base_pixels": 0,         // annotated pixels under pasted instances
  "evaluated_combinations": 300
}
```

The `visible_mask_b64` masks of different instances are pairwise disjoint and
already include both cropping and occlusion, so they can be used directly as
segmentation targets.

### Response: no feasible placement

```jsonc
{"status": "infeasible", "reason": "no candidate combination keeps at least half of every product instance's original mask area"}
```

### Errors

| Status | Meaning |
| ------ | ------- |
| `400`  | Malformed `Content-Length`. |
| `413`  | Body above `MAX_REQUEST_BYTES`. |
| `422`  | Invalid input: bad base64/image, size mismatch, empty mask, overlapping base masks, all candidates outside the canvas, or a search space above the node budget. Body: `{"error": "invalid_input" \| "invalid_request", "detail": ...}`. |

## Run with Docker Compose

```bash
docker compose up --build          # http://localhost:8000
curl http://localhost:8000/health
```

`HOST_PORT=9000 docker compose up --build` publishes on another port.
Environment overrides (see `app/config.py`):

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `MAX_IMAGE_BYTES` | `4194304` | Max decoded bytes per image/mask. |
| `MAX_IMAGE_PIXELS` | `4194304` | Max pixels per image/mask (2048×2048). |
| `MAX_REQUEST_BYTES` | `33554432` | Max request body. |
| `MAX_BASE_INSTANCES` | `64` | Max base annotation masks. |
| `MIN_VISIBLE_RATIO` | `1/2` | Area floor for pasted instances. |

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000
pytest -q
```

## Callable example

`examples/demo_request.py` builds a synthetic 320×240 shelf (two annotated
base instances) plus four non-rectangular product patches with several
candidate positions each (including edge-overhanging ones), POSTs it to the
service, and writes:

- `examples/out/request.json` — the exact request (replayable with `curl`)
- `examples/out/composed.png` — the composite
- `examples/out/instance_<i>_visible.png` — visible mask per instance
- `examples/out/response.json` — the full response

```bash
docker compose up --build -d
pip install numpy pillow
python examples/demo_request.py                       # default localhost:8000
python examples/demo_request.py --base-url http://127.0.0.1:8000 --out examples/out
python examples/demo_request.py --offline             # only write request.json
```

Equivalent `curl` using the generated request:

```bash
curl -sS -X POST http://localhost:8000/v1/compose \
     -H 'Content-Type: application/json' \
     --data @examples/out/request.json > examples/out/response.json
```

## Algorithm

The search is exhaustive because the objective is not separable: a later patch
can hide an earlier one, which changes both the feasibility of the earlier
patch and the set of covered annotated pixels. Greedy per-patch choices can be
suboptimal. The DFS visits patches top-most first while maintaining the union
of already-placed masks, so:

- the visible count of the patch being placed is final at that node (only
  later patches can cover it), enabling immediate pruning below the 50 % floor;
- the incremental covered-annotation count is exact and additive.

At most 625 leaves are visited; per-node work is proportional to patch area.
Ties are broken by the largest minimum visible-area margin, then by candidate
order, making the output deterministic.

## Layout

```
app/
  main.py        FastAPI app, size limiting, error mapping
  schemas.py     Pydantic request/response models and limits
  geometry.py    decoding, exact placement search, compositing, encoding
  config.py      limits and the area-floor ratio (env overridable)
  errors.py      shared exception type
tests/
  test_geometry.py   unit tests for masking, clipping, occlusion, objective
  test_api.py        endpoint tests via TestClient, including error paths
  test_reference.py  differential tests against a naive brute-force reference
examples/demo_request.py
Dockerfile, docker-compose.yml, requirements*.txt
```

## Limitations

- Patch images are pasted upright; no rotation or scaling is performed.
- Only the advertised limits are enforced; images above 2048×2048 are rejected.
- The service is stateless and does not persist requests or outputs.

## Verification status

- `pytest -q` → 342 passed.
- Differential fuzzing: 300 committed random cases plus 6,000 extra seeds run
  manually, comparing the optimised solver against an independent naive
  brute-force reference on position choice, feasibility and rendered masks;
  zero mismatches.
- Live smoke test performed with `uvicorn`: `GET /health`, `POST /v1/compose`
  (4 patches, 300 combinations), `curl --data @examples/out/request.json`,
  OpenAPI/docs endpoints, oversized `Content-Length` → 413, oversized chunked
  upload → 413, malformed/overlapping/empty masks → 422.
- Verified worst-case timing: 4 patches × 5 candidates on a 2048×2048 canvas
  with ~900×900 patches completes in ~1.1 s.

Not verified here: `docker compose up --build` itself (Docker is not installed
in this environment); the Dockerfile and compose file were reviewed by hand
and their Python commands were run directly.
