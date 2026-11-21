"""Example request/response captured from a live run, plus how to replay it.

Run ``python examples/demo_request.py --offline`` to regenerate
``request.json`` without a server, then POST it with curl:

    curl -sS -X POST http://localhost:8000/v1/compose \
         -H 'Content-Type: application/json' \
         --data @examples/out/request.json

``examples/reselect_request.json`` is a committed one-patch request that
demonstrates the base-instance visibility floor changing the chosen placement:

    curl -sS -X POST http://localhost:8000/v1/compose \
         -H 'Content-Type: application/json' \
         --data @examples/reselect_request.json

    -> position [228, 124], overlapped_base_pixels 1672,
       base_instances[0] 256/256 visible (the tiny jar that would otherwise
       be fully hidden by the cheaper candidate [0, 0]),
       base_instances[1] 3192/4864 visible (65.6 %).

Response shape (trimmed; masks and the image are elided):

    {
      "status": "ok",
      "instances": [
        {"patch_index": 0, "position": [120, 118], "visible_pixels": 1152,
         "original_mask_pixels": 1152, "visible_ratio": 1.0,
         "visible_mask_b64": "..."}
      ],
      "base_instances": [
        {"base_index": 0, "visible_pixels": 2304,
         "original_mask_pixels": 2304, "visible_mask_b64": "..."}
      ],
      "overlapped_base_pixels": 0,
      "evaluated_combinations": 300
    }

The infeasible branch returns ``{"status": "infeasible", "reason": "..."}``
with no image, for example when every candidate clips a patch below half of
its mask area or would hide more than half of an annotated base instance.
"""
