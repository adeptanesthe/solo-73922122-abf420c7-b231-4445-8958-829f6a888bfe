"""Request/response schemas.

The wire format is JSON with base64-encoded image payloads. Every image
carries an explicit pixel mask so placements are evaluated on exact masks
rather than on alpha thresholds. Masks must be stored-mode, i.e. exactly the
same width and height as their image; there is no resampling.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import MAX_CANDIDATES, MAX_PATCHES


_B64_HELP = "Base64 image payload. Data URLs and missing padding are accepted."


class BaseImage(BaseModel):
    """Shelf base image with its pre-annotated, non-overlapping instance masks."""

    image_b64: str = Field(..., description=_B64_HELP)
    instance_masks: list[str] = Field(
        default_factory=list,
        description=(
            "One base64 mask per annotated shelf instance, each aligned "
            "pixel-for-pixel with the base image and pairwise non-overlapping. "
            "Their pixels are the annotated pixels the placement search tries "
            "hardest not to cover. May be empty when the base has no annotations."
        ),
    )


class Patch(BaseModel):
    """One commodity cut-out that may be pasted onto the shelf."""

    image_b64: str = Field(..., description=_B64_HELP)
    mask_b64: str = Field(
        ...,
        description=(
            "Base64 8-bit mask aligned pixel-for-pixel with the patch image. "
            "Non-zero samples are the product's pixels."
        ),
    )
    candidates: list[tuple[int, int]] = Field(
        ...,
        min_length=1,
        max_length=MAX_CANDIDATES,
        description=(
            "Top-left (x, y) pixel positions for this patch in canvas "
            "coordinates, at most five. A patch may hang over any canvas edge."
        ),
    )

    @field_validator("candidates")
    @classmethod
    def _unique_candidates(cls, value: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(set(value)) != len(value):
            raise ValueError("candidate coordinates must be unique per patch")
        return value


class ComposeRequest(BaseModel):
    """Shelf base image plus up to four product patches, in stacking order."""

    base: BaseImage = Field(...)
    patches: list[Patch] = Field(
        ...,
        min_length=1,
        max_length=MAX_PATCHES,
        description=(
            "Patches in back-to-front stacking order: index 0 is the bottom-most "
            "layer and the last patch is drawn last, covering the others."
        ),
    )


class InstanceResult(BaseModel):
    """One pasted instance in the composed image."""

    patch_index: int = Field(..., ge=0, description="Index into the request's patches.")
    position: tuple[int, int] = Field(
        ..., description="Chosen top-left (x, y) of the patch in canvas coordinates."
    )
    visible_pixels: int = Field(
        ..., ge=0, description="Unoccluded, in-canvas pixels of this instance."
    )
    original_mask_pixels: int = Field(
        ..., ge=1, description="Area of the patch mask as submitted."
    )
    visible_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="visible_pixels / original_mask_pixels, always at least 0.5.",
    )
    visible_mask_b64: str = Field(
        ...,
        description=(
            "Base64 canvas-sized 8-bit PNG: 255 where this instance is visible in "
            "the final composite, 0 elsewhere. Disjoint across instances and "
            "suitable as a training target."
        ),
    )


class BaseInstanceSummary(BaseModel):
    """How much of a pre-annotated base instance remains visible."""

    base_index: int = Field(..., ge=0, description="Index into base.instance_masks.")
    visible_pixels: int = Field(..., ge=0)
    original_mask_pixels: int = Field(..., ge=1)
    visible_mask_b64: str = Field(
        ...,
        description=(
            "Base64 canvas-sized 8-bit PNG: 255 where this base instance stays "
            "visible after pasting, 0 elsewhere. Disjoint from the patch masks."
        ),
    )


class ComposeResponse(BaseModel):
    status: Literal["ok"] = "ok"
    composed_image_b64: str = Field(..., description="Base64 PNG of the composite.")
    instances: list[InstanceResult] = Field(
        ...,
        description=(
            "One entry per patch, in stacking order. Visible masks are pairwise "
            "disjoint, so they can be used directly as instance-segmentation targets."
        ),
    )
    base_instances: list[BaseInstanceSummary] = Field(
        ...,
        description="Visibility of the pre-annotated base instances, for auditing.",
    )
    overlapped_base_pixels: int = Field(
        ...,
        ge=0,
        description="Annotated base pixels covered by at least one pasted instance.",
    )
    evaluated_combinations: int = Field(
        ...,
        ge=0,
        description="Candidate combinations explored before choosing this placement.",
    )


class InfeasibleResponse(BaseModel):
    status: Literal["infeasible"] = "infeasible"
    reason: str = Field(..., description="Why no arrangement satisfies the constraints.")


class ErrorResponse(BaseModel):
    error: str
    detail: object
