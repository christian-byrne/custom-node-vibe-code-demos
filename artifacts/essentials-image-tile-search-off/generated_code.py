"""Core, framework-free tiling math and tensor ops for the GridTiles pack.

Everything here operates on torch tensors in BHWC layout and is written so the
ComfyUI node wrappers in ``nodes.py`` stay thin. Keeping the math separate also
lets the self-test exercise it directly.

Design contract (from the package specification):

* An axis of length ``dim`` is split into ``count`` tiles of a single uniform
  ``extent`` with a single uniform ``stride``, where ``stride = extent - overlap``
  and the tiles cover the axis exactly::

      extent * count - overlap * (count - 1) == dim

  Solving for the extent::

      extent = (dim + overlap * (count - 1)) / count

  which must be a positive integer, and for ``count > 1`` the stride must be
  >= 1 (otherwise tiles would not advance). Any setting that does not yield such
  an extent is rejected with a clear, named error; nothing is silently cropped,
  padded, reordered or coerced.

* Overlap is specified independently per axis: ``overlap_x`` (horizontal, the
  columns/width axis) and ``overlap_y`` (vertical, the rows/height axis). Each is
  a non-negative integer, solved against its own axis.

* Tiles are emitted in source-batch-major, then row-major order: for each source
  image (in batch order), for each row (top to bottom), for each column (left to
  right).

Reconstruction (second-turn change):

* ``merge_image_tiles`` blends overlap pixels with separable per-axis edge
  weights applied *only across the overlap bands* (interior pixels keep weight
  1.0). The ``blend`` selects the ramp shape across a band:

    - ``linear``  - a monotonic linear ramp,  w(k) = (k+1)/(O+1)
    - ``cosine``  - the corresponding half-cosine ramp,
                    w(k) = 0.5 - 0.5*cos(pi*(k+1)/(O+1))

  A tile's leading band (where it overlaps the previous tile) ramps up; its
  trailing band (where it overlaps the next tile) is the mirror of that ramp.
  Both shapes form a partition of unity across a two-tile band, so with
  identical source values the weighted average returns the original value.
  Every contribution is normalized by the accumulated weight at each pixel, any
  zero-weight pixel is rejected, and when both overlaps are 0 a direct-placement
  fast path makes reconstruction byte-exact.
"""

from __future__ import annotations

import math
from typing import Tuple


class GridTilesError(ValueError):
    """Base class for all bounded, named validation errors in this pack."""


class InvalidGridError(GridTilesError):
    """rows/columns are not valid positive integers."""


class InvalidOverlapError(GridTilesError):
    """An overlap is negative, or too large for a valid tiling to exist."""


class InvalidImageError(GridTilesError):
    """The input tensor is not a BHWC image, or a dimension is non-positive."""


class NonIntegralExtentError(GridTilesError):
    """No positive integral tile extent satisfies the exact-cover equation."""


class TileCountError(GridTilesError):
    """The number of supplied tiles differs from the expected tile count."""


class TileDimensionError(GridTilesError):
    """A supplied tile does not have the expected uniform extent."""


class MetadataError(GridTilesError):
    """Reconstruction metadata (width/height/batch size) is invalid."""


class InvalidBlendError(GridTilesError):
    """The overlap blend mode is not one of the supported values."""


class ZeroWeightError(GridTilesError):
    """A reconstruction pixel received zero accumulated weight (uncovered)."""


_BLEND_MODES = ("linear", "cosine")


def solve_axis(dim: int, count: int, overlap: int, axis_name: str) -> Tuple[int, int]:
    """Return ``(extent, stride)`` for one axis or raise a named error.

    ``axis_name`` is used only for human-readable error messages
    (e.g. "rows/height" or "columns/width").
    """
    if not isinstance(count, int) or count < 1:
        raise InvalidGridError(
            f"{axis_name}: tile count must be an integer >= 1, got {count!r}"
        )
    if not isinstance(overlap, int) or overlap < 0:
        raise InvalidOverlapError(
            f"{axis_name}: overlap must be a non-negative integer, got {overlap!r}"
        )
    if not isinstance(dim, int) or dim < 1:
        raise InvalidImageError(
            f"{axis_name}: image dimension must be an integer >= 1, got {dim!r}"
        )

    numerator = dim + overlap * (count - 1)
    if numerator % count != 0:
        raise NonIntegralExtentError(
            f"{axis_name}: no positive integral tile extent exists for "
            f"dimension={dim}, count={count}, overlap={overlap} "
            f"((dim + overlap*(count-1)) = {numerator} is not divisible by {count})"
        )
    extent = numerator // count
    if extent < 1:
        raise NonIntegralExtentError(
            f"{axis_name}: computed tile extent {extent} is not positive for "
            f"dimension={dim}, count={count}, overlap={overlap}"
        )
    if extent > dim:
        # Can only happen when overlap exceeds the dimension.
        raise InvalidOverlapError(
            f"{axis_name}: tile extent {extent} exceeds dimension {dim}; "
            f"overlap={overlap} is too large for count={count}"
        )
    stride = extent - overlap
    if count > 1 and stride < 1:
        raise InvalidOverlapError(
            f"{axis_name}: overlap={overlap} is too large for extent={extent} "
            f"(stride={stride} must be >= 1 when count>1)"
        )
    return extent, stride


def _check_bhwc(image, name: str = "image"):
    """Validate a BHWC torch tensor and return (B, H, W, C) as python ints."""
    if not hasattr(image, "shape") or not hasattr(image, "ndim"):
        raise InvalidImageError(f"{name}: expected a torch tensor, got {type(image)!r}")
    if image.ndim != 4:
        raise InvalidImageError(
            f"{name}: expected a 4-D BHWC tensor, got {image.ndim}-D shape "
            f"{tuple(image.shape)}"
        )
    b, h, w, c = (int(x) for x in image.shape)
    if b < 1 or h < 1 or w < 1 or c < 1:
        raise InvalidImageError(
            f"{name}: every BHWC dimension must be >= 1, got {(b, h, w, c)}"
        )
    return b, h, w, c


def _validate_blend(blend) -> str:
    if blend not in _BLEND_MODES:
        raise InvalidBlendError(
            f"blend must be one of {_BLEND_MODES}, got {blend!r}"
        )
    return blend


def _edge_ramp(overlap: int, blend: str):
    """Monotonically increasing ramp of length ``overlap`` in (0, 1).

    Used for a tile's *leading* band (ramp up). The trailing band is the mirror
    (``torch.flip``) of this. Both ``linear`` and ``cosine`` are built from the
    same fractions ``(k+1)/(overlap+1)`` so no weight is ever exactly 0 or 1 and
    a two-tile band forms a partition of unity.
    """
    import torch

    k = torch.arange(1, overlap + 1, dtype=torch.float32)
    frac = k / (overlap + 1.0)
    if blend == "linear":
        return frac
    # cosine: half-cosine S-curve matching the linear fractions
    return 0.5 - 0.5 * torch.cos(math.pi * frac)


def _axis_weights(extent: int, overlap: int, count: int, index: int, blend: str):
    """1-D float32 edge-weight vector of length ``extent`` for one tile.

    Weight is 1.0 across the interior. The leading ``overlap`` pixels ramp up
    only when this tile is not the first along the axis; the trailing ``overlap``
    pixels ramp down (mirror of the leading ramp) only when it is not the last.
    Bands that would meet inside a very short tile are combined by multiplication,
    which keeps every weight strictly positive.
    """
    import torch

    w = torch.ones(extent, dtype=torch.float32)
    if overlap <= 0:
        return w
    ramp = _edge_ramp(overlap, blend)  # increasing, length == overlap
    if index > 0:
        w[:overlap] = w[:overlap] * ramp
    if index < count - 1:
        w[extent - overlap:] = w[extent - overlap:] * torch.flip(ramp, (0,))
    return w


def split_image_batch(image, rows: int, columns: int, overlap_x: int, overlap_y: int):
    """Split a BHWC IMAGE batch into tiles.

    Returns ``(tiles, orig_width, orig_height, source_batch_size)`` where
    ``tiles`` is a BHWC tensor of equal-sized tiles in source-batch-major,
    row-major order. No cropping/padding/reordering/coercion: dtype, device,
    value range and layout are inherited straight from slicing the source.

    ``overlap_x`` applies to the columns/width axis, ``overlap_y`` to the
    rows/height axis.
    """
    import torch  # local import keeps the module importable without torch

    b, h, w, c = _check_bhwc(image)
    extent_h, stride_h = solve_axis(h, rows, overlap_y, "rows/height")
    extent_w, stride_w = solve_axis(w, columns, overlap_x, "columns/width")

    tiles = []
    for bi in range(b):
        for r in range(rows):
            y0 = r * stride_h
            for col in range(columns):
                x0 = col * stride_w
                # Pure slice: preserves dtype, device and exact values.
                tile = image[bi, y0 : y0 + extent_h, x0 : x0 + extent_w, :]
                tiles.append(tile)

    out = torch.stack(tiles, dim=0)
    # torch.stack makes a fresh contiguous tensor; keep dtype/device identical.
    out = out.to(dtype=image.dtype, device=image.device)
    return out, w, h, b


def merge_image_tiles(
    tiles,
    orig_width: int,
    orig_height: int,
    source_batch_size: int,
    rows: int,
    columns: int,
    overlap_x: int,
    overlap_y: int,
    blend: str = "cosine",
):
    """Reconstruct the original BHWC batch from tiles.

    Validates the blend mode, reconstruction metadata, the exact expected tile
    count and per-tile dimensions, then blends overlap pixels with separable
    per-axis edge weights normalized by the accumulated weight at each pixel.

    With ``overlap_x == 0 and overlap_y == 0`` a direct-placement fast path is
    used, so the result is byte-exact with the original.
    """
    import torch

    n, th, tw, c = _check_bhwc(tiles, name="tiles")
    blend = _validate_blend(blend)

    if not isinstance(source_batch_size, int) or source_batch_size < 1:
        raise MetadataError(
            f"source_batch_size must be an integer >= 1, got {source_batch_size!r}"
        )
    if not isinstance(orig_width, int) or orig_width < 1:
        raise MetadataError(f"orig_width must be an integer >= 1, got {orig_width!r}")
    if not isinstance(orig_height, int) or orig_height < 1:
        raise MetadataError(f"orig_height must be an integer >= 1, got {orig_height!r}")

    extent_h, stride_h = solve_axis(orig_height, rows, overlap_y, "rows/height")
    extent_w, stride_w = solve_axis(orig_width, columns, overlap_x, "columns/width")

    expected_count = source_batch_size * rows * columns
    if n != expected_count:
        raise TileCountError(
            f"expected {expected_count} tiles "
            f"(source_batch_size={source_batch_size} * rows={rows} * columns={columns}), "
            f"got {n}"
        )
    if th != extent_h or tw != extent_w:
        raise TileDimensionError(
            f"expected tiles of extent (H={extent_h}, W={extent_w}), "
            f"got (H={th}, W={tw})"
        )

    # ---- no-overlap fast path: direct placement, byte-exact ----------------
    if overlap_x == 0 and overlap_y == 0:
        out = torch.zeros(
            (source_batch_size, orig_height, orig_width, c),
            dtype=tiles.dtype,
            device=tiles.device,
        )
        idx = 0
        for bi in range(source_batch_size):
            for r in range(rows):
                y0 = r * stride_h
                for col in range(columns):
                    x0 = col * stride_w
                    out[bi, y0 : y0 + extent_h, x0 : x0 + extent_w, :] = tiles[idx]
                    idx += 1
        return out

    # ---- blended path: separable per-axis edge weights ---------------------
    out = torch.zeros(
        (source_batch_size, orig_height, orig_width, c),
        dtype=torch.float32,
        device=tiles.device,
    )
    wacc = torch.zeros(
        (source_batch_size, orig_height, orig_width, 1),
        dtype=torch.float32,
        device=tiles.device,
    )

    wy_by_row = [
        _axis_weights(extent_h, overlap_y, rows, r, blend).to(tiles.device)
        for r in range(rows)
    ]
    wx_by_col = [
        _axis_weights(extent_w, overlap_x, columns, col, blend).to(tiles.device)
        for col in range(columns)
    ]

    idx = 0
    for bi in range(source_batch_size):
        for r in range(rows):
            y0 = r * stride_h
            for col in range(columns):
                x0 = col * stride_w
                w2d = wy_by_row[r].view(extent_h, 1, 1) * wx_by_col[col].view(
                    1, extent_w, 1
                )
                tile = tiles[idx].to(torch.float32)
                out[bi, y0 : y0 + extent_h, x0 : x0 + extent_w, :] += tile * w2d
                wacc[bi, y0 : y0 + extent_h, x0 : x0 + extent_w, :] += w2d
                idx += 1

    if bool((wacc <= 0).any()):
        raise ZeroWeightError(
            "reconstruction produced at least one zero-weight pixel; "
            "no tile covers it"
        )

    out = out / wacc
    return out.to(dtype=tiles.dtype, device=tiles.device)
