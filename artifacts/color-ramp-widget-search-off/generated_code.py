"""Color Ramp Remap — a V3 ComfyUI custom node.

Remaps an input IMAGE through an adjustable multi-stop color ramp. The ramp
(a list of {pos, color} stops) is carried as a JSON string widget so it
survives save/reload; an interactive frontend widget (web/) edits it.

No external dependencies: torch ships with ComfyUI.
"""

import json
import math

import torch

from comfy_api.latest import ComfyExtension, io

# Revision marker. The math/schema below is unchanged in rev 2 (it already
# matches spec); rev 2's work is the frontend reversible stop-order-edit log.
__revision__ = 2

# Stable, collision-free class id. NEVER rename this (it is the identity in
# saved workflows); change display_name instead.
NODE_ID = "ColorRampRemap_ApplyRamp"

INTERPOLATIONS = ["linear", "smooth", "constant"]

# Default ramp: 0.0 black, 0.4 green, 1.0 magenta.
DEFAULT_RAMP = json.dumps(
    {
        "stops": [
            {"pos": 0.0, "color": "#000000"},
            {"pos": 0.4, "color": "#00ff00"},
            {"pos": 1.0, "color": "#ff00ff"},
        ]
    }
)

_MAX_DIAG = 160  # bounded diagnostic length


class ColorRampError(ValueError):
    """Raised for malformed ramp state; message names the failed condition."""


def _diag(msg: str) -> "ColorRampError":
    if len(msg) > _MAX_DIAG:
        msg = msg[: _MAX_DIAG - 1] + "…"
    return ColorRampError(msg)


def _parse_hex(color):
    """Return (r, g, b) floats in [0, 1] for a '#RRGGBB' string, else raise."""
    if not isinstance(color, str):
        raise _diag("invalid_hex_color: color is not a string")
    c = color.strip()
    if len(c) != 7 or c[0] != "#":
        raise _diag(f"invalid_hex_color: {c!r} is not '#RRGGBB'")
    hexdigits = c[1:]
    try:
        r = int(hexdigits[0:2], 16)
        g = int(hexdigits[2:4], 16)
        b = int(hexdigits[4:6], 16)
    except ValueError:
        raise _diag(f"invalid_hex_color: {c!r} is not '#RRGGBB'")
    return (r / 255.0, g / 255.0, b / 255.0)


def parse_ramp(value):
    """Parse + validate a ramp JSON string.

    Returns a list of (pos: float, (r, g, b)) sorted ascending by pos.
    Raises ColorRampError naming the first failed condition.
    """
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, str):
        raise _diag("malformed_state: ramp is not a JSON string")
    try:
        data = json.loads(value)
    except (ValueError, TypeError):
        raise _diag("malformed_json: ramp is not valid JSON")

    if isinstance(data, list):
        raw_stops = data
    elif isinstance(data, dict) and isinstance(data.get("stops"), list):
        raw_stops = data["stops"]
    else:
        raise _diag("malformed_state: expected object with 'stops' list")

    if len(raw_stops) < 2:
        raise _diag("too_few_stops: need at least two stops")

    positions = []
    stops = []
    for i, s in enumerate(raw_stops):
        if not isinstance(s, dict) or "pos" not in s or "color" not in s:
            raise _diag(f"malformed_state: stop {i} missing 'pos'/'color'")
        pos = s["pos"]
        if isinstance(pos, bool) or not isinstance(pos, (int, float)):
            raise _diag(f"invalid_position: stop {i} position is not a number")
        pos = float(pos)
        if not math.isfinite(pos):
            raise _diag(f"nonfinite_position: stop {i} position is not finite")
        if pos < 0.0 or pos > 1.0:
            raise _diag(f"position_out_of_range: stop {i} pos={pos} not in [0,1]")
        rgb = _parse_hex(s["color"])
        positions.append(pos)
        stops.append((pos, rgb))

    # duplicate positions
    if len(set(positions)) != len(positions):
        raise _diag("duplicate_positions: two or more stops share a position")

    # unsorted positions (must already be ascending)
    if any(positions[i] > positions[i + 1] for i in range(len(positions) - 1)):
        raise _diag("unsorted_positions: positions must be ascending")

    return stops


def apply_color_ramp(image, stops, interpolation):
    """Map ``image`` through the ramp.

    image: IMAGE tensor [B, H, W, C] (C >= 3), floating dtype, any device.
    Preserves B, H, W, C, device and dtype; only channels 0..2 are rewritten.
    """
    if interpolation not in INTERPOLATIONS:
        raise _diag(f"invalid_interpolation: {interpolation!r}")
    if not torch.is_floating_point(image):
        raise _diag("invalid_input: IMAGE tensor must be a floating dtype")
    if image.ndim < 1 or image.shape[-1] < 3:
        raise _diag("invalid_input: IMAGE needs at least 3 channels (RGB)")

    orig_dtype = image.dtype
    device = image.device
    n = len(stops)

    pos = torch.tensor([s[0] for s in stops], dtype=torch.float32, device=device)
    cols = torch.tensor([s[1] for s in stops], dtype=torch.float32, device=device)

    # intensity = arithmetic mean of RGB channels, clamped to [0, 1]
    rgb = image[..., :3].to(torch.float32)
    intensity = rgb.mean(dim=-1).clamp(0.0, 1.0)          # [...]
    flat = intensity.reshape(-1)                           # [M]

    if interpolation == "constant":
        # Hold the left stop of the half-open bracket [pos[i], pos[i+1]);
        # intensity 1.0 (and anything >= last stop) returns the final stop.
        idx = torch.searchsorted(pos, flat, right=True) - 1
        idx = idx.clamp(0, n - 1)
        out = cols[idx]                                    # [M, 3]
    else:
        idx = torch.searchsorted(pos, flat, right=True) - 1
        idx = idx.clamp(0, n - 2)
        left_p = pos[idx]
        right_p = pos[idx + 1]
        left_c = cols[idx]
        right_c = cols[idx + 1]
        seg = right_p - left_p
        u = torch.where(seg > 0, (flat - left_p) / seg, torch.zeros_like(flat))
        u = u.clamp(0.0, 1.0)                              # segment-local fraction
        if interpolation == "smooth":
            u = u * u * (3.0 - 2.0 * u)                    # u*u*(3-2u)
        out = left_c + u.unsqueeze(-1) * (right_c - left_c)

    out = out.reshape(*intensity.shape, 3)
    result = image.clone()                                 # preserve dtype/device/C/alpha
    result[..., :3] = out.to(orig_dtype)
    return result


class ColorRampRemapApplyRamp(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=NODE_ID,
            display_name="Color Ramp Remap (Image)",
            category="image/color",
            description=(
                "Remap an image through an adjustable multi-stop color ramp. "
                "Intensity = mean(RGB), clamped to [0,1], mapped through the "
                "ramp with linear / smooth / constant interpolation."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Image to remap through the ramp"),
                io.String.Input(
                    "ramp",
                    default=DEFAULT_RAMP,
                    multiline=True,
                    tooltip=(
                        "Ramp state as JSON: "
                        '{"stops":[{"pos":float,"color":"#RRGGBB"},...]}. '
                        "Edited by the interactive ramp widget."
                    ),
                ),
                io.Combo.Input(
                    "interpolation",
                    options=INTERPOLATIONS,
                    default="linear",
                    tooltip="Interpolation between stops (ordered after the ramp).",
                ),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def validate_inputs(cls, ramp=None, interpolation=None, **kwargs):
        try:
            parse_ramp(ramp)
        except ColorRampError as e:
            return str(e)
        if interpolation is not None and interpolation not in INTERPOLATIONS:
            return f"invalid_interpolation: {interpolation!r}"
        return True

    @classmethod
    def execute(cls, image, ramp, interpolation) -> io.NodeOutput:
        stops = parse_ramp(ramp)
        result = apply_color_ramp(image, stops, interpolation)
        return io.NodeOutput(result)


class ColorRampRemapExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ColorRampRemapApplyRamp]


async def comfy_entrypoint() -> ColorRampRemapExtension:
    return ColorRampRemapExtension()
