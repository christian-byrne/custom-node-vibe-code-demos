import re

import torch
import torch.nn.functional as F

from comfy_api.latest import ComfyExtension, io, ui


def _parse_amplitudes(text: str) -> list[float]:
    """Parse a free-form list of normalized amplitude values.

    Accepts commas, whitespace and newlines as separators. Values are clamped
    to the normalized range [0, 1]. Returns [0.0] when nothing parses.
    """
    out: list[float] = []
    for tok in re.split(r"[,\s]+", (text or "").strip()):
        if tok == "":
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    out = [min(1.0, max(0.0, v)) for v in out]
    return out or [0.0]


class AmplitudeImageTransform(io.ComfyNode):
    """Transform each image in a batch by its corresponding normalized amplitude.

    Each amplitude drives a zoom (scale about center) plus an x/y displacement.
    Output dimensions always match the input. In cumulative mode, each frame's
    transform is COMPOSED onto the previous frame's accumulated transform
    (matrix product), so zoom multiplies and displacement compounds across the
    batch like a real dolly/pan -- a visibly stronger trajectory than the
    per-frame mode, not just a running sum of amplitudes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AmplitudeTransform_ImageFromAmplitude",
            display_name="Amplitude Image Transform",
            category="amplitude/transform",
            description=(
                "Zoom and x/y displace each image in a batch by a corresponding "
                "normalized amplitude. Preserves image dimensions. Cumulative "
                "mode accumulates the amplitudes across the batch."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Image batch. A single image is broadcast to match "
                    "the number of amplitudes.",
                ),
                io.String.Input(
                    "amplitudes",
                    multiline=True,
                    default="0.0, 0.5, 1.0",
                    tooltip="Normalized amplitude per frame (0..1), separated by "
                    "commas/spaces/newlines. One value per output image.",
                ),
                io.Float.Input(
                    "zoom",
                    default=0.5,
                    min=0.0,
                    max=4.0,
                    step=0.01,
                    tooltip="Extra zoom at amplitude 1.0 (scale = 1 + zoom*amp).",
                ),
                io.Float.Input(
                    "x_shift",
                    default=0.0,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Horizontal displacement (fraction of width) at "
                    "amplitude 1.0. Positive moves content right.",
                ),
                io.Float.Input(
                    "y_shift",
                    default=0.0,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Vertical displacement (fraction of height) at "
                    "amplitude 1.0. Positive moves content down.",
                ),
                io.Boolean.Input(
                    "cumulative",
                    default=False,
                    tooltip="Compose each frame's transform onto the previous "
                    "frame's accumulated transform, so zoom/displacement "
                    "compound across the batch (a progressive dolly/pan).",
                ),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, image, amplitudes, zoom, x_shift, y_shift, cumulative) -> io.NodeOutput:
        amps = _parse_amplitudes(amplitudes)

        b, h, w, c = image.shape

        # Align the frame count to the amplitudes.
        if b == 1:
            n = len(amps)
            frames = image if n == 1 else image.repeat(n, 1, 1, 1)
        else:
            n = b
            if len(amps) < n:
                amps = amps + [amps[-1]] * (n - len(amps))
            amps = amps[:n]
            frames = image

        device = frames.device
        dtype = frames.dtype

        def _base_mat(a: float) -> torch.Tensor:
            """3x3 output-grid -> input-grid transform for one amplitude.

            grid_sample normalized coords span [-1, 1] across the full image.
            """
            scale = 1.0 + float(zoom) * a
            inv = 1.0 / scale if scale != 0.0 else 1.0
            tx = -2.0 * float(x_shift) * a
            ty = -2.0 * float(y_shift) * a
            return torch.tensor(
                [[inv, 0.0, tx], [0.0, inv, ty], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )

        # Per-frame affine matrices mapping output grid -> input sampling grid.
        # Per-frame mode: each frame uses its own amplitude's transform.
        # Cumulative mode: compose each frame's transform onto the running
        # accumulated transform (matrix product), so zoom multiplies and
        # displacement compounds across the batch -- a progressively stronger
        # dolly/pan, clearly distinct from the per-frame result.
        mats: list[torch.Tensor] = []
        if cumulative:
            acc = torch.eye(3, dtype=torch.float32)
            for a in amps:
                acc = acc @ _base_mat(a)
                mats.append(acc.clone())
        else:
            for a in amps:
                mats.append(_base_mat(a))

        theta = torch.stack([m[:2, :] for m in mats]).to(device=device, dtype=torch.float32)

        nchw = frames.permute(0, 3, 1, 2).to(torch.float32)
        grid = F.affine_grid(theta, size=(n, c, h, w), align_corners=False)
        sampled = F.grid_sample(
            nchw, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        out = sampled.permute(0, 2, 3, 1).clamp(0.0, 1.0).to(dtype).contiguous()

        return io.NodeOutput(out, ui=ui.PreviewImage(out, cls=cls))


class AmplitudeTransformExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AmplitudeImageTransform]


async def comfy_entrypoint() -> AmplitudeTransformExtension:
    return AmplitudeTransformExtension()
