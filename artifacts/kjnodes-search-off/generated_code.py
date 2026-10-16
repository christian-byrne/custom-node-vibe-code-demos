import torch
import torch.nn.functional as F

from comfy_api.latest import ComfyExtension, io, ui


def _parse_amplitudes(text, batch):
    """Parse comma/newline separated normalized amplitudes into a list of floats,
    one per image. Fewer values than frames -> repeat the last value to fill."""
    vals = []
    for chunk in text.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            vals.append(float(chunk))
        except ValueError:
            continue
    if not vals:
        vals = [0.0]
    if len(vals) < batch:
        vals = vals + [vals[-1]] * (batch - len(vals))
    return vals[:batch]


class AmplitudeTransform(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AmplitudeTransform_Transform",
            display_name="Amplitude Transform (Batch)",
            category="amplitude_transform",
            description=(
                "Transforms each image in a batch by its corresponding normalized "
                "amplitude value: zoom, x/y displacement, and optional cumulative "
                "accumulation. Output preserves the input dimensions."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Batch of images [B,H,W,C]; one amplitude is applied per image.",
                ),
                io.String.Input(
                    "amplitudes",
                    multiline=True,
                    default="0.0, 0.5, 1.0",
                    tooltip=(
                        "Normalized amplitude (typically 0..1) per frame, comma or "
                        "newline separated. The last value repeats to fill the batch."
                    ),
                ),
                io.Float.Input(
                    "zoom",
                    default=0.5,
                    min=-1.0,
                    max=4.0,
                    step=0.01,
                    tooltip="Zoom gain. Scale = 1 + zoom * amplitude. Positive zooms in.",
                ),
                io.Float.Input(
                    "x_displacement",
                    default=0.0,
                    min=-4096.0,
                    max=4096.0,
                    step=1.0,
                    tooltip="Horizontal shift (pixels) at amplitude 1.0; positive = right.",
                ),
                io.Float.Input(
                    "y_displacement",
                    default=0.0,
                    min=-4096.0,
                    max=4096.0,
                    step=1.0,
                    tooltip="Vertical shift (pixels) at amplitude 1.0; positive = down.",
                ),
                io.Boolean.Input(
                    "cumulative",
                    default=False,
                    tooltip=(
                        "When on, amplitudes accumulate across the batch (running sum) "
                        "so the transform builds up progressively, frame by frame."
                    ),
                ),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, image, amplitudes, zoom, x_displacement, y_displacement, cumulative) -> io.NodeOutput:
        b, h, w, c = image.shape
        amps = _parse_amplitudes(amplitudes, b)

        if cumulative:
            running = 0.0
            acc = []
            for a in amps:
                running += a
                acc.append(running)
            amps = acc

        device = image.device
        dtype = image.dtype
        # [B,H,W,C] -> [B,C,H,W] for affine sampling
        x = image.permute(0, 3, 1, 2)
        out = torch.empty_like(x)

        for i in range(b):
            a = float(amps[i])
            s = 1.0 + zoom * a
            if s <= 1e-3:
                s = 1e-3
            inv = 1.0 / s
            # translation in normalized grid coords (grid spans [-1, 1])
            tx = -2.0 * (x_displacement * a) / w
            ty = -2.0 * (y_displacement * a) / h
            theta = torch.tensor(
                [[inv, 0.0, tx], [0.0, inv, ty]],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
            sampled = F.grid_sample(
                x[i : i + 1].float(),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            out[i : i + 1] = sampled.to(dtype)

        out = out.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)
        return io.NodeOutput(out, ui=ui.PreviewImage(out, cls=cls))


class AmplitudeTransformExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AmplitudeTransform]


async def comfy_entrypoint() -> AmplitudeTransformExtension:
    return AmplitudeTransformExtension()
