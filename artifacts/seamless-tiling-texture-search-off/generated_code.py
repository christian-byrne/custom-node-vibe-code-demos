import torch

from comfy_api.latest import io


def _luminance(image):
    # image: [B, H, W, C] float 0..1  ->  [B, H, W]
    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _to_rgb(gray):
    # gray: [B, H, W]  ->  [B, H, W, 3]
    return gray.unsqueeze(-1).repeat(1, 1, 1, 3)


class TextureToPBRMaps(io.ComfyNode):
    """Derive seamless height, normal and ambient-occlusion maps from a diffuse
    texture. All neighbourhood sampling uses torch.roll (wrap-around), so if the
    input tiles seamlessly the output maps tile seamlessly too."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TextureMaps_TextureToPBRMaps",
            display_name="Texture to PBR Maps (Height/Normal/AO)",
            category="texture/pbr",
            description="Seamless height, normal and AO maps from a diffuse/albedo texture.",
            inputs=[
                io.Image.Input("image", tooltip="Diffuse / albedo texture (ideally already tileable)"),
                io.Float.Input("normal_strength", default=2.0, min=0.0, max=20.0, step=0.1,
                               tooltip="Bump intensity for the normal map"),
                io.Boolean.Input("normal_flip_y", default=False,
                                 tooltip="Flip green channel: off = OpenGL, on = DirectX"),
                io.Int.Input("height_blur", default=0, min=0, max=20, step=1,
                             tooltip="Smoothing passes applied to the height map"),
                io.Float.Input("ao_strength", default=1.5, min=0.0, max=10.0, step=0.1,
                               tooltip="Ambient-occlusion contrast"),
                io.Int.Input("ao_radius", default=4, min=1, max=32, step=1,
                             tooltip="AO sampling radius in pixels"),
            ],
            outputs=[
                io.Image.Output(display_name="height"),
                io.Image.Output(display_name="normal"),
                io.Image.Output(display_name="ao"),
            ],
        )

    @classmethod
    def execute(cls, image, normal_strength, normal_flip_y, height_blur, ao_strength, ao_radius) -> io.NodeOutput:
        img = image.float()
        height = _luminance(img)  # [B, H, W]

        # Optional smoothing via repeated wrap-around 5-tap average.
        for _ in range(int(height_blur)):
            height = (
                height
                + torch.roll(height, 1, dims=2) + torch.roll(height, -1, dims=2)
                + torch.roll(height, 1, dims=1) + torch.roll(height, -1, dims=1)
            ) / 5.0

        # Height map, normalized per image to use the full 0..1 range.
        hmin = height.amin(dim=(1, 2), keepdim=True)
        hmax = height.amax(dim=(1, 2), keepdim=True)
        height_norm = (height - hmin) / (hmax - hmin + 1e-6)

        # Normal map from wrap-around height gradients (seamless).
        dx = (torch.roll(height, -1, dims=2) - torch.roll(height, 1, dims=2)) * 0.5
        dy = (torch.roll(height, -1, dims=1) - torch.roll(height, 1, dims=1)) * 0.5
        nx = -dx * normal_strength
        ny = -dy * normal_strength
        nz = torch.ones_like(height)
        length = torch.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
        nx, ny, nz = nx / length, ny / length, nz / length
        if normal_flip_y:
            ny = -ny
        normal = torch.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], dim=-1)

        # Ambient occlusion: how much higher the neighbourhood is than each pixel.
        occ = torch.zeros_like(height)
        r = int(ao_radius)
        offsets = [(r, 0), (-r, 0), (0, r), (0, -r), (r, r), (r, -r), (-r, r), (-r, -r)]
        for dxs, dys in offsets:
            neigh = torch.roll(torch.roll(height, dys, dims=1), dxs, dims=2)
            occ = occ + torch.clamp((neigh - height) * ao_strength, min=0.0)
        occ = occ / len(offsets)
        ao = torch.clamp(1.0 - occ, 0.0, 1.0)

        return io.NodeOutput(_to_rgb(height_norm), normal, _to_rgb(ao))
