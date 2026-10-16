"""ComfyUI V3 node: Latent Video Crossfade Join."""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from .crossfade_core import crossfade_latents


class LatentVideoCrossfade(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LatentVideoCrossfadeJoin",
            display_name="Latent Video Crossfade Join",
            category="latent/video",
            description=(
                "Join two 5D video latents [B,C,T,H,W] with an N-frame "
                "crossfade (easing: smoothstep or linear). Output length = "
                "T_a + T_b - N. A's top-level metadata is preserved; mismatched "
                "rank or B/C/H/W, or N outside 1..min(T_a,T_b), fail with a "
                "clear error."
            ),
            inputs=[
                io.Latent.Input(
                    "latent_a",
                    tooltip="Clip A. Its top-level metadata is kept; only "
                    "'samples' is replaced.",
                ),
                io.Latent.Input(
                    "latent_b",
                    tooltip="Clip B. Appended after the blended seam.",
                ),
                io.Int.Input(
                    "crossfade_frames",
                    default=1,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="Overlap length N. Must be 1..min(T_a, T_b).",
                ),
                io.Combo.Input(
                    "easing",
                    options=["smoothstep", "linear"],
                    default="smoothstep",
                    tooltip="Interior seam weighting. 'smoothstep' (default) "
                    "uses B weight t*t*(3-2t); 'linear' uses B weight t, where "
                    "t=(i+1)/(N+1).",
                ),
            ],
            outputs=[io.Latent.Output(display_name="LATENT")],
        )

    @classmethod
    def execute(
        cls, latent_a, latent_b, crossfade_frames, easing
    ) -> io.NodeOutput:
        out = crossfade_latents(latent_a, latent_b, crossfade_frames, easing)
        return io.NodeOutput(out)


class LatentVideoCrossfadeExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LatentVideoCrossfade]


async def comfy_entrypoint() -> LatentVideoCrossfadeExtension:
    return LatentVideoCrossfadeExtension()
