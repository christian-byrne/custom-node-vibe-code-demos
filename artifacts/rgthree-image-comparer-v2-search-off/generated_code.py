"""Angular Image Comparer - registered ComfyUI V3 node.

Keeps DOM / framework imports out of this Python module. The interactive
canvas (playback clock, draggable angular split, controls) lives entirely
in the served frontend extension under ``web/comparer.js``.

The node:
  * accepts two IMAGE batches (``image_a``, ``image_b``) and two
    independent positive playback rates (``fps_a``, ``fps_b``);
  * validates them through the runtime-free ``contract`` core;
  * returns a deterministic image passthrough (the inputs unchanged),
    suitable for ordinary graph execution;
  * is an output node that also emits every frame of each side as a UI
    preview plus the per-side counts and rates, so the frontend can drive
    both sides from one shared clock.
"""

from __future__ import annotations

import nodes  # core ComfyUI nodes module (provides PreviewImage)
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from . import contract


class AgentAngularImageComparer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AgentAngularImageComparer",
            display_name="Angular Image Comparer (A/B)",
            category="image/compare",
            description=(
                "Compare two IMAGE batches on one shared playback clock with "
                "a draggable angular split (0-360 degrees) and a resolution-"
                "match toggle. Returns both batches unchanged as a "
                "deterministic passthrough."
            ),
            inputs=[
                io.Image.Input(
                    "image_a",
                    tooltip="First IMAGE batch (side A). BHWC, floating, [0,1].",
                ),
                io.Image.Input(
                    "image_b",
                    tooltip="Second IMAGE batch (side B). BHWC, floating, [0,1].",
                ),
                io.Float.Input(
                    "fps_a",
                    default=2.0,
                    min=0.01,
                    max=1000.0,
                    step=0.01,
                    tooltip="Side A playback rate in frames/second (must be > 0).",
                ),
                io.Float.Input(
                    "fps_b",
                    default=3.0,
                    min=0.01,
                    max=1000.0,
                    step=0.01,
                    tooltip="Side B playback rate in frames/second (must be > 0).",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image_a"),
                io.Image.Output(display_name="image_b"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image_a, image_b, fps_a, fps_b) -> io.NodeOutput:
        # Validate everything and obtain the deterministic passthrough.
        out_a, out_b, rate_a, rate_b = contract.passthrough(
            image_a, image_b, fps_a, fps_b
        )

        count_a = int(out_a.shape[0])
        count_b = int(out_b.shape[0])

        # Emit every frame of each side as individual temp previews so the
        # frontend can animate through them on the shared clock.
        saved_a = nodes.PreviewImage().save_images(
            out_a, filename_prefix="agent.comparer.a"
        )
        saved_b = nodes.PreviewImage().save_images(
            out_b, filename_prefix="agent.comparer.b"
        )

        ui = {
            "a_images": saved_a["ui"]["images"],
            "b_images": saved_b["ui"]["images"],
            "a_count": [count_a],
            "b_count": [count_b],
            "fps_a": [rate_a],
            "fps_b": [rate_b],
        }
        return io.NodeOutput(out_a, out_b, ui=ui)


class AgentAngularImageComparerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AgentAngularImageComparer]


async def comfy_entrypoint() -> AgentAngularImageComparerExtension:
    return AgentAngularImageComparerExtension()
