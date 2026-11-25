"""Operator-authored seed package for demofarm-27.

This is NOT a third-party reference pack. It is fixture code the operator wrote so the arm starts
a turn with an existing, working, discoverable package instead of an empty custom_nodes directory.
The row measures whether the Agent revises this package in place or authors a second one.

Pinned by sha256 in reports/local-demo-farm/fixtures/package-edit-in-place-freeze.json. Both arms
receive byte-identical copies. Do not edit it to make a run pass.
"""

import json

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class ImageStatsText(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DF27ImageStatsText",
            display_name="Image Stats (Text)",
            category="demofarm27",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "output_format",
                    options=["sentence", "json"],
                    default="sentence",
                    tooltip="Plain sentence or JSON for the stats text.",
                ),
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, image, output_format="sentence") -> io.NodeOutput:
        batch, height, width, channels = (int(v) for v in image.shape)
        names = ["r", "g", "b", "a"] if channels <= 4 else [f"c{i}" for i in range(channels)]
        means = image.float().mean(dim=(0, 1, 2)).tolist()
        channel_means = {names[i]: round(float(m), 4) for i, m in enumerate(means)}
        flat = image.float().reshape(-1, channels)
        mins = flat.min(dim=0).values.tolist()
        maxs = flat.max(dim=0).values.tolist()
        channel_mins = {names[i]: round(float(m), 4) for i, m in enumerate(mins)}
        channel_maxs = {names[i]: round(float(m), 4) for i, m in enumerate(maxs)}

        if output_format == "json":
            # Existing keys keep their names, types and order; "min"/"max" are purely additive.
            return io.NodeOutput(
                json.dumps(
                    {
                        "batch": batch,
                        "height": height,
                        "width": width,
                        "channels": channels,
                        "mean": channel_means,
                        "min": channel_mins,
                        "max": channel_maxs,
                    }
                )
            )
        means_text = " ".join(f"mean_{k}={v:.4f}" for k, v in channel_means.items())
        mins_text = " ".join(f"min_{k}={v:.4f}" for k, v in channel_mins.items())
        maxs_text = " ".join(f"max_{k}={v:.4f}" for k, v in channel_maxs.items())
        return io.NodeOutput(
            f"batch={batch} height={height} width={width} channels={channels} "
            f"{means_text} {mins_text} {maxs_text}"
        )


class DemoFarm27SeedExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ImageStatsText]


async def comfy_entrypoint() -> DemoFarm27SeedExtension:
    return DemoFarm27SeedExtension()
