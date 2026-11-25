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

_CHANNEL_NAMES = {1: ["gray"], 3: ["r", "g", "b"], 4: ["r", "g", "b", "a"]}


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
                    tooltip="Return the stats as a plain sentence or as a JSON object.",
                ),
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, image, output_format="sentence") -> io.NodeOutput:
        batch, height, width, channels = (int(v) for v in image.shape)

        # Mean of each channel over batch, height and width, rounded to 4 decimals.
        means = image.float().mean(dim=(0, 1, 2)).tolist()
        names = list(_CHANNEL_NAMES.get(channels, [f"c{i}" for i in range(channels)]))
        channel_means = {name: round(float(m), 4) for name, m in zip(names, means)}

        # Min / max of each channel over batch, height and width, rounded to 4 decimals.
        flat = image.float().reshape(-1, channels)
        mins = flat.min(dim=0).values.tolist()
        maxs = flat.max(dim=0).values.tolist()
        channel_min = {name: round(float(v), 4) for name, v in zip(names, mins)}
        channel_max = {name: round(float(v), 4) for name, v in zip(names, maxs)}

        if output_format == "json":
            # Existing keys keep their names, types and order; new keys are only appended.
            return io.NodeOutput(
                json.dumps(
                    {
                        "batch": batch,
                        "height": height,
                        "width": width,
                        "channels": channels,
                        "channel_means": channel_means,
                        "channel_min": channel_min,
                        "channel_max": channel_max,
                    }
                )
            )

        means_text = ", ".join(f"{k}={v:.4f}" for k, v in channel_means.items())
        min_text = ", ".join(f"{k}={v:.4f}" for k, v in channel_min.items())
        max_text = ", ".join(f"{k}={v:.4f}" for k, v in channel_max.items())
        return io.NodeOutput(
            f"Image has batch={batch}, height={height}, width={width}, "
            f"channels={channels}; channel means: {means_text}; "
            f"channel min: {min_text}; channel max: {max_text}."
        )


class DemoFarm27SeedExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ImageStatsText]


async def comfy_entrypoint() -> DemoFarm27SeedExtension:
    return DemoFarm27SeedExtension()
