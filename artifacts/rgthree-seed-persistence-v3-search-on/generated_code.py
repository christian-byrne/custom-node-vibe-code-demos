"""Persistent Seed Sequence — a narrow, CPU-only seed source node (V3 API).

The Python side is deliberately minimal: one integer seed input, one integer
output, pure-Python pass-through (no torch, CPU only). All of the sequencing
behaviour (fixed / increment / decrement / randomize-after-queue) and the
"last-used seed" display live in the browser extension in web/, which advances
the seed once per queue using the widget-value serialization convention.
"""

from comfy_api.latest import io


class PersistentSeedSequence(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            # Stable, collision-free identity in saved workflows — never rename.
            node_id="PersistentSeedSequence_Seed",
            display_name="Persistent Seed Sequence",
            category="persistent_seed_sequence",
            description=(
                "CPU-only seed source. Outputs the current seed as an integer, "
                "exactly once per queue. A browser extension then advances the "
                "seed according to the selected mode (fixed / increment / "
                "decrement / randomize-after-queue) and shows the value that was "
                "just used as a separate, read-only last-used seed. Selected "
                "mode, current seed, widget order and the last-used seed all "
                "persist across workflow save and reload."
            ),
            inputs=[
                io.Int.Input(
                    "seed",
                    default=123,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    # We supply our own mode control widget in the frontend, so
                    # the built-in control_after_generate widget must stay off.
                    control_after_generate=False,
                    tooltip=(
                        "Current seed. Output once per queue, then advanced by "
                        "the mode widget after the prompt is sent."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(display_name="seed"),
            ],
        )

    @classmethod
    def execute(cls, seed) -> io.NodeOutput:
        # Pure, deterministic, CPU-only: emit the current seed unchanged.
        return io.NodeOutput(int(seed))
