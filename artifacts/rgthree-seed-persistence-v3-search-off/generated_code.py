"""Persistent Seed Sequence — CPU-only backend node.

The Python node is a pure, CPU-only integer passthrough: it returns exactly the
integer it is handed on the ``seed`` input. All of the sequencing behaviour
(fixed / increment / decrement / randomize-after-queue), the step amount that
increment/decrement advance by, the "last used seed" display, and persistence
live in the browser extension at ``web/persistent_seed_sequence.js``. The
extension advances the ``seed`` widget once per queue, *after* the current value
has already been sent to this node, so the node emits the current seed exactly
once per queue.
"""

from comfy_api.latest import ComfyExtension, io


class PersistentSeedSequence(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PersistentSeedSequence",
            display_name="Persistent Seed Sequence",
            category="utils/seed",
            description=(
                "Outputs the current seed as an integer. A browser extension "
                "advances the seed after each queue according to the selected "
                "mode (fixed / increment / decrement / randomize-after-queue) "
                "and shows the value that was just used as 'last_seed'."
            ),
            inputs=[
                io.Int.Input(
                    "seed",
                    default=123,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    tooltip=(
                        "Current seed. The browser extension updates this value "
                        "after each queue according to the selected mode."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(display_name="seed"),
            ],
        )

    @classmethod
    def execute(cls, seed) -> io.NodeOutput:
        # Pure CPU passthrough — emit the current seed exactly once.
        return io.NodeOutput(int(seed))


class PersistentSeedSequenceExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PersistentSeedSequence]


async def comfy_entrypoint() -> PersistentSeedSequenceExtension:
    return PersistentSeedSequenceExtension()
