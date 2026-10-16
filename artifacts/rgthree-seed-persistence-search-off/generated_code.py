"""Persistent Seed Sequence — a CPU-only seed node.

Revision 2: adds an integer `step` widget. The browser extension now uses that
step for the increment / decrement modes (randomize-after-queue and fixed are
unaffected).

The Python side is deliberately tiny and pure: it receives the seed chosen for
*this* queue and passes it straight through as an INT. All sequencing (fixed /
increment / decrement / randomize-after-queue, by `step`) and the "last-used
seed" display live in the browser extension (web/persistent_seed_sequence.js),
following ComfyUI's widget `afterQueued()` convention — the same mechanism the
native seed `control_after_generate` control (and rgthree's seed widget) use to
mutate a widget value *after* a prompt has been queued. No torch / CUDA is
touched here, so the node is CPU-only by construction.
"""

from comfy_api.latest import ComfyExtension, io, ui

# Widest range ComfyUI uses for seeds. Declared once so schema + frontend agree.
SEED_MIN = 0
SEED_MAX = 0xFFFFFFFFFFFFFFFF


class PersistentSeedSequenceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        # Widget ORDER here is the on-disk order (widgets_values is positional),
        # so declaring every widget in the schema in a FIXED order is what keeps
        # the saved widget order stable across save/reload. Revision-2 order:
        #   index 0: seed      (current seed)
        #   index 1: mode      (fixed / increment / decrement / randomize-after-queue)
        #   index 2: step      (increment/decrement amount)
        #   index 3: last_seed (separately visible last-used seed, read-only in UI)
        return io.Schema(
            node_id="PersistentSeedSequence_Seed",
            display_name="Persistent Seed Sequence",
            category="persistent_seed_sequence",
            description=(
                "Outputs the current seed once per queue, mirrors it to a "
                "separate last-used display, then advances the current seed by "
                "`step` per the selected mode (handled in the browser)."
            ),
            inputs=[
                io.Int.Input(
                    "seed",
                    default=123,
                    min=SEED_MIN,
                    max=SEED_MAX,
                    step=1,
                    # We advance the seed ourselves in the frontend; do NOT let
                    # ComfyUI add its own control_after_generate widget, which
                    # would both fight our logic and disturb widget order.
                    control_after_generate=False,
                    tooltip="Current seed. Sent to the output for this queue.",
                ),
                io.Combo.Input(
                    "mode",
                    options=["fixed", "increment", "decrement", "randomize-after-queue"],
                    default="increment",
                    tooltip="How the current seed changes AFTER each queue.",
                ),
                io.Int.Input(
                    "step",
                    default=1,
                    min=1,
                    max=SEED_MAX,
                    step=1,
                    tooltip="Amount added (increment) or subtracted (decrement) "
                    "from the current seed after each queue.",
                ),
                io.Int.Input(
                    "last_seed",
                    default=123,
                    min=SEED_MIN,
                    max=SEED_MAX,
                    step=1,
                    optional=True,
                    control_after_generate=False,
                    tooltip="Last-used seed: the value output by the most recent queue.",
                ),
            ],
            outputs=[io.Int.Output(display_name="seed")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, seed, mode=None, step=None, last_seed=None) -> io.NodeOutput:
        # Pure pass-through of the seed chosen for this queue. `step`/`mode` are
        # frontend-only sequencing controls; the backend never mutates the seed.
        # The preview lets us read back exactly which seed the backend used on
        # each invocation.
        value = int(seed)
        return io.NodeOutput(value, ui=ui.PreviewText(str(value)))


class PersistentSeedSequenceExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PersistentSeedSequenceNode]


async def comfy_entrypoint() -> PersistentSeedSequenceExtension:
    return PersistentSeedSequenceExtension()
