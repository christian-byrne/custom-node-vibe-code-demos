"""Persistent Seed Sequence — a CPU-only seed-sequencing node.

The Python node is a pure integer pass-through: it takes the current seed and
emits it unchanged, exactly once per queue. All sequencing behaviour
(fixed / increment / decrement / randomize-after-queue) and the "last-used
seed" display live in the browser extension under ./web, because the step must
happen AFTER a prompt is queued so the value sent for the current queue is
exactly the current seed.

CPU-only: this node does no tensor / GPU work; it is pure Python integer logic.

Persistence convention (verifiable): the state fields are real backend widgets
declared in a fixed order — seed, mode, last_seed, step. ComfyUI serializes
each widget's value positionally into node.widgets_values in that declaration
order and restores them positionally on load, so workflow save/reload
round-trips the current seed, the selected mode, the step, the stable widget
order, and the separately visible last-used seed with no custom serialization.

Revision 2: adds an integer `step` widget (appended last, so the existing
seed / mode / last_seed positions are left exactly where they were and the
widget order stays stable). increment and decrement now advance the current
seed by `step` instead of by 1; fixed and randomize-after-queue are unchanged.
"""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

# 64-bit unsigned range, matching ComfyUI's own seed widgets.
SEED_MIN = 0
SEED_MAX = 0xFFFFFFFFFFFFFFFF

# -1 is the "no queue yet" sentinel for the last-used seed display.
LAST_SEED_NONE = -1

MODES = ["fixed", "increment", "decrement", "randomize-after-queue"]


class PersistentSeedSequenceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PersistentSeedSequence_Seed",
            display_name="Persistent Seed Sequence",
            category="persistent_seed_sequence",
            description=(
                "CPU-only seed sequencer. Emits the current seed once per "
                "queue, records it as the last-used seed, then advances the "
                "current seed by the selected mode (handled in the browser "
                "after each queue). increment/decrement advance by `step`. "
                "Widget order is fixed: seed, mode, last_seed, step."
            ),
            inputs=[
                io.Int.Input(
                    "seed",
                    default=123,
                    min=SEED_MIN,
                    max=SEED_MAX,
                    # Suppress ComfyUI's built-in control-after-generate widget:
                    # sequencing is governed solely by our `mode` widget, so a
                    # second native stepper would be redundant and could
                    # double-step the seed. Keeping it out also keeps widget
                    # order fixed at exactly: seed, mode, last_seed, step.
                    control_after_generate=False,
                    tooltip=(
                        "Current seed. This exact value is emitted for the "
                        "next queue."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=MODES,
                    default="increment",
                    tooltip="How the current seed advances AFTER each queue.",
                ),
                io.Int.Input(
                    "last_seed",
                    default=LAST_SEED_NONE,
                    min=LAST_SEED_NONE,
                    max=SEED_MAX,
                    tooltip=(
                        "Last-used seed: the value emitted by the most recent "
                        "queue (display only; updated after each queue)."
                    ),
                ),
                io.Int.Input(
                    "step",
                    default=1,
                    min=1,
                    max=SEED_MAX,
                    # Appended LAST so seed/mode/last_seed keep their existing
                    # serialization positions (0/1/2) and widget order stays
                    # stable; no built-in stepper for the same reason as seed.
                    control_after_generate=False,
                    tooltip=(
                        "Amount increment/decrement advance the current seed "
                        "by after each queue (ignored by fixed/randomize)."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(display_name="seed"),
            ],
        )

    @classmethod
    def execute(cls, seed: int, mode: str = "increment",
                last_seed: int = LAST_SEED_NONE,
                step: int = 1) -> io.NodeOutput:
        # Pure CPU pass-through: emit the current seed exactly once.
        # `mode`, `last_seed` and `step` are display/behaviour state owned by
        # the browser extension; the backend intentionally ignores them.
        return io.NodeOutput(int(seed))


class PersistentSeedSequenceExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PersistentSeedSequenceNode]


async def comfy_entrypoint() -> PersistentSeedSequenceExtension:
    return PersistentSeedSequenceExtension()
