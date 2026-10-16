"""Universal N-to-1 switch node (V3 / nodes 2.0). Grounding: see REFERENCES.md.

Local-only grounding (this ComfyUI install; no external pack copied/installed):
  autogrow slots  -> comfy_extras/nodes_logic.py AndNode/OrNode (io.Autogrow.TemplatePrefix)
  any type        -> comfy_extras/nodes_logic.py NotNode (io.AnyType)
  lazy skip       -> comfy_extras/nodes_logic.py SwitchNode (connected-unevaluated == None)
  bounded error   -> io.NodeOutput(block_execution=...) -> ExecutionBlocker

No external reference pack (WAS Node Suite, rgthree, etc.) was copied, installed, or read.
The requested WAS `wildcard switch input` search returned no results (per-turn search limit),
so NO behavioral detail is claimed to be adopted from it. See REFERENCES.md.
"""

import json
import os
import re
import time

from comfy_api.latest import io

MISSING = object()  # a NOT-connected input (mirrors nodes_logic.py SwitchNode)
MAX_SLOTS = 64

_PACK_ROOT = os.path.dirname(os.path.abspath(__file__))
_EVIDENCE_DIR = os.path.join(_PACK_ROOT, "_evidence")
EXEC_LOG = os.path.join(_EVIDENCE_DIR, "exec_log.jsonl")


def _log(path, obj):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, default=repr) + "\n")
    except Exception:
        pass


_NUM_RE = re.compile(r"(\d+)")


def _suffix_num(name):
    nums = _NUM_RE.findall(str(name))
    return int(nums[-1]) if nums else 0


def _group_items(group):
    """Autogrow group -> ordered [(slot_name, value)], sorted by numeric suffix.

    Connected-but-unevaluated lazy inputs arrive as value None, but their KEY is still
    present, so the connected count is correct regardless of evaluation.
    """
    items = []
    if isinstance(group, dict):
        for key, val in group.items():
            items.append((str(key), val))
    elif isinstance(group, (list, tuple)):
        for entry in group:
            if isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[0], str):
                items.append((entry[0], entry[1]))
            else:
                items.append((str(len(items)), entry))
    items.sort(key=lambda kv: _suffix_num(kv[0]))
    return items


def _decide(names, select, fallback):
    """Return (target_slot_or_None, mode, message). select is 1-based over connected inputs."""
    n = len(names)
    if n == 0:
        return None, "error", "no inputs are connected"
    if 1 <= select <= n:
        return names[select - 1], "selected", f"selected connected slot {select} of {n}"
    if fallback:
        return names[0], "fallback", (
            f"selector {select} outside [1,{n}]; fallback_to_first_connected=True "
            f"-> lowest-numbered connected input (slot 1)"
        )
    return None, "error", (
        f"selector {select} outside valid range [1,{n}] and fallback_to_first_connected=False"
    )


class UniversalSwitch(io.ComfyNode):
    """Universal N-to-1 switch: any type, autogrow inputs, integer selector, explicit
    bounded diagnostic on out-of-range/disconnected, optional fallback to the lowest
    connected input, and lazy skipping of every unselected branch.
    """

    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(
            input=io.AnyType.Input("value", lazy=True),
            prefix="value",
            min=1,
            max=MAX_SLOTS,
        )
        return io.Schema(
            node_id="UniversalNto1Switch",
            display_name="Universal N-to-1 Switch",
            category="utils/switch",
            description=(
                "Route one of N autogrowing any-type inputs to a single output. `select` is "
                "1-based over the CONNECTED inputs. Unselected branches are never evaluated. "
                "Out-of-range/disconnected selection yields an explicit bounded diagnostic "
                "unless fallback_to_first_connected is set (then the lowest connected input)."
            ),
            is_experimental=True,
            # Stable widget order: select FIRST, then the fallback flag, then autogrow inputs.
            inputs=[
                io.Int.Input(
                    "select", default=1, min=1, max=MAX_SLOTS,
                    tooltip="1-based index over the CONNECTED inputs.",
                ),
                io.Boolean.Input(
                    "fallback_to_first_connected", default=False,
                    tooltip=(
                        "If the selected slot is out of range/disconnected: True uses the "
                        "lowest connected input (no other branch evaluated); False emits the "
                        "explicit bounded diagnostic."
                    ),
                ),
                io.Autogrow.Input("inputs", template=template, optional=True),
            ],
            outputs=[io.AnyType.Output(display_name="selected")],
        )

    @classmethod
    def check_lazy_status(cls, select, fallback_to_first_connected, inputs=None, **kwargs):
        """Request evaluation of ONLY the chosen slot, so unselected branches never run."""
        items = _group_items(inputs)
        names = [n for n, _ in items]
        if not names:
            return []
        target, _mode, _msg = _decide(names, int(select), bool(fallback_to_first_connected))
        if target is None:
            return []  # error mode: evaluate nothing; bounded diagnostic emitted in execute
        valmap = dict(items)
        if valmap.get(target, MISSING) is None:  # None == connected-but-unevaluated
            return [target]
        return []

    @classmethod
    def execute(cls, select, fallback_to_first_connected, inputs=None, **kwargs) -> io.NodeOutput:
        items = _group_items(inputs)
        names = [n for n, _ in items]
        valmap = dict(items)
        sel = int(select)
        fb = bool(fallback_to_first_connected)
        target, mode, msg = _decide(names, sel, fb)

        evaluated = [n for n, v in items if v is not None]
        _log(EXEC_LOG, {
            "ts": time.time(),
            "node": "UniversalNto1Switch",
            "select": sel,
            "fallback_to_first_connected": fb,
            "connected_slots": names,
            "connected_count": len(names),
            "evaluated_slots": evaluated,
            "target_slot": target,
            "mode": mode,
            "message": msg,
        })

        if target is None:
            # Explicit bounded diagnostic -> ExecutionBlocker (downstream skipped cleanly).
            return io.NodeOutput(block_execution=f"UniversalNto1Switch: {msg}")
        return io.NodeOutput(valmap.get(target))

