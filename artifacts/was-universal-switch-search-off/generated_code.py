"""Universal N-to-1 switch for ComfyUI (V3 custom-node API).

Unpublished draft. See REFERENCES.md for provenance.

Behaviour (revision 2):
  * accepts any Comfy type on each input (wildcard "*" socket)
  * input slots grow as connections are made (autogrow)
  * an integer selector (1-based, positional over connected inputs) chooses
    which input is forwarded
  * NEW: a boolean ``fallback_to_first_connected`` flag. When the selected slot
    is missing/disconnected:
        - flag TRUE  -> forward the lowest-numbered connected input instead,
                        still without evaluating any other branch;
        - flag FALSE -> raise the explicit, bounded diagnostic (revision 1
                        behaviour, preserved).
  * widget order is stable: select, then fallback_to_first_connected, then the
    growing value inputs.
  * unselected branches are NOT evaluated (lazy inputs + check_lazy_status):
    check_lazy_status requests exactly ONE slot (or none), so at most the
    chosen/fallback branch is ever executed.
  * connectivity is read authoritatively from the prompt graph (DYNPROMPT +
    UNIQUE_ID hidden inputs), so a connected-but-unevaluated lazy slot is never
    confused with a disconnected one.
"""

import os
import re

from comfy_api.latest import ComfyExtension, io, ui

_RUNTIME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime")
_SLOT_RE = re.compile(r"(\d+)")

# Fixed (non-autogrow) input ids on UniversalSwitch -- never a value slot.
_FIXED_INPUTS = ("select", "fallback_to_first_connected")


def _runtime_dir():
    os.makedirs(_RUNTIME_DIR, exist_ok=True)
    return _RUNTIME_DIR


def _diag(tag, payload):
    """Append one diagnostic line to runtime/diag.log (test instrumentation)."""
    try:
        with open(os.path.join(_runtime_dir(), "diag.log"), "a", encoding="utf-8") as fh:
            fh.write(f"{tag}: {payload}\n")
    except Exception:
        pass


class _AnyStr(str):
    """A type string that compares equal to every other type string.

    This is the long-standing ComfyUI 'wildcard' trick: a socket whose declared
    type never fails a type-equality check, so it accepts (and emits) any type.
    """

    def __eq__(self, other):  # noqa: D401
        return True

    def __ne__(self, other):
        return False

    def __hash__(self):
        return hash("*")


ANY = io.Custom(_AnyStr("*"))


def _slot_index(key):
    m = _SLOT_RE.search(str(key))
    return int(m.group(1)) if m else 0


def _ordered_items(values):
    """Return (key, value) pairs of an autogrow payload in stable slot order.

    Autogrow may hand execute()/check_lazy_status() a dict keyed by per-slot ids
    ("value0", "value1", ...) or a list; normalise both into an ordered list.
    """
    if isinstance(values, dict):
        return [(k, values[k]) for k in sorted(values.keys(), key=lambda k: (_slot_index(k), str(k)))]
    if isinstance(values, (list, tuple)):
        return [(i, v) for i, v in enumerate(values)]
    return [(0, values)]


def _is_link(val):
    """True if a serialized prompt input value is a link [node_id, output_index]."""
    return (
        isinstance(val, (list, tuple))
        and len(val) == 2
        and not isinstance(val[0], bool)
        and isinstance(val[1], int)
        and not isinstance(val[1], bool)
    )


def _connected_from_prompt(hidden):
    """Authoritative connectivity: ordered list of (slot_index, flat_key) for
    autogrow value slots that carry a link, taken from the prompt graph.

    Returns None when the prompt graph is unavailable (so callers can fall back
    to the lazy payload).
    """
    if hidden is None:
        return None
    dynprompt = getattr(hidden, "dynprompt", None)
    unique_id = getattr(hidden, "unique_id", None)
    if dynprompt is None or unique_id is None:
        return None
    try:
        node = dynprompt.get_node(unique_id)
        inputs = node.get("inputs", {})
    except Exception:
        return None
    out = []
    for key, val in inputs.items():
        if key in _FIXED_INPUTS:
            continue
        if _is_link(val):
            out.append((_slot_index(key), str(key)))
    out.sort(key=lambda t: (t[0], t[1]))
    _diag("connected_from_prompt", out)
    return out


def _connected_slots(hidden, values):
    """Ordered list of (slot_index, key) for connected value slots.

    Prefers the authoritative prompt-graph view; falls back to the lazy payload
    (where, for autogrow, only materialised/connected slots are present).
    """
    conn = _connected_from_prompt(hidden)
    if conn is not None:
        return conn
    items = _ordered_items(values)
    _diag("connected_from_payload_keys", [k for k, _ in items])
    return [(_slot_index(k), k) for k, _v in items]


def _value_for_slot(values, slot_index, position):
    """Retrieve the evaluated value for a chosen slot from the lazy payload.

    Tries (1) match on embedded slot index, then (2) positional fallback.
    """
    items = _ordered_items(values)
    by_index = {}
    for k, v in items:
        by_index.setdefault(_slot_index(k), v)
    if slot_index in by_index:
        return by_index[slot_index]
    if 0 <= position < len(items):
        return items[position][1]
    return None


def _decide(select, fallback, connected):
    """Pure decision over the ordered connected-slot list.

    ``select`` is 1-based and positional over the connected inputs.
    Returns (kind, slot_index, position) where kind is:
        "selected"  -> the selector named a connected slot
        "fallback"  -> selector invalid, flag true, use lowest connected
        "error"     -> selector invalid, flag false (or nothing connected)
    position is the 0-based index into ``connected`` (for value retrieval).
    """
    n = len(connected)
    sel = int(select)
    if 1 <= sel <= n:
        idx = connected[sel - 1][0]
        return ("selected", idx, sel - 1)
    if fallback and n >= 1:
        idx = connected[0][0]
        return ("fallback", idx, 0)
    return ("error", None, None)


class UniversalSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UniversalSwitch_NTo1",
            display_name="Universal Switch (N-to-1)",
            category="universal_switch",
            description=(
                "Forwards one of N wildcard inputs chosen by an integer "
                "selector. Input slots grow as they are connected; unselected "
                "branches are not evaluated. Optionally falls back to the "
                "lowest-numbered connected input when the selection is invalid."
            ),
            inputs=[
                io.Int.Input(
                    "select",
                    default=1,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip="1-based index (over connected inputs) to forward.",
                ),
                io.Boolean.Input(
                    "fallback_to_first_connected",
                    default=False,
                    tooltip=(
                        "If the selected slot is missing/disconnected: when on, "
                        "forward the lowest-numbered connected input; when off, "
                        "raise an explicit bounded error."
                    ),
                ),
                io.Autogrow.Input(
                    "values",
                    template=io.Autogrow.TemplatePrefix(
                        input=ANY.Input("value", lazy=True),
                        prefix="value",
                        min=1,
                        max=100,  # Autogrow._MaxNames hard cap in this comfy_api
                    ),
                ),
            ],
            outputs=[ANY.Output(display_name="selected")],
            hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
        )

    # --- lazy gate: only the chosen (or fallback) slot is ever evaluated -----
    @classmethod
    def check_lazy_status(cls, select, fallback_to_first_connected, values):
        connected = _connected_slots(getattr(cls, "hidden", None), values)
        kind, _idx, position = _decide(select, fallback_to_first_connected, connected)
        _diag("check_lazy", {"select": int(select), "fallback": bool(fallback_to_first_connected),
                             "kind": kind, "position": position, "n": len(connected)})
        if kind == "error":
            # Ask for nothing; execute() raises the bounded diagnostic.
            return []
        return [connected[position][1]]

    @classmethod
    def execute(cls, select, fallback_to_first_connected, values) -> io.NodeOutput:
        connected = _connected_slots(getattr(cls, "hidden", None), values)
        n = len(connected)
        sel = int(select)
        kind, slot_index, position = _decide(select, fallback_to_first_connected, connected)
        _diag("execute", {"select": sel, "fallback": bool(fallback_to_first_connected),
                          "kind": kind, "position": position, "n": n})
        if kind == "error":
            raise ValueError(
                f"UniversalSwitch: selector {sel} is out of range; valid range "
                f"is 1..{n} ({n} connected input slot(s)). "
                f"Enable 'fallback_to_first_connected' to forward slot 1 instead."
            )
        value = _value_for_slot(values, slot_index, position)
        if value is None:
            raise ValueError(
                f"UniversalSwitch: resolved slot (index {slot_index}) carried no "
                f"value; connect an input or choose a connected slot (1..{n})."
            )
        return io.NodeOutput(value)


class ScalarSource(io.ComfyNode):
    """Deterministic CPU scalar fixture.

    Emits a fixed integer and appends one line to runtime/exec_<label>.log on
    every execution, so whether this node actually ran is observable from disk.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UniversalSwitch_ScalarSource",
            display_name="Scalar Source (fixture)",
            category="universal_switch/test",
            description="Deterministic scalar; records a marker when executed.",
            inputs=[
                io.Int.Input("value", default=0, min=-(2**31), max=2**31 - 1, step=1),
                io.String.Input("label", default="A"),
            ],
            outputs=[io.Int.Output(display_name="value")],
            is_experimental=True,
        )

    @classmethod
    def fingerprint_inputs(cls, value, label):
        # Force re-execution every run so the side-effect marker is fresh.
        import time

        return f"{label}:{value}:{time.time_ns()}"

    @classmethod
    def execute(cls, value, label) -> io.NodeOutput:
        path = os.path.join(_runtime_dir(), f"exec_{label}.log")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{label}={value}\n")
        return io.NodeOutput(int(value))


class ScalarProbe(io.ComfyNode):
    """Terminal observation node: records the scalar it receives."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UniversalSwitch_ScalarProbe",
            display_name="Scalar Probe (fixture)",
            category="universal_switch/test",
            description="Writes the received scalar to runtime/result.txt.",
            inputs=[ANY.Input("value")],
            outputs=[],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, value) -> io.NodeOutput:
        path = os.path.join(_runtime_dir(), "result.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(str(value))
        return io.NodeOutput(ui=ui.PreviewText(f"probe received: {value}"))


class UniversalSwitchExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [UniversalSwitch, ScalarSource, ScalarProbe]


async def comfy_entrypoint() -> UniversalSwitchExtension:
    return UniversalSwitchExtension()
