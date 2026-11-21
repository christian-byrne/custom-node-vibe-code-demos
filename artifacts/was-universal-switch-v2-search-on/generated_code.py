"""Independent universal N-to-1 switch (unpublished draft).

IndieSwitch_AnyIndexN1
  * accepts ANY Comfy type on every candidate slot (io.AnyType, socket "*")
  * candidate slots grow as connections are made (core io.Autogrow,
    TemplateNames slot_1..slot_32, min 2 visible)
  * integer `selector` is 1-based and picks slot_<selector> (2 -> slot_2)
  * every candidate slot is lazy; check_lazy_status requests ONLY the
    selected slot, so branches feeding only unselected slots never execute
  * out-of-range or disconnected selections raise a bounded ValueError
  * rev2: boolean `fallback_to_first_connected` (widget 2, after selector).
    When true and the selected slot is out of range or disconnected, the
    lowest-numbered CONNECTED slot is used instead, and only that slot is
    requested lazily. When false, the bounded error above is unchanged.
    A non-integer selector, or fallback with nothing connected, still errors.

Written independently against this install's comfy_api.latest (V3).
No code from WAS Node Suite or rgthree is included; see REFERENCES.md.
"""

from comfy_api.latest import ComfyExtension, io

NODE_ID = "IndieSwitch_AnyIndexN1"
SLOT_PREFIX = "slot_"
MIN_SLOTS = 2
MAX_SLOTS = 32
SLOT_NAMES = [f"{SLOT_PREFIX}{i}" for i in range(1, MAX_SLOTS + 1)]
_MAX_DIAG_CHARS = 240  # hard bound on any diagnostic this node raises


def _bounded(msg):
    msg = " ".join(str(msg).split())
    if len(msg) > _MAX_DIAG_CHARS:
        msg = msg[: _MAX_DIAG_CHARS - 3] + "..."
    return msg


def _slot_name(selector):
    return f"{SLOT_PREFIX}{selector}"


def _unwrap(entry, name):
    """check_lazy_status gets autogrow leaves as (value, flat_key) tuples
    (comfy_api/latest/_io.py:2125-2126, create_dynamic_tuple)."""
    if (
        isinstance(entry, tuple)
        and len(entry) == 2
        and isinstance(entry[1], str)
        and (entry[1] == name or entry[1].endswith("." + name))
    ):
        return entry[0], entry[1]
    return entry, f"values.{name}"


def _connected(values):
    names = [k for k in (values or {}) if str(k).startswith(SLOT_PREFIX)]

    def _key(n):
        try:
            return int(n[len(SLOT_PREFIX):])
        except ValueError:
            return 1 << 30

    return sorted(names, key=_key)


def _diagnose(selector, values):
    """Bounded diagnostic for an unusable selection, else None."""
    if isinstance(selector, bool) or not isinstance(selector, int):
        return _bounded(f"[{NODE_ID}] selector must be an integer, "
                        f"got {type(selector).__name__}.")
    connected = _connected(values)
    shown = ",".join(connected[:8]) + (",..." if len(connected) > 8 else "")
    if selector < 1 or selector > MAX_SLOTS:
        return _bounded(
            f"[{NODE_ID}] selector {selector} is out of range; valid slots "
            f"are 1..{MAX_SLOTS} (connected: {shown or 'none'}).")
    if _slot_name(selector) not in (values or {}):
        return _bounded(
            f"[{NODE_ID}] selector {selector} points at "
            f"{_slot_name(selector)}, which is disconnected "
            f"(connected: {shown or 'none'}).")
    return None


def _resolve(selector, values, fallback):
    """Return (slot_name, None) for the slot to pass through, or
    (None, bounded_diagnostic). Fallback never applies to a non-int selector."""
    entries = values or {}
    diag = _diagnose(selector, entries)
    if diag is None:
        return _slot_name(selector), None
    if not fallback or isinstance(selector, bool) or not isinstance(selector, int):
        return None, diag
    connected = _connected(entries)
    if not connected:
        return None, _bounded(
            f"[{NODE_ID}] selector {selector} is unusable and "
            f"fallback_to_first_connected found no connected slot.")
    return connected[0], None


class IndieAnyIndexSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=NODE_ID,
            display_name="Any Index Switch (N-to-1, lazy)",
            category="utils/indie_switch",
            description=(
                "Universal N-to-1 switch: passes through the slot whose number "
                "equals `selector` (2 -> slot_2). Any type, grows slots as you "
                "connect, evaluates only the selected branch, and raises a "
                "bounded error for out-of-range or disconnected selections."),
            search_aliases=["any switch", "index switch", "n to 1 switch"],
            inputs=[
                io.Int.Input(
                    "selector", default=1,
                    min=-2147483648, max=2147483647, step=1,
                    tooltip=f"Slot number to pass through (slot_<selector>); "
                            f"valid 1..{MAX_SLOTS}."),
                io.Boolean.Input(
                    "fallback_to_first_connected", default=False,
                    tooltip="If the selected slot is out of range or "
                            "disconnected: true = use the lowest-numbered "
                            "connected slot; false = raise a bounded error."),
                io.Autogrow.Input(
                    "values",
                    template=io.Autogrow.TemplateNames(
                        input=io.AnyType.Input(
                            "slot", optional=True, lazy=True,
                            tooltip="Candidate branch (any type)."),
                        names=SLOT_NAMES, min=MIN_SLOTS),
                ),
            ],
            outputs=[io.AnyType.Output(display_name="value")],
        )

    @classmethod
    def check_lazy_status(cls, selector, fallback_to_first_connected=False,
                          values=None):
        entries = values or {}
        name, diag = _resolve(selector, entries, fallback_to_first_connected)
        if diag is not None:
            return []  # request nothing; execute() raises the diagnostic
        value, flat_key = _unwrap(entries[name], name)
        return [flat_key] if value is None else []

    @classmethod
    def execute(cls, selector, fallback_to_first_connected=False,
                values=None) -> io.NodeOutput:
        entries = values or {}
        name, diag = _resolve(selector, entries, fallback_to_first_connected)
        if diag is not None:
            raise ValueError(diag)
        return io.NodeOutput(entries[name])


class IndieSwitchExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [IndieAnyIndexSwitch]


async def comfy_entrypoint() -> IndieSwitchExtension:
    return IndieSwitchExtension()
