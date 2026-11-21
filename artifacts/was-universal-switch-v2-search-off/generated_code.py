"""Pure selection logic for the Universal N-to-1 Switch (draft).

No ComfyUI / torch imports here, so the rules can be exercised in isolation.

Contract
--------
* Slots are named input_1 .. input_<MAX_SLOTS>; the selector is 1-based and
  addresses a slot by that number (select=2 -> input_2).
* A selection is valid only when 1 <= select <= MAX_SLOTS AND that slot is
  connected. Anything else yields an explicit diagnostic whose length is
  bounded by MAX_DIAG_CHARS and which lists at most MAX_LISTED slot numbers.
* Revision 2 — fallback_to_first_connected:
  - False (default): an out-of-range ("missing") or disconnected selection
    keeps raising the bounded diagnostic exactly as in revision 1.
  - True: an out-of-range or disconnected selection resolves to the
    LOWEST-numbered connected slot instead. If no slot is connected at all,
    a bounded diagnostic is raised. A non-integer selector is never a
    "missing slot" and always raises.
* Lazy evaluation: only the resolved slot is ever requested; on an invalid
  selection nothing is requested, so no candidate branch is evaluated.
"""

MAX_SLOTS = 32
SLOT_PREFIX = "input_"
SLOT_NAMES = [f"{SLOT_PREFIX}{i}" for i in range(1, MAX_SLOTS + 1)]
GROUP_ID = "candidates"
MAX_DIAG_CHARS = 240
MAX_LISTED = 8
TAG = "UniversalNSwitch"


class SelectionError(ValueError):
    """Raised for an out-of-range or disconnected selection."""


def slot_name(index: int) -> str:
    return f"{SLOT_PREFIX}{index}"


def unwrap_lazy(value):
    """check_lazy_status receives dynamic (Autogrow) values as
    (value, original_flat_key) tuples (ComfyUI core _io.py:1678, 2125-2126).
    Only used in check_lazy_status; execute receives plain values, so a user
    value that happens to be a tuple is never unwrapped there.
    Returns (flat_key_or_None, value)."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        return value[1], value[0]
    return None, value


def connected_indices(candidates) -> list:
    out = []
    for key in (candidates or {}):
        if isinstance(key, str) and key.startswith(SLOT_PREFIX):
            try:
                out.append(int(key[len(SLOT_PREFIX):]))
            except ValueError:
                continue
    return sorted(out)


def _bounded(msg: str) -> str:
    if len(msg) <= MAX_DIAG_CHARS:
        return msg
    return msg[: MAX_DIAG_CHARS - 3] + "..."


def _fmt(indices: list) -> str:
    shown = ",".join(str(i) for i in indices[:MAX_LISTED])
    if len(indices) > MAX_LISTED:
        shown += f",...(+{len(indices) - MAX_LISTED} more)"
    return "[" + shown + "]"


def diagnose(select, candidates):
    """None when the selection is usable as-is, otherwise a bounded diagnostic."""
    if isinstance(select, bool) or not isinstance(select, int):
        return _bounded(f"{TAG}: selector must be an integer, got {type(select).__name__}")
    conn = connected_indices(candidates)
    if select < 1 or select > MAX_SLOTS:
        return _bounded(
            f"{TAG}: selector {select} is out of range 1..{MAX_SLOTS}; "
            f"connected slots {_fmt(conn)}"
        )
    if select not in conn:
        return _bounded(
            f"{TAG}: selector {select} points at disconnected slot {slot_name(select)}; "
            f"connected slots {_fmt(conn)}"
        )
    return None


def resolve(select, candidates, fallback=False):
    """Return (slot_index, None) for the slot to use, or (None, diagnostic).

    Only ever names ONE slot, so at most one branch is evaluated."""
    err = diagnose(select, candidates)
    if err is None:
        return select, None
    if isinstance(select, bool) or not isinstance(select, int):
        return None, err  # a malformed selector is not a missing slot
    if fallback is True:
        conn = connected_indices(candidates)
        if conn:
            return conn[0], None
        return None, _bounded(
            f"{TAG}: selector {select} unusable and fallback_to_first_connected "
            f"found no connected input"
        )
    return None, err


def lazy_request(select, candidates, fallback=False) -> list:
    """Input keys to evaluate next: only the resolved slot, or nothing."""
    index, err = resolve(select, candidates, fallback)
    if err is not None:
        return []
    key, value = unwrap_lazy(candidates[slot_name(index)])
    if value is None:
        return [key if key else f"{GROUP_ID}.{slot_name(index)}"]
    return []


def pick(select, candidates, fallback=False):
    """Return (value, slot_index) or raise a bounded SelectionError."""
    index, err = resolve(select, candidates, fallback)
    if err is not None:
        raise SelectionError(err)
    value = candidates[slot_name(index)]
    if value is None:
        raise SelectionError(_bounded(
            f"{TAG}: selected slot {slot_name(index)} produced no value (branch not evaluated)"
        ))
    return value, index
