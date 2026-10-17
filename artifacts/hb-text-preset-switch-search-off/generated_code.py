"""DemoFarm11 Text Preset Switch - CPU-only text preset switch node.

A single node that exposes a `preset` combo plus a `Manage Presets` button
(the button and the management dialog live in the frontend extension under
./web). The ordered preset entries and the selected label are kept as
workflow-local serialized widget state:

  * ``preset``        - combo, the selected label (serialized)
  * ``presets_json``  - JSON array of ``[name, value]`` pairs, ordered
                        (serialized; edited through the Manage Presets dialog)

The Python side looks up the selected label's value and returns three
deterministic outputs: the raw string, an integer conversion, and a boolean
derived from that integer. CPU only; no models, GPU, network, or external
dependencies.
"""

import json

from comfy_api.latest import ComfyExtension, io, ui

# Initial, ordered preset entries. Kept in one place so the Python default and
# the documented behaviour cannot drift apart.
DEFAULT_ENTRIES = [["Alpha", "1"], ["Beta", "2"], ["Gamma", "3"]]
DEFAULT_PRESETS_JSON = json.dumps(DEFAULT_ENTRIES)
DEFAULT_LABELS = [name for name, _ in DEFAULT_ENTRIES]


def _parse_entries(presets_json):
    """Parse the serialized entry list into a list of [name, value] str pairs."""
    try:
        parsed = json.loads(presets_json)
    except Exception:
        return [list(pair) for pair in DEFAULT_ENTRIES]
    entries = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                entries.append([str(item[0]), str(item[1])])
    if not entries:
        return [list(pair) for pair in DEFAULT_ENTRIES]
    return entries


def _value_for_label(entries, label):
    """Return the value string for the first entry matching ``label``."""
    target = str(label)
    for name, value in entries:
        if name == target:
            return str(value)
    return ""


def _to_int(raw):
    """Deterministic int conversion: invalid numeric text -> 0."""
    try:
        return int(str(raw).strip())
    except Exception:
        return 0


class DemoFarm11TextPresetSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DemoFarm11_TextPresetSwitch",
            display_name="Text Preset Switch (DemoFarm11)",
            category="demofarm11/text",
            description=(
                "CPU-only text preset switch. Pick a preset label; outputs its "
                "value as a raw string, an integer conversion, and a boolean "
                "derived from that integer."
            ),
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=list(DEFAULT_LABELS),
                    default=DEFAULT_LABELS[0],
                    tooltip="Selected preset label (managed via Manage Presets).",
                ),
                io.String.Input(
                    "presets_json",
                    default=DEFAULT_PRESETS_JSON,
                    multiline=False,
                    tooltip=(
                        "Ordered preset entries as JSON [[name, value], ...]. "
                        "Edited through the Manage Presets dialog."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="value_string"),
                io.Int.Output(display_name="value_int"),
                io.Boolean.Output(display_name="value_bool"),
            ],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, preset=None, presets_json=None, **kwargs):
        # Accept any selected label: the combo's options are managed at runtime
        # by the frontend (presets can be added/renamed), so membership in the
        # schema's initial option list must not gate execution.
        return True

    @classmethod
    def execute(cls, preset, presets_json) -> io.NodeOutput:
        entries = _parse_entries(presets_json)
        raw = _value_for_label(entries, preset)
        value_int = _to_int(raw)
        value_bool = bool(value_int)
        preview = json.dumps([raw, value_int, value_bool])
        return io.NodeOutput(
            raw, value_int, value_bool, ui=ui.PreviewText(preview)
        )


class DemoFarm11TextPresetSwitchExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DemoFarm11TextPresetSwitch]


async def comfy_entrypoint() -> DemoFarm11TextPresetSwitchExtension:
    return DemoFarm11TextPresetSwitchExtension()
