"""Deterministic, security-scoped image-directory iterator for ComfyUI.

Iterates image files found under a user-selected directory that is resolved
*relative to ComfyUI's input directory* (treated as a frozen, read-only
filesystem root). Enumeration is recursive and deterministic (natural sort),
filenames are returned POSIX-style and relative to the selected directory, and
traversal / escape attempts are rejected.

Revision 2 adds:

* ``end_behavior`` — exactly one of ``wrap`` / ``stop`` / ``error`` (default
  ``stop``). It only changes what happens once the cursor runs past the last
  item: ``wrap`` returns to index zero, ``stop`` clamps to the final item
  without presenting it as a fresh item, and ``error`` raises a bounded named
  exhaustion diagnostic.
* ``extensions`` — a user-editable allowlist of file extensions, defaulting to
  PNG only. Changing it deterministically changes which files are eligible for
  the sequence. The security boundary is unchanged by this widget.

Design notes that satisfy the contract:

* State is NOT held in any module-global, process-global singleton, or hidden
  shared file. The iteration cursor is the node's own ``index`` widget, which is
  serialized inside the workflow and is independent per node instance. Two
  instances in one workflow therefore have fully independent cursors by
  construction, and a saved index persists across reloads with the node.
* Each emitted IMAGE is a float32 tensor in BHWC layout with exactly one batch
  item, the source image's exact height/width/channel count, and values in
  [0, 1].
* Unsupported files never enter the iteration sequence. Symlinks whose resolved
  target escapes the input root are excluded. A corrupt *selected* image raises
  a bounded diagnostic that names its relative path; it is never silently
  skipped or duplicated.
* CPU-only and dependency-light: standard library + numpy + torch + Pillow,
  all of which ComfyUI already ships.
"""

import os
import re

import numpy as np
import torch
from PIL import Image

import folder_paths

from comfy_api.latest import ComfyExtension, io, ui

# Default allowlist: PNG only. The ``extensions`` widget starts here and the
# user may widen it. Kept as the single documented default.
DEFAULT_EXTENSIONS = ("png",)

END_BEHAVIORS = ("wrap", "stop", "error")

# Keep diagnostics bounded so a malformed file cannot flood logs/UI.
_MAX_DIAG_CHARS = 200

_NUM_RE = re.compile(r"(\d+)")


def _natural_key(name):
    """Natural-sort key for one path component.

    Splits into digit / non-digit runs. Each run becomes a (type, value) tuple
    so ints and strings are never compared directly (which would raise on
    Python 3). Digits sort numerically; text sorts case-insensitively.
    """
    key = []
    for part in _NUM_RE.split(name):
        if part == "":
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


def _path_sort_key(posix_path):
    """Natural-sort key for a POSIX relative path, component by component.

    Sorting per component keeps the directory tree grouped correctly and still
    applies natural (img1 < img2 < img10) ordering inside each level.
    """
    return [_natural_key(component) for component in posix_path.split("/")]


def _parse_extensions(raw):
    """Parse the user-editable allowlist into a normalized tuple of ``.ext``.

    Accepts comma / semicolon / whitespace separation, case-insensitively, with
    or without a leading dot. De-duplicates while preserving order. Raises a
    bounded ValueError if the result is empty (an empty allowlist is almost
    always a mistake and would otherwise silently match nothing).
    """
    tokens = re.split(r"[,;\s]+", (raw or "").strip().lower())
    exts = []
    for token in tokens:
        if not token:
            continue
        ext = token if token.startswith(".") else "." + token
        if ext not in exts:
            exts.append(ext)
    if not exts:
        raise ValueError(
            "FrozenInputDirImageIterator: extension allowlist is empty; "
            "give at least one extension (e.g. 'png')."
        )
    return tuple(exts)


def _within_root(root, path):
    """True iff ``path`` is the root itself or lives beneath it."""
    return path == root or path.startswith(root + os.sep)


def _frozen_root():
    """The frozen filesystem root: ComfyUI's input directory, realpath'd."""
    return os.path.realpath(folder_paths.get_input_directory())


def _resolve_selected_dir(root, directory):
    """Validate the user-selected directory and return its absolute realpath.

    Rejects absolute paths, drive-qualified paths, ``..`` traversal, and any
    selection whose resolved path escapes the frozen root. Raises ValueError
    with a bounded message on any violation.
    """
    raw = (directory or "").strip()
    normalized = raw.replace("\\", "/")

    if normalized in ("", "."):
        components = []
    else:
        if os.path.isabs(raw) or (len(raw) >= 2 and raw[1] == ":"):
            raise ValueError(
                "FrozenInputDirImageIterator: absolute paths are not allowed; "
                "give a directory relative to ComfyUI's input folder."
            )
        components = [c for c in normalized.split("/") if c not in ("", ".")]
        if any(c == ".." for c in components):
            raise ValueError(
                "FrozenInputDirImageIterator: '..' traversal is not allowed in "
                "the selected directory."
            )

    selected_rel = "/".join(components)
    selected_abs = os.path.realpath(os.path.join(root, *components))

    if not _within_root(root, selected_abs):
        raise ValueError(
            "FrozenInputDirImageIterator: selected directory resolves outside "
            "the input root and was rejected."
        )
    if not os.path.isdir(selected_abs):
        raise ValueError(
            "FrozenInputDirImageIterator: directory not found under input root: "
            "'%s'." % (selected_rel or ".")
        )
    return selected_abs, selected_rel


def _enumerate_sequence(root, selected_abs, extensions):
    """Return the deterministic, natural-sorted list of POSIX relative paths.

    Relative to the selected directory. Recurses without following symlinked
    directories. Excludes files whose extension is not in ``extensions`` and
    symlinks (or any entry) whose resolved target escapes the frozen root.
    """
    allow = tuple(extensions)
    entries = []
    for dirpath, dirnames, filenames in os.walk(selected_abs, followlinks=False):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in allow:
                continue  # not in allowlist: never enters the sequence
            abspath = os.path.join(dirpath, name)
            real = os.path.realpath(abspath)
            if not _within_root(root, real):
                continue  # symlink / entry escaping the root: excluded
            rel = os.path.relpath(abspath, selected_abs).replace(os.sep, "/")
            entries.append(rel)
    entries.sort(key=_path_sort_key)
    return entries


def _resolve_cursor(raw_index, count, end_behavior, selected_rel):
    """Map the serialized ``index`` onto a valid cursor per ``end_behavior``.

    ``count`` is guaranteed > 0 by the caller. Returns the resolved zero-based
    index. For ``error`` past the end it raises a bounded named diagnostic.
    """
    idx = int(raw_index)
    if idx < 0:
        idx = 0

    if end_behavior == "wrap":
        return idx % count
    if end_behavior == "stop":
        # Bounded: clamp to the final item; never re-enters the sequence at 0.
        return idx if idx < count else count - 1
    if end_behavior == "error":
        if idx >= count:
            raise RuntimeError(
                "FrozenInputDirImageIterator: iteration exhausted — index %d is "
                "out of range for %d image(s) under '%s' (end_behavior='error')."
                % (idx, count, selected_rel or ".")
            )
        return idx
    raise ValueError(
        "FrozenInputDirImageIterator: unknown end_behavior %r; expected one of "
        "%r." % (end_behavior, END_BEHAVIORS)
    )


def _load_image_bhwc(abspath, posix_rel, idx):
    """Load one image as a float32 BHWC tensor in [0, 1], preserving channels.

    Raises a bounded RuntimeError naming ``posix_rel`` if the file cannot be
    decoded.
    """
    try:
        with Image.open(abspath) as im:
            im.load()
            arr = np.array(im)
    except Exception as exc:  # corrupt / unreadable selected image
        detail = str(exc)
        if len(detail) > _MAX_DIAG_CHARS:
            detail = detail[:_MAX_DIAG_CHARS] + "..."
        raise RuntimeError(
            "FrozenInputDirImageIterator: failed to decode selected image "
            "'%s' (sequence index %d): %s: %s"
            % (posix_rel, idx, type(exc).__name__, detail)
        )

    if arr.ndim == 2:
        arr = arr[:, :, None]  # grayscale H,W -> H,W,1 (preserve channel count)
    elif arr.ndim != 3:
        raise RuntimeError(
            "FrozenInputDirImageIterator: unexpected array rank %d for image "
            "'%s'." % (arr.ndim, posix_rel)
        )

    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max_value
    else:
        arr = arr.astype(np.float32)

    arr = np.clip(arr, 0.0, 1.0)
    tensor = torch.from_numpy(arr)[None, ...]  # BHWC, one batch item
    return tensor.contiguous()


class FrozenInputDirImageIterator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FrozenInputDirImageIterator",
            display_name="Frozen Input Dir Image Iterator",
            category="io/iterate",
            description=(
                "Iterate image files under a directory relative to ComfyUI's "
                "input folder, in deterministic natural-sort order. Emits one "
                "image per call with its relative filename and zero-based "
                "index. The cursor is this node's serialized 'index' widget, so "
                "each instance iterates independently and no global state is "
                "used. 'end_behavior' controls past-the-end handling; "
                "'extensions' is the eligibility allowlist."
            ),
            inputs=[
                io.String.Input(
                    "directory",
                    default="iterator-fixtures",
                    tooltip=(
                        "Directory relative to ComfyUI's input folder. "
                        "Absolute paths and '..' are rejected."
                    ),
                ),
                io.String.Input(
                    "extensions",
                    default="png",
                    tooltip=(
                        "Allowlist of file extensions eligible for iteration, "
                        "comma/space separated (e.g. 'png, jpg'). Defaults to "
                        "PNG only. Changing it changes eligibility; it does not "
                        "affect the security boundary."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    step=1,
                    control_after_generate=True,
                    tooltip=(
                        "Zero-based cursor into the sorted sequence. This value "
                        "is serialized with the node, so it persists across "
                        "save/reload and is independent per instance. Set the "
                        "control to 'increment' to advance one image per run."
                    ),
                ),
                io.Combo.Input(
                    "end_behavior",
                    options=list(END_BEHAVIORS),
                    default="stop",
                    tooltip=(
                        "What to do once the cursor runs past the last image: "
                        "'wrap' returns to index 0, 'stop' clamps to the final "
                        "item (bounded), 'error' raises a named exhaustion "
                        "diagnostic."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="filename"),
                io.Int.Output(display_name="index"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, directory, extensions, index, end_behavior):
        # Re-run whenever the selection, allowlist, cursor, or policy changes.
        return (directory, extensions, int(index), end_behavior)

    @classmethod
    def execute(cls, directory, extensions, index, end_behavior) -> io.NodeOutput:
        root = _frozen_root()
        allow = _parse_extensions(extensions)
        selected_abs, selected_rel = _resolve_selected_dir(root, directory)
        sequence = _enumerate_sequence(root, selected_abs, allow)
        count = len(sequence)
        if count == 0:
            raise RuntimeError(
                "FrozenInputDirImageIterator: no images matching %r found "
                "under '%s'." % (list(allow), selected_rel or ".")
            )

        idx = _resolve_cursor(index, count, end_behavior, selected_rel)
        posix_rel = sequence[idx]
        abspath = os.path.join(selected_abs, *posix_rel.split("/"))
        image = _load_image_bhwc(abspath, posix_rel, idx)

        info = (
            "index=%d / count=%d  end_behavior=%s\nallow=%s\nfile=%s\n"
            "shape=%s dtype=%s\nrange=[%.3f, %.3f]"
            % (
                idx,
                count,
                end_behavior,
                list(allow),
                posix_rel,
                tuple(image.shape),
                str(image.dtype),
                float(image.min()),
                float(image.max()),
            )
        )
        return io.NodeOutput(
            image,
            posix_rel,
            idx,
            count,
            ui=ui.PreviewText(info),
        )


class FrozenInputDirIteratorExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [FrozenInputDirImageIterator]


async def comfy_entrypoint() -> FrozenInputDirIteratorExtension:
    return FrozenInputDirIteratorExtension()
