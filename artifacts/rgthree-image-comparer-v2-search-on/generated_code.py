"""Angular Image Comparer custom node (server side).

Accepts two IMAGE batches plus independent positive playback rates, validates
them, and returns the two batches unchanged (a deterministic passthrough). It is
also an OUTPUT node: it saves each frame as a preview PNG and sends a bounded UI
payload (``a_images``/``b_images`` descriptor arrays plus counts and rates) to
the browser widget, which renders the angular comparison on a shared clock.

No DOM/framework imports live here; the frontend lives entirely under ``web/``.
There is no mutable module-global playback or widget state -- all per-run state
is local to the call, and the per-instance preview filename prefix is set in
``__init__``.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image

import folder_paths

from .comparer_core import validated_passthrough


_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _frame_to_pil(frame: "torch.Tensor") -> Image.Image:
    """Convert one HWC float frame to a PIL image for preview only.

    Clamps a detached CPU copy to [0,1]; the source tensor is never mutated, so
    the node's passthrough output keeps its original dtype, device and range.
    """
    arr = torch.clamp(frame.detach().to(device="cpu", dtype=torch.float32), 0.0, 1.0)
    arr = (arr.numpy() * 255.0 + 0.5).astype(np.uint8)
    mode = "RGB" if arr.shape[2] == 3 else "RGBA"
    return Image.fromarray(arr, mode=mode)


class AgentAngularImageComparer:
    """Interactive angular image comparer with independent per-side playback."""

    def __init__(self) -> None:
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        # Per-instance, set once; avoids any mutable module-global counter.
        self.prefix_append = "_agac_" + "".join(random.choice(_ALPHABET) for _ in range(8))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "fps_a": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.01, "max": 120.0, "step": 0.01},
                ),
                "fps_b": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.01, "max": 120.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image_a", "image_b")
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "image/compare"

    def _save_side(self, batch: "torch.Tensor", side: str):
        filename_prefix = f"agac_{side}{self.prefix_append}"
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, int(batch.shape[2]), int(batch.shape[1])
        )
        results = []
        for i in range(int(batch.shape[0])):
            pil = _frame_to_pil(batch[i])
            file = f"{filename}_{side}_{i:03d}_{counter:05d}.png"
            pil.save(os.path.join(full_output_folder, file), compress_level=self.compress_level)
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
        return results

    def compare(self, image_a, image_b, fps_a, fps_b):
        a, b, ra, rb = validated_passthrough(image_a, image_b, fps_a, fps_b)
        a_images = self._save_side(a, "a")
        b_images = self._save_side(b, "b")
        ui = {
            "a_images": a_images,
            "b_images": b_images,
            "fps_a": [ra],
            "fps_b": [rb],
            "count_a": [int(a.shape[0])],
            "count_b": [int(b.shape[0])],
        }
        # Passthrough returns the exact same tensors -> layout/order/dtype/device/range preserved.
        return {"ui": ui, "result": (a, b)}


NODE_CLASS_MAPPINGS = {
    "AgentAngularImageComparer": AgentAngularImageComparer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AgentAngularImageComparer": "Angular Image Comparer (Agent)",
}
