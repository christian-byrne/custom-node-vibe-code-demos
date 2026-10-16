"""DemoVideoCombiner — a narrow, local-only output node.

Takes an IMAGE batch, a frame-rate integer and a filename prefix, then writes
the frames in order to an MP4 using the system-installed ffmpeg binary. It
preserves the supplied RGB frame dimensions, writes to a run-owned path under
ComfyUI's output directory, declares itself an output node, and returns both an
inspectable UI preview payload (filename/subfolder/type for the frontend) and a
machine-readable output reference (a JSON string describing the written file).

Deliberately does NOT depend on, install, or copy VideoHelperSuite. The only
external dependency is the already-installed `ffmpeg` executable, invoked as a
subprocess.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np

import folder_paths
from comfy_api.latest import ComfyExtension, io, ui


def _resolve_ffmpeg() -> str:
    """Locate the already-installed ffmpeg binary. No bundling, no download."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # Honour an explicit override if the host sets one.
    env_exe = os.environ.get("FFMPEG_BINARY") or os.environ.get("IMAGEIO_FFMPEG_EXE")
    if env_exe and os.path.isfile(env_exe) and os.access(env_exe, os.X_OK):
        return env_exe
    raise RuntimeError(
        "ffmpeg executable not found on PATH. DemoVideoCombiner requires the "
        "system-installed ffmpeg; it does not bundle one."
    )


class DemoVideoCombiner(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DemoVideoCombiner",
            display_name="Demo Video Combiner",
            category="demo/video",
            description=(
                "Encode an IMAGE batch to an MP4 using the system ffmpeg, "
                "preserving the supplied RGB frame dimensions."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="IMAGE batch; frames are written to the MP4 in batch order.",
                ),
                io.Int.Input(
                    "frame_rate",
                    default=8,
                    min=1,
                    max=240,
                    step=1,
                    tooltip="Playback frames per second.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="DemoVideoCombiner",
                    tooltip="Filename prefix under the run output folder.",
                ),
                io.Boolean.Input(
                    "ping_pong",
                    default=False,
                    tooltip=(
                        "When enabled, append the interior frames in reverse "
                        "after the forward pass (endpoints are not duplicated)."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="video_reference",
                    tooltip="JSON reference: filename, subfolder, path, dims, frame_count, fps, duration.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, frame_rate, filename_prefix, ping_pong=False) -> io.NodeOutput:
        # IMAGE is a torch tensor [B, H, W, C], float 0..1.
        arr = images
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f"Expected an IMAGE batch of shape [B,H,W,C]; got {arr.shape}")

        batch, height, width, channels = arr.shape
        if batch < 1:
            raise ValueError("IMAGE batch is empty; nothing to encode.")

        # Preserve exactly 3 RGB channels at the supplied dimensions.
        if channels >= 3:
            rgb = arr[..., :3]
        else:
            rgb = np.repeat(arr[..., :1], 3, axis=-1)
        frames = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

        # Ping-pong: forward pass, then interior frames reversed. Endpoints
        # (first and last) are NOT duplicated. For N input frames this emits
        # N + (N - 2) = 2N - 2 frames; six in -> ten out (0..5 then 4,3,2,1).
        if ping_pong and batch > 2:
            interior_reversed = frames[-2:0:-1]
            frames = np.concatenate([frames, interior_reversed], axis=0)
        frames = np.ascontiguousarray(frames)

        emitted = int(frames.shape[0])
        fps = int(frame_rate)

        # Run-owned output path: counter-scoped, under ComfyUI's output directory.
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height
        )
        os.makedirs(full_output_folder, exist_ok=True)
        file = f"{filename}_{counter:05}_.mp4"
        output_path = os.path.join(full_output_folder, file)

        # yuv420p needs even dimensions; fall back to yuv444p so odd-sized frames
        # keep their exact supplied dimensions rather than being padded/cropped.
        even_dims = (width % 2 == 0) and (height % 2 == 0)
        pix_fmt = "yuv420p" if even_dims else "yuv444p"

        ffmpeg = _resolve_ffmpeg()
        cmd = [
            ffmpeg,
            "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", pix_fmt,
            "-r", str(fps),
            output_path,
        ]
        proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
        if proc.returncode != 0 or not os.path.exists(output_path):
            stderr = proc.stderr.decode("utf-8", "replace")[-2000:]
            raise RuntimeError(
                f"ffmpeg failed (exit {proc.returncode}) encoding {file}:\n{stderr}"
            )

        duration_seconds = round(emitted / float(fps), 6)
        reference = {
            "filename": file,
            "subfolder": subfolder,
            "type": "output",
            "fullpath": output_path,
            "format": "video/mp4",
            "width": int(width),
            "height": int(height),
            "frame_count": emitted,
            "input_frame_count": int(batch),
            "ping_pong": bool(ping_pong),
            "fps": fps,
            "duration_seconds": duration_seconds,
        }

        preview = ui.PreviewVideo(
            values=[ui.SavedResult(file, subfolder, io.FolderType.output)]
        )
        return io.NodeOutput(json.dumps(reference), ui=preview)


class DemoVideoCombinerExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DemoVideoCombiner]


async def comfy_entrypoint() -> DemoVideoCombinerExtension:
    return DemoVideoCombinerExtension()
