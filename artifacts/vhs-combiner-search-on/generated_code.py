"""DemoVideoCombiner — a narrow, local IMAGE-batch -> MP4 output node.

Encodes an IMAGE batch to an H.264 MP4 using the system-installed ffmpeg.
It deliberately does NOT depend on, install, or copy VideoHelperSuite: the
only runtime requirements are ComfyUI's own stack (torch/numpy) plus an
ffmpeg binary already present on the host.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

import folder_paths
from comfy_api.latest import ComfyExtension, io, ui


def _ensure_executable(path: str) -> str:
    try:
        os.chmod(path, os.stat(path).st_mode | 0o111)
    except Exception:
        pass
    return path


def _imageio_from_syspath() -> str | None:
    try:  # fallback if THIS interpreter can import imageio-ffmpeg
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return _ensure_executable(exe)
    except Exception:
        pass
    return None


def _discover_bundled_ffmpeg() -> str | None:
    """Find an imageio-ffmpeg bundled binary on disk, even when it lives in a
    sibling virtualenv that is not on this interpreter's sys.path. Never downloads."""
    here = os.path.dirname(os.path.abspath(__file__))
    wave_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    roots: list[str] = []
    roots += [p for p in os.environ.get("DEMO_VIDEO_COMBINER_FFMPEG_ROOTS", "").split(os.pathsep) if p]
    roots += [p for p in sys.path if p]
    roots += [wave_root]
    patterns = (
        "imageio_ffmpeg/binaries/ffmpeg-*",
        "*/venv/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*",
    )
    seen: set[str] = set()
    for root in roots:
        if not root or root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        for pat in patterns:
            for cand in sorted(glob.glob(os.path.join(root, pat))):
                if os.path.isfile(cand):
                    return _ensure_executable(cand)
    return None


def _find_ffmpeg() -> str:
    """Locate an already-installed ffmpeg binary. Never downloads one."""
    for env_name in ("DEMO_VIDEO_COMBINER_FFMPEG", "IMAGEIO_FFMPEG_EXE", "FFMPEG_BINARY"):
        override = os.environ.get(env_name)
        if override and os.path.isfile(override):
            return _ensure_executable(override)
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    exe = _imageio_from_syspath()
    if exe:
        return exe
    exe = _discover_bundled_ffmpeg()
    if exe:
        return exe
    raise RuntimeError(
        "DemoVideoCombiner requires an installed ffmpeg binary; none was found on PATH, "
        "via imageio-ffmpeg, or in a sibling virtualenv."
    )


def _find_ffprobe(ffmpeg_path: str) -> str | None:
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    cand = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe")
    if os.path.isfile(cand):
        return cand
    return None


class DemoVideoCombiner(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DemoVideoCombiner",
            display_name="Demo Video Combiner",
            category="demo/video",
            description=(
                "Write an IMAGE batch to an MP4 (H.264) using the system ffmpeg. "
                "Preserves the supplied RGB frame dimensions, writes to a run-owned "
                "output path, and returns both a UI preview and a machine-readable "
                "output reference. No VideoHelperSuite dependency."
            ),
            inputs=[
                io.Image.Input("images", tooltip="Frames to encode, in batch order."),
                io.Int.Input(
                    "frame_rate",
                    default=8,
                    min=1,
                    max=240,
                    step=1,
                    tooltip="Output frames per second.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="DemoVideo",
                    tooltip="Prefix for the run-owned output file.",
                ),
                io.Boolean.Input(
                    "ping_pong",
                    default=False,
                    tooltip=(
                        "When enabled, play the forward pass and then the interior "
                        "frames in reverse (boomerang) without duplicating the first "
                        "or last frame. N frames -> 2N-2 frames."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="video_reference",
                    tooltip="JSON reference to the written MP4 and its frame/duration evidence.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, frame_rate, filename_prefix, ping_pong=False) -> io.NodeOutput:
        ffmpeg = _find_ffmpeg()

        # IMAGE tensor is [B, H, W, C], float 0..1.
        arr = images.detach().cpu().numpy()
        if arr.ndim != 4:
            raise ValueError(f"Expected IMAGE batch [B,H,W,C]; got shape {tuple(arr.shape)}")
        b, h, w, c = arr.shape
        if b < 1:
            raise ValueError("DemoVideoCombiner received an empty image batch.")

        # Coerce to exactly 3 RGB channels while preserving the supplied H x W.
        if c >= 3:
            rgb = arr[..., :3]
        else:
            rgb = np.repeat(arr[..., :1], 3, axis=-1)
        frames = np.ascontiguousarray(
            np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        )

        # Frame ordering. Default: forward only (original behavior). Ping-pong:
        # forward pass, then the interior frames in reverse -- endpoints are NOT
        # duplicated, so N frames become 2N-2.
        if ping_pong and b >= 3:
            order = list(range(b)) + list(range(b - 2, 0, -1))
        else:
            order = list(range(b))
        sequence = np.ascontiguousarray(frames[order])
        written = len(order)

        # Run-owned output path with ComfyUI's counter-based naming.
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h
        )
        os.makedirs(full_output_folder, exist_ok=True)
        file = f"{filename}_{counter:05}_.mp4"
        out_path = os.path.join(full_output_folder, file)

        # yuv420p demands even dimensions; preserve exact W x H by choosing a
        # pixel format that does not require subsampling-friendly sizes.
        pix_fmt_out = "yuv420p" if (w % 2 == 0 and h % 2 == 0) else "yuv444p"

        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(int(frame_rate)),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", pix_fmt_out,
            "-r", str(int(frame_rate)),
            out_path,
        ]
        proc = subprocess.run(
            cmd,
            input=sequence.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not os.path.isfile(out_path):
            tail = proc.stderr.decode("utf-8", "replace")[-2000:]
            raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {tail}")

        # Self-reported evidence.
        reference = {
            "filename": file,
            "subfolder": subfolder,
            "type": "output",
            "fullpath": out_path,
            "format": "video/mp4",
            "codec": "h264",
            "pix_fmt": pix_fmt_out,
            "frame_rate": int(frame_rate),
            "width": int(w),
            "height": int(h),
            "ping_pong": bool(ping_pong),
            "input_frame_count": int(b),
            "frame_count_written": int(written),
            "frame_order": [int(i) for i in order],
            "expected_duration_s": round(written / float(frame_rate), 6),
        }

        # Independent evidence: probe the file that was actually written.
        ffprobe = _find_ffprobe(ffmpeg)
        if ffprobe:
            p = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-count_frames",
                    "-select_streams", "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,width,height,duration,r_frame_rate,avg_frame_rate",
                    "-show_entries", "format=duration",
                    "-of", "json",
                    out_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if p.returncode == 0:
                try:
                    info = json.loads(p.stdout.decode("utf-8", "replace"))
                    st = (info.get("streams") or [{}])[0]
                    reference["probed_nb_frames"] = st.get("nb_read_frames")
                    reference["probed_width"] = st.get("width")
                    reference["probed_height"] = st.get("height")
                    reference["probed_stream_duration"] = st.get("duration")
                    reference["probed_format_duration"] = (info.get("format") or {}).get("duration")
                    reference["probed_r_frame_rate"] = st.get("r_frame_rate")
                    reference["probed_avg_frame_rate"] = st.get("avg_frame_rate")
                except Exception as exc:  # pragma: no cover - probe best-effort
                    reference["probe_error"] = str(exc)

        # Retain the evidence as a persisted sidecar next to the playable file.
        evidence_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.evidence.json")
        try:
            with open(evidence_path, "w", encoding="utf-8") as fh:
                json.dump(reference, fh, indent=2, sort_keys=True)
            reference["evidence_file"] = os.path.basename(evidence_path)
        except Exception as exc:  # pragma: no cover
            reference["evidence_write_error"] = str(exc)

        # Inspectable UI filename/preview payload (video preview).
        preview = {
            "filename": file,
            "subfolder": subfolder,
            "type": "output",
            "format": "video/mp4",
            "frame_rate": int(frame_rate),
        }
        ui_payload = ui.PreviewVideo(values=[preview])

        # Machine-readable output reference on the STRING socket.
        return io.NodeOutput(json.dumps(reference), ui=ui_payload)


class DemoVideoCombinerExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DemoVideoCombiner]


async def comfy_entrypoint() -> DemoVideoCombinerExtension:
    return DemoVideoCombinerExtension()
