"""Scheduled CFG Guider custom node (ComfyUI V3 node API).

Exposes exactly one GUIDER-producing node. The guider interpolates the effective
CFG scale linearly across normalized sampling progress and subclasses
``comfy.samplers.CFGGuider`` so it stays compatible with ComfyUI's
custom-sampling contract (``SamplerCustomAdvanced`` and friends).

Design guarantees:
  * No mutable module-global state. The default sampling-function reference, the
    per-run sigma schedule, the schedule parameters and the conditioning all
    live on the guider *instance*.
  * Caller-owned conditioning values and tensors are never mutated. ``set_conds``
    delegates to ``comfy.sampler_helpers.convert_cond``, which copies the
    metadata dict and only *references* the caller's tensors.
  * The CFG prediction preserves the input's batch, dtype, device and shape,
    because it delegates to the same ``sampling_function`` the base guider uses.
"""

from __future__ import annotations

import torch

import comfy.samplers

from comfy_api.latest import ComfyExtension, io

# Dual import: ComfyUI loads this file as a package submodule (relative import
# works); the contract tests import it as a top-level module with the pack dir on
# sys.path (absolute import works). Both paths reach the same pure logic.
try:  # pragma: no cover - import plumbing
    from .schedule import (
        ScheduleError,
        linear_cfg_schedule,
        progress_from_sigmas,
        validate_schedule,
    )
except ImportError:  # pragma: no cover - import plumbing
    from schedule import (  # type: ignore
        ScheduleError,
        linear_cfg_schedule,
        progress_from_sigmas,
        validate_schedule,
    )


class ScheduledCFGGuider(comfy.samplers.CFGGuider):
    """A ``CFGGuider`` whose cfg scale follows a per-instance linear schedule.

    The full sigma schedule is captured per ``sample()`` call (instance-local),
    and ``predict_noise()`` recomputes the effective scale for the current step
    before delegating to the exact CFG prediction path the base guider uses.
    """

    def __init__(self, model_patcher, start_scale, end_scale, start_fraction, end_fraction):
        super().__init__(model_patcher)
        # Validate + coerce once, at construction time; store locally.
        s0, s1, f0, f1 = validate_schedule(start_scale, end_scale, start_fraction, end_fraction)
        self._start_scale = s0
        self._end_scale = s1
        self._start_fraction = f0
        self._end_fraction = f1
        # Per-run sigma schedule as plain CPU floats; populated only during sample().
        self._sigmas_cpu = None
        # Instance-local seam over the core CFG prediction function. Defaults to
        # ComfyUI's own sampling_function, so production behavior is unchanged;
        # contract tests swap this on a single instance without touching any
        # module-global state.
        self._sampling_function = comfy.samplers.sampling_function
        # Initialize the inherited cfg to the schedule start for introspection.
        self.set_cfg(s0)

    def scale_at_progress(self, progress) -> float:
        """Effective CFG scale at a given normalized progress in [0, 1]."""
        return linear_cfg_schedule(
            progress,
            self._start_scale,
            self._end_scale,
            self._start_fraction,
            self._end_fraction,
        )

    def _current_scale(self, timestep) -> float:
        if self._sigmas_cpu is None:
            # Outside a sampling run progress is unknown; fall back to the start.
            return self._start_scale
        if torch.is_tensor(timestep):
            sigma_value = float(timestep.detach().reshape(-1)[0].item())
        else:
            sigma_value = float(timestep)
        progress = progress_from_sigmas(sigma_value, self._sigmas_cpu)
        return self.scale_at_progress(progress)

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
               callback=None, disable_pbar=False, seed=None):
        # Capture the schedule as plain CPU floats, instance-local, so per-step
        # progress is deterministic and independent of the sigma tensor's device.
        self._sigmas_cpu = [float(s) for s in sigmas.detach().reshape(-1).tolist()]
        try:
            return super().sample(
                noise, latent_image, sampler, sigmas,
                denoise_mask=denoise_mask, callback=callback,
                disable_pbar=disable_pbar, seed=seed,
            )
        finally:
            self._sigmas_cpu = None

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        # Mirrors comfy.samplers.CFGGuider.predict_noise, substituting the
        # scheduled scale for the current step. Conditioning is passed through
        # untouched; the returned prediction keeps x's batch/dtype/device/shape
        # because the CFG combination is a scalar-weighted sum of predictions.
        scale = self._current_scale(timestep)
        self.cfg = scale
        return self._sampling_function(
            self.inner_model,
            x,
            timestep,
            self.conds.get("negative", None),
            self.conds.get("positive", None),
            scale,
            model_options=model_options,
            seed=seed,
        )


class ScheduledCFGGuiderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ScheduledCFGGuiderExt_LinearCFGGuider",
            display_name="Scheduled CFG Guider (Linear)",
            category="sampling/custom_sampling/guiders",
            description=(
                "Guider whose CFG scale moves linearly from start_scale to "
                "end_scale across normalized sampling progress. Clamped to "
                "start_scale before start_fraction and to end_scale after "
                "end_fraction. Descending schedules (start_scale > end_scale) "
                "are allowed."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Float.Input(
                    "start_scale", default=8.0, min=0.0, max=100.0, step=0.1,
                    tooltip="CFG scale at and before start_fraction.",
                ),
                io.Float.Input(
                    "end_scale", default=2.0, min=0.0, max=100.0, step=0.1,
                    tooltip="CFG scale at and after end_fraction. May be below start_scale.",
                ),
                io.Float.Input(
                    "start_fraction", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Normalized progress [0,1] where interpolation begins.",
                ),
                io.Float.Input(
                    "end_fraction", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Normalized progress [0,1] where interpolation ends. Must exceed start_fraction.",
                ),
            ],
            outputs=[io.Guider.Output()],
        )

    @classmethod
    def validate_inputs(cls, start_scale=None, end_scale=None, start_fraction=None,
                        end_fraction=None, **kwargs):
        """Graph-time validation: return True, or an error string naming the fault."""
        try:
            validate_schedule(start_scale, end_scale, start_fraction, end_fraction)
        except ScheduleError as exc:
            return str(exc)
        return True

    @classmethod
    def execute(cls, model, positive, negative, start_scale, end_scale,
                start_fraction, end_fraction) -> io.NodeOutput:
        guider = ScheduledCFGGuider(model, start_scale, end_scale, start_fraction, end_fraction)
        # set_conds -> convert_cond builds a fresh internal structure; the
        # caller's conditioning lists, dicts and tensors are left untouched.
        guider.set_conds(positive, negative)
        return io.NodeOutput(guider)


class ScheduledCFGGuiderExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ScheduledCFGGuiderNode]


async def comfy_entrypoint() -> ScheduledCFGGuiderExtension:
    return ScheduledCFGGuiderExtension()
