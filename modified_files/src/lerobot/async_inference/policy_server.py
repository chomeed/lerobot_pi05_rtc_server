# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import math
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig, reanchor_relative_rtc_prefix
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.types import PolicyAction

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # --- Real-Time Chunking (RTC), Option B: server-cached chunk ---
        # The last action chunk we generated, kept in NORMALIZED MODEL SPACE (i.e. before the
        # postprocessor unnormalizes it), shape (T, A). This mirrors what the synchronous rollout
        # caches in ActionQueue.original_queue and feeds back as `prev_chunk_left_over`. Used as the
        # leftover directly for ABSOLUTE-action policies.
        self._last_chunk: torch.Tensor | None = None
        # The same chunk in ABSOLUTE (post-postprocessor, robot) space, shape (T, A). For
        # RELATIVE-action policies the model-space leftover is anchored to the previous observation's
        # state, so it cannot be fed as-is; instead we keep the absolute (frame-independent) targets
        # and re-anchor them to the current state each cycle (see `_compute_rtc_kwargs`).
        self._last_chunk_abs: torch.Tensor | None = None
        # Absolute (global, monotonic) control timestep of index 0 of both cached chunks, i.e. the
        # `i_0` they were stamped with. Used to align the leftover slice with the next observation.
        self._last_chunk_start: int | None = None
        # Client-clock timestamp of the observation that produced the cached chunk. Used to flush the
        # cache after a long observation gap (e.g. homing / pause-resume), where the cached chunk
        # describes the pre-gap trajectory and must not guide the post-gap chunk.
        self._last_chunk_ts: float | None = None
        # Relative-action re-anchoring: populated from the preprocessor in SendPolicyInstructions.
        # `_relative_step` is non-None only when the checkpoint uses relative actions (enabled).
        self._relative_step: RelativeActionsProcessorStep | None = None
        self._normalizer_step: NormalizerProcessorStep | None = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

        # Drop the cached RTC chunk: the timestep index restarts for a new client, so any
        # leftover from the previous session would be misaligned. A None cache makes RTC no-op
        # on the first chunk after reset (prev_chunk_left_over=None).
        self._flush_rtc_cache()

    def _flush_rtc_cache(self) -> None:
        """Drop the cached chunk so the next chunk restarts RTC from scratch (prev_chunk_left_over
        becomes None). Called on server reset and after a stale observation gap (homing / pause)."""
        self._last_chunk = None
        self._last_chunk_abs = None
        self._last_chunk_start = None
        self._last_chunk_ts = None

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        # Optionally force-enable RTC from the server config, since checkpoints may ship with
        # rtc_config=null. Must run before relative-step introspection and status logging.
        self._maybe_override_rtc()

        # Introspect the preprocessor for relative-action handling, so RTC can re-anchor its
        # leftover prefix to the current state (only relevant for relative-action policies).
        self._init_relative_action_steps()

        # One explicit line so a silently-off RTC checkpoint is obvious at startup.
        self._log_rtc_status()

        return services_pb2.Empty()

    def _maybe_override_rtc(self) -> None:
        """Force-enable / retune RTC from the server config, independent of the checkpoint.

        Checkpoints often ship with ``rtc_config=null`` (RTC off). When ``--enable_rtc`` is set,
        create or enable the policy's ``rtc_config`` and (re)initialize its RTC processor so RTC
        runs without hand-editing the checkpoint. No-ops for policies that do not support RTC.
        """
        if not self.config.enable_rtc:
            return

        if not hasattr(self.policy.config, "rtc_config") or not hasattr(self.policy, "init_rtc_processor"):
            self.logger.warning(
                f"--enable_rtc was set but policy type '{self.policy_type}' does not support RTC; ignoring."
            )
            return

        rtc_config = self.policy.config.rtc_config
        if rtc_config is None:
            rtc_config = RTCConfig(enabled=True)
        else:
            rtc_config.enabled = True
        if self.config.rtc_execution_horizon is not None:
            rtc_config.execution_horizon = self.config.rtc_execution_horizon

        self.policy.config.rtc_config = rtc_config
        self.policy.init_rtc_processor()  # (re)creates the RTC processor from the updated config
        self.logger.info(
            f"RTC force-enabled via --enable_rtc | execution_horizon={rtc_config.execution_horizon} | "
            f"schedule={rtc_config.prefix_attention_schedule}"
        )

    def _log_rtc_status(self) -> None:
        """Log whether RTC guidance is active for the loaded policy (so a null/disabled
        rtc_config, which silently produces zero guidance, is visible at startup)."""
        if self._policy_rtc_enabled():
            relative = "on" if self._relative_step is not None else "off"
            self.logger.info(
                f"RTC ACTIVE | execution_horizon={self.policy.config.rtc_config.execution_horizon} | "
                f"relative re-anchoring: {relative}"
            )
        else:
            self.logger.info(
                f"RTC INACTIVE for policy '{self.policy_type}' (checkpoint rtc_config is null/disabled, "
                f"or the policy does not support RTC). Pass --enable_rtc to force-enable on "
                f"flow-matching policies."
            )

    def _init_relative_action_steps(self) -> None:
        """Cache references to the preprocessor's relative-action and normalizer steps.

        Mirrors ``RTCInferenceEngine.__init__``: RTC on a relative-action policy must re-express
        the (absolute) leftover prefix relative to the robot's current state before feeding it back
        as guidance. Sets ``self._relative_step`` only when the checkpoint actually uses relative
        actions (``RelativeActionsProcessorStep`` present and enabled); otherwise it stays ``None``
        and RTC uses the model-space leftover directly.
        """
        self._relative_step = None
        self._normalizer_step = None
        if self.preprocessor is None:
            return

        self._relative_step = next(
            (s for s in self.preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        self._normalizer_step = next(
            (s for s in self.preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )

        if self._relative_step is None:
            return

        # The re-anchor mask needs action names to know which dims are relative. The checkpoint's
        # saved processor config usually carries them; fall back to the policy config (no robot is
        # available server-side, unlike the sync rollout's robot_wrapper fallback).
        if self._relative_step.action_names is None:
            cfg_names = getattr(self.policy.config, "action_feature_names", None)
            if cfg_names:
                self._relative_step.action_names = list(cfg_names)
            else:
                self.logger.warning(
                    "RTC relative re-anchoring: action_names are unresolved and no exclude_joints "
                    "mask is set; all action dims will be treated as relative."
                )
        self.logger.info(
            "Relative-action policy detected: RTC will re-anchor its leftover prefix to the "
            "current state before guidance."
        )

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor], **rtc_kwargs: Any) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only ``actions_per_chunk`` actions.

        Any ``rtc_kwargs`` (``prev_chunk_left_over``, ``inference_delay``, ``execution_horizon``) are
        forwarded verbatim to RTC-capable flow-matching policies. For non-RTC policies the caller
        passes no kwargs, so ``predict_action_chunk`` is invoked exactly as before.
        """
        chunk = self.policy.predict_action_chunk(observation, **rtc_kwargs)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _policy_rtc_enabled(self) -> bool:
        """Whether RTC guidance should be applied for the currently loaded policy.

        True only for flow-matching policies (pi0 / pi05 / smolvla) that expose ``_rtc_enabled``,
        have it enabled in their ``rtc_config``, and have actually initialized an ``rtc_processor``.
        ACT and other policies do not accept the RTC kwargs, so they must never be gated in here.
        """
        if self.policy is None:
            return False
        rtc_enabled = getattr(self.policy, "_rtc_enabled", None)
        if not callable(rtc_enabled) or not rtc_enabled():
            return False
        # The processor is created in the policy's `init_rtc_processor`; guard against it missing.
        return getattr(self.policy, "rtc_processor", None) is not None

    def _compute_rtc_kwargs(self, observation_t: TimedObservation) -> dict[str, Any]:
        """Build the RTC ``ActionSelectKwargs`` for the current observation (Option B).

        Returns an empty dict for non-RTC policies so ``_get_action_chunk`` behaves exactly as
        before. For RTC policies it returns ``prev_chunk_left_over`` (the unexecuted tail of the
        cached previous chunk, aligned so index 0 == the current timestep), ``inference_delay``
        (steps elapsed from observation capture to now), and ``execution_horizon`` (from the
        policy's ``rtc_config``).

        MUST be called AFTER the preprocessor has run for this observation: the relative-action
        re-anchoring reads the current state that the preprocessor cached
        (``RelativeActionsProcessorStep.get_cached_state()``).
        """
        if not self._policy_rtc_enabled():
            return {}

        # Flush a stale cache after a long observation gap (homing / pause-resume): the cached chunk
        # describes the pre-gap trajectory, and (for relative actions) re-anchoring it to the new
        # state would tug the robot back toward where it was. The client sends no observations while
        # paused, so such a gap is exactly the home/pause signal.
        self._flush_rtc_cache_if_stale(observation_t.get_timestamp())

        i_0 = observation_t.get_timestep()
        prev_chunk_left_over = self._prev_chunk_left_over(i_0)

        # inference_delay: number of control steps that elapse between when the client captured the
        # observation and now (chunk generation). This is the async analog of the sync rollout's
        # `ceil(latency / dt)`; we derive it from timestamps because there is no in-process latency
        # tracker on the server. Clamp to >= 0 (clock skew) and round up to match the sync `ceil`.
        elapsed_s = max(0.0, time.time() - observation_t.get_timestamp())
        inference_delay = math.ceil(elapsed_s / self.config.environment_dt)

        return {
            "prev_chunk_left_over": prev_chunk_left_over,
            "inference_delay": inference_delay,
            "execution_horizon": self.policy.config.rtc_config.execution_horizon,
        }

    def _flush_rtc_cache_if_stale(self, obs_timestamp: float) -> None:
        """Flush the cache if this observation arrived long after the cached chunk was generated.

        A gap larger than ``config.rtc_reset_gap_s`` (client clock) means the robot was not
        continuously executing the cached chunk — the tell-tale of a home / pause-resume, during
        which the client sends no observations. Guiding the next chunk from that stale cache would be
        wrong (see the caller). ``rtc_reset_gap_s == 0`` disables the check.
        """
        gap_limit = self.config.rtc_reset_gap_s
        if gap_limit <= 0 or self._last_chunk_ts is None:
            return
        gap = obs_timestamp - self._last_chunk_ts
        if gap > gap_limit:
            self.logger.info(
                f"RTC cache flushed: observation arrived {gap:.2f}s after the last chunk "
                f"(> {gap_limit:.2f}s) — restarting RTC guidance from scratch."
            )
            self._flush_rtc_cache()

    def _prev_chunk_left_over(self, i_0: int) -> torch.Tensor | None:
        """Slice the unexecuted leftover of the cached previous chunk, in model space.

        Alignment: the cached chunk's index k has absolute timestep ``_last_chunk_start + k``, so
        the current step (i_0) lives at ``offset = i_0 - _last_chunk_start``. The returned tensor is
        aligned so index 0 == i_0, i.e. the same absolute timestep as index 0 of the chunk we are
        about to generate. Returns ``None`` (RTC no-ops) when there is no cache, the observation is
        stale (offset < 0), or the chunk is fully drained (offset >= len).

        For RELATIVE-action policies the model-space leftover is anchored to the *previous*
        observation's state and is not directly comparable to the new chunk (the robot moved in
        between). We instead re-anchor the ABSOLUTE leftover to the current state and re-normalize,
        mirroring ``RTCInferenceEngine`` (rollout/inference/rtc.py). For absolute-action policies
        the model-space leftover is returned directly.
        """
        if self._last_chunk_start is None:
            return None
        offset = i_0 - self._last_chunk_start

        if self._relative_step is not None:
            # Relative-action policy: re-anchor the absolute (frame-independent) leftover.
            if self._last_chunk_abs is None or not (0 <= offset < self._last_chunk_abs.shape[0]):
                return None
            abs_left_over = self._last_chunk_abs[offset:]
            current_state = self._relative_step.get_cached_state()
            if current_state is None or abs_left_over.numel() == 0:
                return None
            # Re-express the absolute targets as deltas from the current state, then normalize ->
            # model space, so the leftover lives in the SAME frame as the chunk being generated.
            return reanchor_relative_rtc_prefix(
                prev_actions_absolute=abs_left_over,
                current_state=current_state,
                relative_step=self._relative_step,
                normalizer_step=self._normalizer_step,
                policy_device=self.device,
            )

        # Absolute-action policy: model-space leftover is directly comparable, shape (T_left, A).
        # The RTC processor adds the batch dim, right-pads the time axis to the chunk length, and
        # pads/clamps action-dim itself. (The sync rollout additionally forces this to
        # execution_horizon length via _normalize_prev_actions_length for torch.compile stability;
        # unnecessary here since this path is not compiled and the processor handles variable lengths.)
        if self._last_chunk is None or not (0 <= offset < self._last_chunk.shape[0]):
            return None
        return self._last_chunk[offset:]

    def _cache_action_chunk(self, action_tensor: torch.Tensor, i_0: int, ts: float) -> None:
        """Cache the freshly generated chunk (model space) for RTC guidance on the next observation.

        Caches only for RTC-enabled policies. ``action_tensor`` is (B, T, A) in NORMALIZED MODEL
        SPACE (this must be called BEFORE the postprocessor unnormalizes it); we store index 0 of
        the batch as (T, A) to match the sync rollout's ``ActionQueue.original_queue`` contract.
        ``i_0`` / ``ts`` are the producing observation's timestep and client-clock timestamp (the
        latter used by ``_flush_rtc_cache_if_stale``). The absolute (robot-space) copy used for
        relative re-anchoring is cached separately by ``_cache_action_chunk_abs`` after postprocessing.

        NOTE: Option B assumes the client executes this chunk verbatim, so that the server's cache
        matches what the robot actually runs. If the client aggregates chunks (its default
        ``aggregate_fn_name="weighted_average"``), the executed trajectory drifts from this cache
        and RTC guidance degrades. When RTC is on, the client should use "take newest"
        (``aggregate_fn_name="latest_only"``). This is intentionally NOT changed here — see the
        accompanying note; it must be set client-side. (Relative re-anchoring reads the true current
        state each cycle, so it is more forgiving of the aggregate-fn choice, but not immune.)
        """
        if not self._policy_rtc_enabled():
            return
        # Detach + clone so the subsequent in-place-free postprocessor loop (which rebuilds
        # action_tensor) cannot alias or mutate what we cache.
        self._last_chunk = action_tensor[0].detach().clone()
        self._last_chunk_start = i_0
        self._last_chunk_ts = ts

    def _cache_action_chunk_abs(self, action_tensor: torch.Tensor) -> None:
        """Cache the postprocessed chunk in ABSOLUTE (robot) space for relative RTC re-anchoring.

        Caches only for RTC-enabled policies. ``action_tensor`` is (T, A) after the postprocessor
        (unnormalized, and for relative-action policies already converted back to absolute joint
        targets by the paired ``AbsoluteActionsProcessorStep``). Keyed by the same
        ``_last_chunk_start`` set in ``_cache_action_chunk``.
        """
        if not self._policy_rtc_enabled():
            return
        self._last_chunk_abs = action_tensor.detach().clone()

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        # RTC (Option B): derive guidance kwargs from the cached previous chunk BEFORE inference,
        # then cache the fresh chunk (still in model space) AFTER inference. Both are no-ops for
        # non-RTC policies (empty kwargs, no caching).
        rtc_kwargs = self._compute_rtc_kwargs(observation_t)

        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation, **rtc_kwargs)
        inference_time = time.perf_counter() - start_inference

        # Cache in normalized model space (pre-postprocessor), keyed by this observation's timestep
        # and timestamp (the timestamp lets a later gap flush the cache after homing/pause).
        self._cache_action_chunk(action_tensor, observation_t.get_timestep(), observation_t.get_timestamp())

        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        # Cache the absolute (robot-space) chunk for relative-action RTC re-anchoring on the next
        # observation. Must happen here, after the postprocessor, since the model-space cache (set
        # pre-postprocessor) is anchored to this observation's state and can't be reused directly.
        self._cache_action_chunk_abs(action_tensor)

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
