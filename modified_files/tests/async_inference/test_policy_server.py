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
"""Unit-tests for the `PolicyServer` core logic.
Monkey-patch the `policy` attribute with a stub so that no real model inference is performed.
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import OBS_STATE
from tests.utils import skip_if_package_missing

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


class MockPolicy:
    """A minimal mock for an actual policy, returning zeros.
    Refer to tests/policies for tests of the individual policies supported."""

    class _Config:
        robot_type = "dummy_robot"

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            """Empty image features since this test doesn't use images."""
            return {}

    def predict_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk of 20 dummy actions."""
        batch_size = len(observation[OBS_STATE])
        return torch.zeros(batch_size, 20, 6)

    def __init__(self):
        self.config = self._Config()

    def to(self, *args, **kwargs):
        # The server calls `policy.to(device)`. This stub ignores it.
        return self

    def model(self, batch: dict) -> torch.Tensor:
        # Return a chunk of 20 dummy actions.
        batch_size = len(batch["robot_type"])
        return torch.zeros(batch_size, 20, 6)


@pytest.fixture
@skip_if_package_missing("grpcio", "grpc")
def policy_server():
    """Fresh `PolicyServer` instance with a stubbed-out policy model."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer

    test_config = PolicyServerConfig(host="localhost", port=9999)
    server = PolicyServer(test_config)
    # Replace the real policy with our fast, deterministic stub.
    server.policy = MockPolicy()
    server.actions_per_chunk = 20
    server.device = "cpu"

    # Add mock lerobot_features that the observation similarity functions need
    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        }
    }

    return server


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_obs(state: torch.Tensor, timestep: int = 0, must_go: bool = False):
    """Create a TimedObservation with a given state vector."""
    # Import only when needed
    from lerobot.async_inference.helpers import TimedObservation

    return TimedObservation(
        observation={
            "joint1": state[0].item() if len(state) > 0 else 0.0,
            "joint2": state[1].item() if len(state) > 1 else 0.0,
            "joint3": state[2].item() if len(state) > 2 else 0.0,
            "joint4": state[3].item() if len(state) > 3 else 0.0,
            "joint5": state[4].item() if len(state) > 4 else 0.0,
            "joint6": state[5].item() if len(state) > 5 else 0.0,
        },
        timestamp=time.time(),
        timestep=timestep,
        must_go=must_go,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_time_action_chunk(policy_server):
    """Verify that `_time_action_chunk` assigns correct timestamps and timesteps."""
    start_ts = time.time()
    start_t = 10
    # A chunk of 3 action tensors.
    action_tensors = [torch.randn(6) for _ in range(3)]

    timed_actions = policy_server._time_action_chunk(start_ts, action_tensors, start_t)

    assert len(timed_actions) == 3
    # Check timesteps
    assert [ta.get_timestep() for ta in timed_actions] == [10, 11, 12]
    # Check timestamps
    expected_timestamps = [
        start_ts,
        start_ts + policy_server.config.environment_dt,
        start_ts + 2 * policy_server.config.environment_dt,
    ]
    for ta, expected_ts in zip(timed_actions, expected_timestamps, strict=True):
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_maybe_enqueue_observation_must_go(policy_server):
    """An observation with `must_go=True` is always enqueued."""
    obs = _make_obs(torch.zeros(6), must_go=True)
    assert policy_server._enqueue_observation(obs) is True
    assert policy_server.observation_queue.qsize() == 1
    assert policy_server.observation_queue.get_nowait() is obs


def test_maybe_enqueue_observation_dissimilar(policy_server):
    """A dissimilar observation (not `must_go`) is enqueued."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, dissimilar observation.
    new_obs = _make_obs(torch.ones(6) * 5)  # High norm difference

    assert policy_server._enqueue_observation(new_obs) is True
    assert policy_server.observation_queue.qsize() == 1


def test_maybe_enqueue_observation_is_skipped(policy_server):
    """A similar observation (not `must_go`) is skipped."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, very similar observation.
    new_obs = _make_obs(torch.zeros(6) + 1e-4)

    assert policy_server._enqueue_observation(new_obs) is False
    assert policy_server.observation_queue.empty() is True


def test_obs_sanity_checks(policy_server):
    """Unit-test the private `_obs_sanity_checks` helper."""
    prev = _make_obs(torch.zeros(6), timestep=0)

    # Case 1 – timestep already predicted
    policy_server._predicted_timesteps.add(1)
    obs_same_ts = _make_obs(torch.ones(6), timestep=1)
    assert policy_server._obs_sanity_checks(obs_same_ts, prev) is False

    # Case 2 – observation too similar
    policy_server._predicted_timesteps.clear()
    obs_similar = _make_obs(torch.zeros(6) + 1e-4, timestep=2)
    assert policy_server._obs_sanity_checks(obs_similar, prev) is False

    # Case 3 – genuinely new & dissimilar observation passes
    obs_ok = _make_obs(torch.ones(6) * 5, timestep=3)
    assert policy_server._obs_sanity_checks(obs_ok, prev) is True


def test_predict_action_chunk(monkeypatch, policy_server):
    """End-to-end test of `_predict_action_chunk` with a stubbed _get_action_chunk."""
    # Import only when needed
    from lerobot.async_inference.policy_server import PolicyServer

    # Force server to act-style policy; patch method to return deterministic tensor
    policy_server.policy_type = "act"
    # NOTE(Steven): Smelly tests as the Server is a state machine being partially mocked. Adding these processors as a quick fix.
    policy_server.preprocessor = lambda obs: obs
    policy_server.postprocessor = lambda tensor: tensor
    action_dim = 6
    batch_size = 1
    actions_per_chunk = policy_server.actions_per_chunk

    def _fake_get_action_chunk(_self, _obs, _type="act"):
        return torch.zeros(batch_size, actions_per_chunk, action_dim)

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    obs = _make_obs(torch.zeros(6), timestep=5)
    timed_actions = policy_server._predict_action_chunk(obs)

    assert len(timed_actions) == actions_per_chunk
    assert [ta.get_timestep() for ta in timed_actions] == list(range(5, 5 + actions_per_chunk))

    for i, ta in enumerate(timed_actions):
        expected_ts = obs.get_timestamp() + i * policy_server.config.environment_dt
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6

    # A non-RTC (ACT) policy must never be gated into RTC nor have a chunk cached.
    assert policy_server._policy_rtc_enabled() is False
    assert policy_server._last_chunk is None
    assert policy_server._last_chunk_start is None


# -----------------------------------------------------------------------------
# RTC (Option B: server-cached chunk) tests
# -----------------------------------------------------------------------------


class MockRTCPolicy:
    """Mock flow-matching policy with RTC enabled.

    ``predict_action_chunk`` records the kwargs it receives (so tests can assert on
    ``prev_chunk_left_over`` / ``inference_delay`` / ``execution_horizon``) and returns a
    deterministic chunk where action value at timestep ``t`` equals ``t`` (broadcast across the
    action dim), making time-axis slices trivial to verify.
    """

    class _RTCConfig:
        enabled = True
        execution_horizon = 10

    class _Config:
        robot_type = "dummy_robot"

        def __init__(self):
            self.rtc_config = MockRTCPolicy._RTCConfig()

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            return {}

    def __init__(self, chunk_size: int = 20, action_dim: int = 6):
        self.config = self._Config()
        # A non-None sentinel: the server checks the processor was initialized, not its type.
        self.rtc_processor = object()
        self._chunk_size = chunk_size
        self._action_dim = action_dim
        self.received_kwargs: list[dict] = []

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def predict_action_chunk(self, observation: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        self.received_kwargs.append(kwargs)
        batch_size = len(observation[OBS_STATE])
        base = torch.arange(self._chunk_size, dtype=torch.float32).view(1, self._chunk_size, 1)
        return base.expand(batch_size, self._chunk_size, self._action_dim).clone()

    def to(self, *args, **kwargs):
        return self


@pytest.fixture
@skip_if_package_missing("grpcio", "grpc")
def rtc_server():
    """Fresh `PolicyServer` with a stubbed RTC-enabled policy and identity (de)processors."""
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer

    server = PolicyServer(PolicyServerConfig(host="localhost", port=9999))
    server.policy = MockRTCPolicy()
    server.policy_type = "pi05"
    server.actions_per_chunk = 20
    server.device = "cpu"
    server.preprocessor = lambda obs: obs
    server.postprocessor = lambda tensor: tensor
    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        }
    }
    return server


def _model_space_chunk(chunk_size: int = 20, action_dim: int = 6) -> torch.Tensor:
    """(1, T, A) chunk whose value at timestep t equals t across the action dim."""
    base = torch.arange(chunk_size, dtype=torch.float32).view(1, chunk_size, 1)
    return base.expand(1, chunk_size, action_dim).clone()


def test_rtc_gating(rtc_server, policy_server):
    """`_policy_rtc_enabled` is True only for an RTC-enabled policy with a processor."""
    assert rtc_server._policy_rtc_enabled() is True

    # Disabled in config -> not gated.
    rtc_server.policy.config.rtc_config.enabled = False
    assert rtc_server._policy_rtc_enabled() is False
    rtc_server.policy.config.rtc_config.enabled = True

    # Processor not initialized -> not gated.
    rtc_server.policy.rtc_processor = None
    assert rtc_server._policy_rtc_enabled() is False

    # ACT-style policy (no `_rtc_enabled`) -> not gated.
    assert policy_server._policy_rtc_enabled() is False


def test_rtc_first_chunk_passes_none(rtc_server):
    """With no cached chunk, RTC kwargs carry `prev_chunk_left_over=None` (RTC no-ops)."""
    obs = _make_obs(torch.zeros(6), timestep=0)
    kwargs = rtc_server._compute_rtc_kwargs(obs)

    assert kwargs["prev_chunk_left_over"] is None
    assert kwargs["execution_horizon"] == 10
    assert isinstance(kwargs["inference_delay"], int) and kwargs["inference_delay"] >= 0


def test_rtc_cache_is_model_space_and_shape(rtc_server):
    """`_cache_action_chunk` stores the (T, A) pre-postprocessor tensor, keyed by i_0."""
    chunk = _model_space_chunk()  # (1, 20, 6)
    rtc_server._cache_action_chunk(chunk, i_0=7, ts=123.0)

    assert rtc_server._last_chunk.shape == (20, 6)
    assert rtc_server._last_chunk_start == 7
    assert rtc_server._last_chunk_ts == 123.0
    # Values are the raw model-space arange (not unnormalized/robot space).
    assert torch.equal(rtc_server._last_chunk[:, 0], torch.arange(20, dtype=torch.float32))
    # Cache is decoupled from the source tensor (detach + clone).
    chunk[0, 0, 0] = -999.0
    assert rtc_server._last_chunk[0, 0] == 0.0


def test_rtc_second_chunk_slices_and_aligns(rtc_server):
    """The leftover is sliced at `offset = i_0 - last_start` and aligned so index 0 == i_0."""
    rtc_server._cache_action_chunk(_model_space_chunk(), i_0=0, ts=time.time())

    obs = _make_obs(torch.zeros(6), timestep=3)
    kwargs = rtc_server._compute_rtc_kwargs(obs)
    prev = kwargs["prev_chunk_left_over"]

    # 20 cached steps, offset 3 -> 17 remaining, in (T, A) model space.
    assert prev.shape == (17, 6)
    # Index 0 of the leftover corresponds to absolute timestep 3.
    assert torch.equal(prev[:, 0], torch.arange(3, 20, dtype=torch.float32))


def test_rtc_drained_chunk_passes_none(rtc_server):
    """If the observation timestep is past the cached chunk, no guidance is applied."""
    rtc_server._cache_action_chunk(_model_space_chunk(), i_0=0, ts=time.time())  # covers steps [0, 20)

    obs = _make_obs(torch.zeros(6), timestep=25)  # offset 25 >= 20
    kwargs = rtc_server._compute_rtc_kwargs(obs)
    assert kwargs["prev_chunk_left_over"] is None


def test_rtc_stale_obs_passes_none(rtc_server):
    """An out-of-order observation older than the cached chunk yields no guidance (offset < 0)."""
    rtc_server._cache_action_chunk(_model_space_chunk(), i_0=10, ts=time.time())

    obs = _make_obs(torch.zeros(6), timestep=5)  # offset -5
    kwargs = rtc_server._compute_rtc_kwargs(obs)
    assert kwargs["prev_chunk_left_over"] is None


def test_rtc_reset_clears_cache(rtc_server):
    """`_reset_server` drops the cached chunk so the next chunk passes `prev_chunk_left_over=None`."""
    rtc_server._cache_action_chunk(_model_space_chunk(), i_0=0, ts=time.time())
    assert rtc_server._last_chunk is not None

    rtc_server._reset_server()

    assert rtc_server._last_chunk is None
    assert rtc_server._last_chunk_start is None
    assert rtc_server._last_chunk_ts is None
    kwargs = rtc_server._compute_rtc_kwargs(_make_obs(torch.zeros(6), timestep=2))
    assert kwargs["prev_chunk_left_over"] is None


def test_rtc_gap_flushes_cache(rtc_server, monkeypatch):
    """An observation arriving well after the cached chunk (homing / pause-resume) flushes the
    cache, so the next chunk starts RTC from scratch."""
    monkeypatch.setattr(time, "time", lambda: 1003.0)
    rtc_server._cache_action_chunk(_model_space_chunk(), i_0=0, ts=1000.0)  # 3s before "now"
    assert rtc_server._last_chunk is not None

    # Obs 3s after the cached chunk (default rtc_reset_gap_s=0.5) -> flush, no guidance.
    obs = _make_obs(torch.zeros(6), timestep=1)  # _make_obs stamps timestamp=time.time()=1003.0
    kwargs = rtc_server._compute_rtc_kwargs(obs)

    assert kwargs["prev_chunk_left_over"] is None
    assert rtc_server._last_chunk is None
    assert rtc_server._last_chunk_abs is None
    assert rtc_server._last_chunk_start is None
    assert rtc_server._last_chunk_ts is None


def test_rtc_small_gap_keeps_cache(rtc_server, monkeypatch):
    """A normal inter-observation gap (< rtc_reset_gap_s) does NOT flush the cache."""
    monkeypatch.setattr(time, "time", lambda: 1000.2)
    rtc_server._cache_action_chunk(_model_space_chunk(), i_0=0, ts=1000.0)  # 0.2s before "now"

    obs = _make_obs(torch.zeros(6), timestep=3)  # timestamp=1000.2 -> gap 0.2s < 0.5s
    kwargs = rtc_server._compute_rtc_kwargs(obs)

    assert rtc_server._last_chunk is not None
    assert torch.equal(kwargs["prev_chunk_left_over"][:, 0], torch.arange(3, 20, dtype=torch.float32))


def test_rtc_gap_flush_disabled(rtc_server, monkeypatch):
    """rtc_reset_gap_s=0 disables the flush — even a huge gap keeps the cache."""
    rtc_server.config.rtc_reset_gap_s = 0.0
    monkeypatch.setattr(time, "time", lambda: 2000.0)
    rtc_server._cache_action_chunk(_model_space_chunk(), i_0=0, ts=1000.0)  # 1000s before "now"

    obs = _make_obs(torch.zeros(6), timestep=3)  # gap 1000s, but check disabled
    kwargs = rtc_server._compute_rtc_kwargs(obs)

    assert rtc_server._last_chunk is not None
    assert kwargs["prev_chunk_left_over"] is not None


def test_rtc_inference_delay_from_timestamps(rtc_server, monkeypatch):
    """`inference_delay = ceil((now - obs_capture) / environment_dt)` (matches the sync `ceil`)."""
    dt = rtc_server.config.environment_dt
    obs = _make_obs(torch.zeros(6), timestep=0)
    obs.timestamp = 1000.0
    # Freeze the server clock ~2.4 control steps after capture -> ceil == 3.
    # (Use a non-integer multiple so the assertion is robust to float rounding of the boundary.)
    monkeypatch.setattr(time, "time", lambda: 1000.0 + 2.4 * dt)

    kwargs = rtc_server._compute_rtc_kwargs(obs)
    assert kwargs["inference_delay"] == 3

    # Negative elapsed (clock skew) clamps to 0, never negative.
    monkeypatch.setattr(time, "time", lambda: 1000.0 - 5 * dt)
    kwargs = rtc_server._compute_rtc_kwargs(obs)
    assert kwargs["inference_delay"] == 0


def test_rtc_inference_delay_pinned_override(rtc_server, monkeypatch):
    """`--rtc_inference_delay` pins inference_delay to a fixed step count, ignoring timestamps."""
    dt = rtc_server.config.environment_dt
    obs = _make_obs(torch.zeros(6), timestep=0)
    obs.timestamp = 1000.0
    # A timestamp diff that would derive 3 steps; the pin must override it.
    monkeypatch.setattr(time, "time", lambda: 1000.0 + 2.4 * dt)

    rtc_server.config.rtc_inference_delay = 6
    assert rtc_server._compute_rtc_kwargs(obs)["inference_delay"] == 6

    # A pin of 0 is valid (no frozen prefix) and still overrides the derived value.
    rtc_server.config.rtc_inference_delay = 0
    assert rtc_server._compute_rtc_kwargs(obs)["inference_delay"] == 0

    # Back to None -> derived from timestamps again (2.4 steps -> ceil 3).
    rtc_server.config.rtc_inference_delay = None
    assert rtc_server._compute_rtc_kwargs(obs)["inference_delay"] == 3


def test_rtc_disabled_policy_returns_empty_kwargs(policy_server):
    """A non-RTC policy gets an empty kwargs dict, so `predict_action_chunk` is called as before."""
    obs = _make_obs(torch.zeros(6), timestep=0)
    assert policy_server._compute_rtc_kwargs(obs) == {}


def test_rtc_end_to_end_two_observations(rtc_server, monkeypatch):
    """Drive `_predict_action_chunk` twice: first passes None, second passes the aligned leftover,
    and the cache holds model-space (pre-postprocessor) actions throughout."""
    from lerobot.async_inference import policy_server as ps_module

    # Bypass the raw->lerobot conversion; identity preprocessor is already set on the fixture.
    monkeypatch.setattr(
        ps_module,
        "raw_observation_to_observation",
        lambda raw, feats, img_feats: {OBS_STATE: torch.zeros(1, 6)},
    )
    # Postprocessor scales by 10 so robot space is clearly distinguishable from model space.
    rtc_server.postprocessor = lambda tensor: tensor * 10.0

    policy = rtc_server.policy

    # --- First observation at timestep 0: no cache yet -> None guidance. ---
    first = rtc_server._predict_action_chunk(_make_obs(torch.zeros(6), timestep=0))
    assert policy.received_kwargs[0]["prev_chunk_left_over"] is None
    assert policy.received_kwargs[0]["execution_horizon"] == 10
    # Client-facing actions are robot space (x10); cache stays model space (arange).
    assert math.isclose(first[1].get_action()[0].item(), 10.0, rel_tol=1e-6)
    assert torch.equal(rtc_server._last_chunk[:, 0], torch.arange(20, dtype=torch.float32))
    assert rtc_server._last_chunk_start == 0

    # --- Second observation at timestep 4: leftover sliced at offset 4, aligned to i_0=4. ---
    rtc_server._predict_action_chunk(_make_obs(torch.zeros(6), timestep=4))
    prev = policy.received_kwargs[1]["prev_chunk_left_over"]
    assert prev.shape == (16, 6)
    assert torch.equal(prev[:, 0], torch.arange(4, 20, dtype=torch.float32))
    # Cache refreshed to the new chunk, re-keyed to the new start timestep.
    assert rtc_server._last_chunk_start == 4


# -----------------------------------------------------------------------------
# RTC relative-action re-anchoring tests
# -----------------------------------------------------------------------------


class MockRelativeStep:
    """Stub of RelativeActionsProcessorStep: holds a cached state and reports enabled."""

    enabled = True
    action_names = None

    def __init__(self, state: torch.Tensor | None = None):
        self._state = state

    def get_cached_state(self) -> torch.Tensor | None:
        return self._state


def _abs_chunk(chunk_size: int = 20, action_dim: int = 6) -> torch.Tensor:
    """(T, A) absolute chunk whose value at timestep t equals t across the action dim."""
    return torch.arange(chunk_size, dtype=torch.float32).view(chunk_size, 1).expand(chunk_size, action_dim).clone()


def test_rtc_cache_action_chunk_abs(rtc_server):
    """`_cache_action_chunk_abs` stores a decoupled copy of the postprocessed (absolute) chunk."""
    abs_chunk = _abs_chunk()
    rtc_server._cache_action_chunk_abs(abs_chunk)

    assert torch.equal(rtc_server._last_chunk_abs, abs_chunk)
    # Decoupled from the source (detach + clone).
    abs_chunk[0, 0] = -999.0
    assert rtc_server._last_chunk_abs[0, 0] == 0.0


def test_rtc_relative_reanchors_absolute_leftover(rtc_server, monkeypatch):
    """For a relative-action policy, the leftover is the ABSOLUTE tail re-anchored to the current
    state — not the raw model-space cache."""
    from lerobot.async_inference import policy_server as ps_module

    seen: dict = {}
    sentinel = torch.full((17, 6), 7.0)

    def spy(**kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(ps_module, "reanchor_relative_rtc_prefix", spy)

    state = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    rtc_server._relative_step = MockRelativeStep(state)
    rtc_server._normalizer_step = None
    rtc_server._last_chunk_abs = _abs_chunk()  # covers steps [0, 20)
    rtc_server._last_chunk_start = 0

    kwargs = rtc_server._compute_rtc_kwargs(_make_obs(torch.zeros(6), timestep=3))

    # The re-anchored tensor is what gets passed to the policy.
    assert kwargs["prev_chunk_left_over"] is sentinel
    # Re-anchoring was fed the absolute leftover sliced at offset 3, aligned to timestep 3...
    assert torch.equal(seen["prev_actions_absolute"][:, 0], torch.arange(3, 20, dtype=torch.float32))
    # ...and the current (post-preprocessor) state.
    assert seen["current_state"] is state
    assert seen["relative_step"] is rtc_server._relative_step


def test_rtc_relative_first_chunk_passes_none(rtc_server, monkeypatch):
    """No absolute cache yet -> None, and re-anchoring is never invoked."""
    from lerobot.async_inference import policy_server as ps_module

    called = {"n": 0}
    monkeypatch.setattr(
        ps_module, "reanchor_relative_rtc_prefix", lambda **kw: called.__setitem__("n", called["n"] + 1)
    )
    rtc_server._relative_step = MockRelativeStep(torch.zeros(1, 6))

    kwargs = rtc_server._compute_rtc_kwargs(_make_obs(torch.zeros(6), timestep=0))

    assert kwargs["prev_chunk_left_over"] is None
    assert called["n"] == 0


def test_rtc_relative_drained_passes_none(rtc_server, monkeypatch):
    """Drained absolute chunk (offset >= len) -> None, no re-anchoring."""
    from lerobot.async_inference import policy_server as ps_module

    called = {"n": 0}
    monkeypatch.setattr(
        ps_module, "reanchor_relative_rtc_prefix", lambda **kw: called.__setitem__("n", called["n"] + 1)
    )
    rtc_server._relative_step = MockRelativeStep(torch.zeros(1, 6))
    rtc_server._last_chunk_abs = _abs_chunk()  # steps [0, 20)
    rtc_server._last_chunk_start = 0

    kwargs = rtc_server._compute_rtc_kwargs(_make_obs(torch.zeros(6), timestep=25))

    assert kwargs["prev_chunk_left_over"] is None
    assert called["n"] == 0


def test_rtc_relative_reset_clears_absolute_cache(rtc_server):
    """`_reset_server` also drops the absolute cache used for re-anchoring."""
    rtc_server._cache_action_chunk_abs(_abs_chunk())
    rtc_server._last_chunk_start = 0
    assert rtc_server._last_chunk_abs is not None

    rtc_server._reset_server()
    assert rtc_server._last_chunk_abs is None
    assert rtc_server._last_chunk_start is None


def test_rtc_relative_end_to_end(rtc_server, monkeypatch):
    """Drive `_predict_action_chunk` twice with a relative step: the absolute cache is the
    postprocessed chunk, and the second call re-anchors its aligned tail."""
    from lerobot.async_inference import policy_server as ps_module

    monkeypatch.setattr(
        ps_module,
        "raw_observation_to_observation",
        lambda raw, feats, img_feats: {OBS_STATE: torch.zeros(1, 6)},
    )
    # Postprocessor scales by 10, so the absolute (robot-space) chunk is clearly != model space.
    rtc_server.postprocessor = lambda tensor: tensor * 10.0

    state = torch.ones(1, 6)
    rtc_server._relative_step = MockRelativeStep(state)
    rtc_server._normalizer_step = None

    seen: dict = {}
    sentinel = torch.full((16, 6), 5.0)

    def spy(**kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(ps_module, "reanchor_relative_rtc_prefix", spy)

    # --- First observation: no absolute cache yet -> None guidance. ---
    rtc_server._predict_action_chunk(_make_obs(torch.zeros(6), timestep=0))
    assert rtc_server.policy.received_kwargs[0]["prev_chunk_left_over"] is None
    # Absolute cache holds the postprocessed chunk (model arange x10); model cache stays arange.
    assert torch.equal(rtc_server._last_chunk_abs[:, 0], torch.arange(20, dtype=torch.float32) * 10.0)
    assert torch.equal(rtc_server._last_chunk[:, 0], torch.arange(20, dtype=torch.float32))

    # --- Second observation at timestep 4: re-anchor the absolute tail sliced at offset 4. ---
    rtc_server._predict_action_chunk(_make_obs(torch.zeros(6), timestep=4))
    assert rtc_server.policy.received_kwargs[1]["prev_chunk_left_over"] is sentinel
    assert torch.equal(seen["prev_actions_absolute"][:, 0], torch.arange(4, 20, dtype=torch.float32) * 10.0)
    assert seen["current_state"] is state


# -----------------------------------------------------------------------------
# --enable_rtc server-side override tests
# -----------------------------------------------------------------------------


class MockOverridablePolicy:
    """Flow-matching-style policy that starts with RTC OFF (rtc_config=None), like a checkpoint
    that shipped with `rtc_config: null`. Exposes a real `init_rtc_processor` so `--enable_rtc`
    can switch it on."""

    class _Config:
        robot_type = "dummy_robot"

        def __init__(self):
            self.rtc_config = None

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            return {}

    def __init__(self):
        self.config = self._Config()
        self.rtc_processor = None

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def init_rtc_processor(self):
        from lerobot.policies.rtc import RTCProcessor

        self.rtc_processor = RTCProcessor(self.config.rtc_config) if self.config.rtc_config is not None else None

    def to(self, *args, **kwargs):
        return self


def test_enable_rtc_flag_creates_rtc_config(rtc_server):
    """`--enable_rtc` turns RTC on for a checkpoint that shipped with rtc_config=None."""
    rtc_server.policy = MockOverridablePolicy()
    assert rtc_server._policy_rtc_enabled() is False  # off by default

    rtc_server.config.enable_rtc = True
    rtc_server._maybe_override_rtc()

    assert rtc_server.policy.config.rtc_config is not None
    assert rtc_server.policy.config.rtc_config.enabled is True
    assert rtc_server.policy.rtc_processor is not None
    assert rtc_server._policy_rtc_enabled() is True


def test_enable_rtc_execution_horizon_override(rtc_server):
    """`--rtc_execution_horizon` overrides the horizon when force-enabling."""
    rtc_server.policy = MockOverridablePolicy()
    rtc_server.config.enable_rtc = True
    rtc_server.config.rtc_execution_horizon = 25
    rtc_server._maybe_override_rtc()

    assert rtc_server.policy.config.rtc_config.execution_horizon == 25


def test_enable_rtc_noop_without_flag(rtc_server):
    """Without the flag, a null rtc_config stays null (RTC off)."""
    rtc_server.policy = MockOverridablePolicy()
    rtc_server._maybe_override_rtc()  # enable_rtc defaults False

    assert rtc_server.policy.config.rtc_config is None
    assert rtc_server._policy_rtc_enabled() is False


def test_enable_rtc_ignored_for_unsupported_policy(policy_server):
    """`--enable_rtc` on an ACT-style policy (no rtc_config / init_rtc_processor) is a safe no-op."""
    policy_server.config.enable_rtc = True
    policy_server._maybe_override_rtc()  # must not raise

    assert policy_server._policy_rtc_enabled() is False
