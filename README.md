# RTC changes to the async policy server

The **minimal changes** we made to LeRobot's async inference policy server to run
**Real-Time Chunking (RTC)** guidance server-side ("Option B": the server caches
the previously generated chunk and feeds its unexecuted tail back as
`prev_chunk_left_over`).

`modified_files/` holds path-mirrored copies of the changed files. These are the
live files under `lerobot/` — this folder is just a visible index of them.

## Files changed (4)

| File | What it does |
|------|--------------|
| [`async_inference/configs.py`](modified_files/src/lerobot/async_inference/configs.py) | New server knobs: `enable_rtc`, `rtc_execution_horizon`, `rtc_reset_gap_s` (+ validation) |
| [`async_inference/policy_server.py`](modified_files/src/lerobot/async_inference/policy_server.py) | The RTC logic — caches each chunk, feeds the unexecuted tail back next cycle, re-anchors for relative-action policies |
| [`policies/rtc/configuration_rtc.py`](modified_files/src/lerobot/policies/rtc/configuration_rtc.py) | Default `prefix_attention_schedule`: `LINEAR` → `EXP` |
| [`tests/async_inference/test_policy_server.py`](modified_files/tests/async_inference/test_policy_server.py) | Unit tests (mock RTC policy, no real inference) |

## How to run

Run from the `lerobot/` directory.

```bash
# Force-enable RTC on a pi05 checkpoint that shipped rtc_config=null
python -m lerobot.async_inference.policy_server \
    --enable_rtc \
    --rtc_execution_horizon 10 \
    --rtc_reset_gap_s 0.5
```

Flags (all optional, from `configs.py`):
- `--enable_rtc` — force RTC on even if the checkpoint's `rtc_config` is null/disabled. No effect on non-RTC policies (e.g. ACT).
- `--rtc_execution_horizon N` — override the execution horizon (default: checkpoint value, else `RTCConfig` default).
- `--rtc_reset_gap_s S` — flush the cached chunk when an observation arrives >S s late, e.g. after homing / pause-resume (default `0.5`; `0` disables).

```bash
# Tests
python -m pytest tests/async_inference/test_policy_server.py
```

## Client note

Option B assumes the **client executes the chunk verbatim**. With RTC on, set the
client to `aggregate_fn_name="latest_only"` (the default `"weighted_average"`
makes the executed trajectory drift from the server's cache and degrades
guidance). This must be set client-side (`orin_rollout`), not here.
