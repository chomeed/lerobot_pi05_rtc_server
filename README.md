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

The example commands below are the actual ones used for a `board_insertion`
pi05 RTC eval (server on the workstation, client on the robot).

### 1. Server — on the workstation (this repo)

The server loads the checkpoint and applies RTC guidance. It **must be launched
as a module** (`-m lerobot.async_inference.policy_server`), never by file path:
`policy_server.py` uses package-relative imports (`from .configs import ...`), so
`python .../policy_server.py` — or running the copy under `modified_files/` —
fails with `ImportError: attempted relative import with no known parent package`.

`lerobot` is editable-installed from this fork, so `-m` already works from any
directory. To pin the absolute path explicitly (independent of what's installed),
point `PYTHONPATH` at this fork's `src`:

```bash
PYTHONPATH=/home/rllab4/workspace/chomeed/hdr_robot/policy_learning/lerobot_pi05_rtc_server/lerobot/src \
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30 \
    --enable_rtc=true \
    --rtc_execution_horizon=20 \
    --inference_latency=0
```

RTC flags added by this change (all optional, from `configs.py`):
- `--enable_rtc` — force RTC on even if the checkpoint's `rtc_config` is null/disabled. No effect on non-RTC policies (e.g. ACT).
- `--rtc_execution_horizon N` — override the execution horizon (default: checkpoint value, else `RTCConfig` default).
- `--rtc_reset_gap_s S` — flush the cached chunk when an observation arrives >S s late, e.g. after homing / pause-resume (default `0.5`; `0` disables).
- `--rtc_inference_delay N` — pin the RTC `inference_delay` to a fixed number of **control steps** instead of deriving it from observation timestamps. Use when the robot/workstation clocks aren't NTP-synced, or when you have a stable measured latency (e.g. 200 ms at 30 fps → pin `6`). Default `None` = derive per observation. **The server logs both the pinned and derived values every chunk**, so you can watch how the timestamp estimate compares. Note the derived value is a lower bound (sampled before inference, so it excludes inference + the return trip) and can land one step high at exact boundaries due to float rounding (`0.2 / (1/30) = 6.0000…1 → ceil 7`).

(`--host` / `--port` / `--fps` / `--inference_latency` are pre-existing server flags. Use the
underscore — draccus rejects `--inference-latency`. `--inference_latency` is in **seconds**, not
steps: a per-response pacing target (`time.sleep` pads each reply up to this duration; default
`1/fps` ≈ 0.033s; `0` disables the padding). It is unrelated to the RTC `inference_delay`, which is
in control **steps** and is derived from the actual observation→inference elapsed time.)

### 2. Client — on the robot (`lerobot-inference`, from the separate `orin_rollout` repo)

```bash
lerobot-inference \
    --server-address 169.254.186.74:8080 \
    --policy-type pi05 \
    --pretrained-name-or-path /home/rllab4/workspace/chomeed/hdr_robot/policy_learning/outputs/sirius/board_insertion_pi05_sirius_round2 \
    --task board_insertion \
    --actions-per-chunk 30 \
    --chunk-size-threshold 0.4 \
    --mode insertion_15 \
    --home-pose \
    --rollout-dir /root/demo_data/legato_rtc\
    --dry-run \
    --aggregate-fn-name latest_only
```

- **`--aggregate-fn-name latest_only`** is the key RTC setting — see the Client note below.
- `--server-address` points at the workstation running the server; the server (not the client) loads `--pretrained-name-or-path`.
- `--mode` / `--home-pose` / `--rollout-dir` / `--dry-run` are `orin_rollout` client flags, not part of this repo's `RobotClientConfig`.

### Tests

```bash
python -m pytest tests/async_inference/test_policy_server.py
```

## Client note

Option B assumes the **client executes the chunk verbatim**. With RTC on, set the
client to `aggregate_fn_name="latest_only"` (the default `"weighted_average"`
makes the executed trajectory drift from the server's cache and degrades
guidance). This must be set client-side (`orin_rollout`), not here.
