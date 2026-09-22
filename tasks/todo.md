# Fix the expert controller and regenerate demonstrations

## Problem

`ExpertController.compute_action` maps **world-frame** desired acceleration directly to
roll/pitch commands, but tilt acts in the drone's **body (yaw-rotated)** frame — and the
controller's own yaw law turns the drone to face its velocity. Gate 0 sits straight ahead
(yaw≈0, world == body) so it always passes; from the first turn onward the commanded force
is mis-rotated by the heading angle.

Measured on `simple_oval`: 0/500 expert episodes complete a lap, mean tracking error 1.94 m
against a 0.55 m gate half-width. BC clones that, PPO inherits it, and the trained agent
passes 1.12/4 gates.

`metadata.json` reported `successful_episodes: 500` because the counter increments on
`gates_passed > 0`.

## Tasks

- [x] `planning/trajectory_generator.py`: also return reference accelerations (exact 2nd
      derivative from the existing CubicSpline) so the controller can feed them forward
- [x] `planning/expert_policy.py`: rewrite `compute_action`
      - rotate desired world acceleration into the yaw frame before inverting to roll/pitch
      - add reference-acceleration feedforward and linear-drag compensation
      - tilt-compensated thrust (a 30° tilt currently loses 1.31 m/s² of lift)
      - replace the `/5.0` magic scaling with an exact accel→tilt inversion
      - raise kp_pos 3.0 → 8.0 (a pure-P loop at kp=3 needs a 1.33 m standing error to
        hold the 4 m/s² of a turn)
      - take gravity / drag / max_angle from the sim instead of hardcoding
- [x] `planning/expert_policy.py`: generation loop
      - stop on `truncated` as well as `terminated` (a completed lap currently keeps
        stepping and `_get_next_gate` wraps modulo, polluting the dataset)
      - allow a modest time margin past the reference length
      - keep only episodes that complete the lap; make `successful_episodes` mean that
- [x] Verify: 50-episode probe should show full laps and tracking error < 0.55 m
- [x] Regenerate the 500-episode dataset
- [x] Confirm existing tests still pass

## Review

### Changes

**`planning/trajectory_generator.py`** — `generate_min_jerk_trajectory` /
`generate_racing_trajectory` now also return accelerations, taken as the exact 2nd
derivative of the CubicSpline that was already being built. Return signature went from a
3-tuple to a 4-tuple; `planning/expert_policy.py` was the only caller.

**`planning/expert_policy.py`**

- `ExpertController.compute_action` rewritten. The desired world-frame acceleration is now
  rotated into the drone's yaw frame before being inverted into roll/pitch, which was the
  actual bug. It also feeds forward the reference acceleration, cancels the model's linear
  drag, and compensates thrust for tilt. The `/5.0` magic scaling is replaced by
  `arctan2`-based inversion, and roll/pitch are normalised by the sim's own `max_angle`.
- Gains raised to `kp_pos=8.0, kd_pos=6.0` (were 3.0 / 1.5). A pure-P loop at kp=3 needs a
  1.33 m standing error to hold the 4 m/s^2 of a turn, against a 0.55 m gate half-width.
- Gravity, drag and max tilt are read from `env._sim` rather than hardcoded.
- Generation loop stops on `truncated` as well as `terminated`. Previously a completed lap
  kept stepping, and `_get_next_gate` wraps modulo the gate count, so observations after
  the last gate pointed back at gate 0.
- Episodes are kept only if they pass every gate. `successful_episodes` in metadata is
  renamed `completed_lap_episodes` and now means that; the old counter incremented on
  `gates_passed > 0`, which is why it read 500/500 while nothing finished a lap.
- Added `REFERENCE_TIME_MARGIN = 1.25` so a slightly lagging run can still finish; the
  reference is held at its final sample once it runs out.

### Results

Dataset `outputs/expert_data/expert_simple_oval_20260922_130309`, 500 episodes,
`simple_oval`, perturbation 0.3 m:

| | before | after |
|---|---|---|
| complete laps | 0 / 500 | **500 / 500** |
| gates passed per episode | 1 x295, 2 x176, 3 x29 | 4 x500 |
| mean reference-tracking error | 1.94 m | **0.29 m** (gate half-width 0.55 m) |
| mean altitude | 0.85 m | **1.03 m** (gates at z = 1.0) |
| samples | 114 640 | 112 584 |

Gain sweep (50 episodes each) isolating how much each part contributes: with the yaw
rotation and feedforward in place but the original gains (3.0 / 1.5) the controller already
reaches 14/50 laps at 0.69 m error, so the frame bug was the dominant factor and the gains
are a refinement. 8.0 / 6.0 was the best point tested at 50/50 and 0.29 m.

Roll and pitch still saturate for ~20% of steps. That is roughly constant across the gain
sweep, so it is a property of the track's turn radius at 2 m/s against the 30 degree tilt
limit, not of the gains.

`pytest tests/ -q`: 31 passed, 2 failed. Both failures (`test_gate_passage_bonus`,
`test_collision_penalty`) pre-exist this change and are identical on a clean checkout --
they assert gate_bonus 100 / collision -50, which commit 8fcb8c9 changed to 200 / -150.

### Not done here

BC and PPO have not been rerun. Also still outstanding from the analysis: the training eval
callback uses `initial_perturbation=0` so all its episodes are identical, physics
randomization is a silent no-op under Gymnasium 1.2, the curriculum's level 0 ends the
episode after one gate, promoting to an all-gates level keeps a stale 3-gate count, the
BC->PPO weight transfer is invalid (LayerNorm / obs-normalisation / tanh mismatches), and
the progress reward is relative rather than telescoping.

---

# Fix the BC -> PPO transfer and the training eval callback

## Tasks

- [x] `RacingMLP`: drop LayerNorm and the output tanh so it mirrors SB3's actor exactly
- [x] `_load_bc_weights_into_ppo`: fold the BC observation normalization into the first
      layer; raise on shape mismatch instead of silently skipping; lower the initial
      policy sigma so the cloned behaviour survives the first rollout
- [x] Eval callback: use the same start perturbation as `evaluate.py`, and pass `reward_cfg`
- [x] Share `DEFAULT_EVAL_PERTURBATION` between `evaluate.py` and the callback
- [x] Verify PPO's deterministic action reproduces BC's exactly
- [x] (found while verifying) Expert demos contain no recovery behaviour -> add action-noise
      injection during collection
- [x] Regenerate expert data + BC, smoke-test the full PPO pipeline

## Review

### BC -> PPO transfer

Three independent mismatches made the transfer meaningless. All are now fixed:

1. **LayerNorm.** `RacingMLP` was `Linear -> LayerNorm -> Tanh`, SB3's `mlp_extractor` is
   `Linear -> Tanh`. Weights trained with LayerNorm produce entirely different activations
   without it. LayerNorm removed; `RacingMLP`'s docstring now states the constraint.
2. **Observation normalization.** BC trained on `(obs - mu) / sigma`, PPO fed raw obs.
   Rather than adding a wrapper every caller must remember, the normalization is folded
   into the first layer, which is exact:
   `W1 @ ((x - mu)/sigma) + b1  ==  (W1/sigma) @ x + (b1 - W1 @ (mu/sigma))`.
3. **Output squashing.** `RacingMLP.forward` applied `tanh`, but SB3 uses `action_net`
   output directly as a Gaussian mean (`squash_output=False`). Removed; `predict` clips.

Also: SB3 initializes `log_std = 0`, i.e. sigma = 1 on a [-1, 1] action space, which is
near-uniform noise and washes out the cloned behaviour in the first rollout. Now set to
`log(0.2)` on BC init. And the function raises on a shape mismatch instead of printing a
warning and silently continuing with random weights.

Verified: `max |BC(normalized obs) - PPO_deterministic(raw obs)| = 1.7e-07` over 200 real
observations. The transferred policy *is* the BC policy.

### Expert demos had no recovery behaviour

Found while verifying the above. With the transfer exact, BC still only reached 0.64/4
gates while the expert reached 4.00/4. Open-loop fit was excellent (RMSE 0.011) but
closed-loop it crashed 50/50, drifting from ~1 sigma to 44 sigma off the demo distribution
within 300 steps -- textbook covariate shift. The demos only randomized the *start* by
0.3 m, so they contained no state a drifting policy would actually reach.

Fix: `generate_expert_data(action_noise=0.1)` adds Gaussian noise to the **executed**
action while recording the **clean** expert action as the label, so the dataset gains
(off-trajectory state -> corrective action) pairs. The expert is a feedback controller, so
it produces the right correction from any state for free.

Sweep over noise levels (200 episodes, BC closed-loop over 50 episodes):

| noise | BC val loss | BC mean gates | BC full laps | BC crashes |
|---|---|---|---|---|
| 0.00 | 4.2e-04 | 0.68 / 4 | 0 / 50 | 21 / 50 |
| 0.05 | 1.3e-03 | 3.08 / 4 | 30 / 50 | 6 / 50 |
| **0.10** | 1.9e-03 | **3.48 / 4** | **40 / 50** | 9 / 50 |
| 0.20 | 2.9e-03 | 3.00 / 4 | 30 / 50 | 16 / 50 |
| 0.30 | 4.3e-03 | 3.02 / 4 | 32 / 50 | 18 / 50 |

Validation loss gets monotonically *worse* as closed-loop performance improves -- open-loop
BC loss is the wrong thing to optimize here.

### Training eval callback

`DroneRacingEnv(track_name=...)` was built with `initial_perturbation=0`. With deterministic
dynamics, a deterministic reset and `deterministic=True` prediction, all 50 eval episodes
were byte-identical -- the curriculum promoted and demoted on a sample size of one, which is
why its history contains only exact 1.0 and exact 0.333. It now uses
`DEFAULT_EVAL_PERTURBATION`, shared with `evaluate.py` so the two cannot drift, and is
passed `reward_cfg` so the logged breakdown describes the reward actually being trained on.

### Results

Rebuilt with 500 episodes at `action_noise=0.1`
(`outputs/expert_data/expert_simple_oval_20260922_131412`,
`outputs/bc/bc_simple_oval_20260922_131413`), evaluated over 50 episodes at perturbation 0.3:

| | mean gates | full laps | crashes |
|---|---|---|---|
| PPO, random init, 0 steps | 0.06 / 4 | 0 / 50 | 29 |
| BC standalone | 4.00 / 4 | 50 / 50 | 0 |
| **PPO, BC init, 0 steps (deterministic)** | **3.94 / 4** | **49 / 50** | 0 |
| PPO, BC init, 0 steps (sampled, sigma=0.2) | 3.94 / 4 | 49 / 50 | 0 |

PPO now *starts* from a policy that flies the track, and sampling at sigma=0.2 does not
destroy it. Eval callback at curriculum level 1 reports 96.67% (would promote) where the
old run reported a hard 33.33% (demoted) every time.

Also fixed a cosmetic reporting bug: a run shorter than `eval_freq` printed
"Best gate success: -100.00%" from the sentinel init.

`pytest tests/ -q`: 31 passed, 2 failed -- the same two pre-existing stale reward-constant
assertions, unchanged. Full PPO pipeline smoke-tested end to end at 12k steps.

### Still outstanding

Physics randomization is a silent no-op under Gymnasium 1.2 (and hides a compounding
in-place parameter drift if naively "fixed"); curriculum level 0 ends the episode after one
gate and teaches the agent to stop there; promoting to an all-gates level keeps a stale
3-gate count; the progress reward is relative rather than telescoping; most of
`configs/default.yaml` never reaches the code.

---

# Fix the progress reward and the curriculum, then retrain

## Tasks

- [x] Make the progress reward telescoping (metres closed) instead of relative
- [x] Retune `progress_weight` so the dense term actually guides between gates
- [x] Redesign the curriculum so every level flies the full track
- [x] Fix the stale `_active_gate_count` when promoting to an all-gates level
- [x] Apply curriculum settings through `unwrapped` (Gymnasium 1.x drops attribute forwarding)
- [x] Remove the dead `max_speed` curriculum field
- [x] Repair the two stale reward tests; add guards for the telescoping property
- [x] Expose `--eval-freq` / `--checkpoint-freq`
- [x] Launch the 2.5M-step retrain

## Review

### Progress reward

Was `(d_prev - d_curr) / d_prev` -- a *relative* measure. Closing 0.1 m was worth 100x more
at 0.2 m from a gate than at 20 m, so hovering beside a gate paid better than flying the
track, and a whole perfect approach summed to ~4 against a 200 gate bonus. Now
`progress_weight * clip(d_prev - d_curr, +/-progress_clip)`: metres closed, so an episode
telescopes to (start distance - end distance) and oscillating nets exactly zero.

`progress_weight` 1.0 -> 10.0. Gates are ~2.2 m apart on `simple_oval`, so approaching one
is now worth ~22 against a 200 gate bonus -- enough to guide, not enough to compete with
passing. Measured in training eval: `R_progress` went from +0.81 to +85 per episode.

`progress_clip` (1.0 m) guards a pathological jump only; at 5 m/s and a 1/48 s policy step
a normal step closes at most ~0.10 m, so it never binds in flight.

The velocity term is close to a rescaling of this (progress ~= speed_toward * dt), so the
two add rather than measure different things. Noted in the code; left alone to keep this
change contained.

### Curriculum

Levels were `1 gate -> 3 gates -> full -> full+noise`. SB3 *does* bootstrap the value on
truncation, so an N-gate level is not literally "stop after N gates", but two things still
made it harmful: the agent only ever saw the opening of the track (500k steps at a time on
a 1-gate task), and the bootstrapped terminal value is read at an observation where
`_get_next_gate()` has wrapped modulo back to gate 0. With a BC init that already flies the
whole track, level 0 was pure regression.

Levels now all fly the full track and scale difficulty on robustness axes instead:

| level | start perturbation | obs noise |
|---|---|---|
| full_track | 0.3 m | off |
| full_track_wide_start | 0.8 m | off |
| full_track_noise | 0.8 m | on (0.02) |

`min_steps_per_level` 500000 -> 250000, now a multiple of `eval_freq` so a level boundary
lands on an evaluation; previously only every other eval could act. Promotion 0.75 -> 0.9,
demotion 0.5 -> 0.6, sized for a policy that already starts near ceiling.

Two bugs fixed in `CurriculumWrapper`:

- Promoting to an `n_gates: -1` level left the previous level's `_active_gate_count` in
  place, so every level above the last gate-limited one silently kept running the shorter
  track. The count is now always assigned, including the `None` case.
- Settings are written to `self.env.unwrapped` rather than probed with
  `hasattr(self.env, ...)`. Gymnasium 1.x removed `Wrapper.__getattr__`, so the old form
  silently dropped the setting whenever another wrapper sat in between -- the same failure
  that made `PhysicsRandomizationWrapper` a no-op.

`max_speed` was read from config into `CurriculumLevel` and then used by nothing. Removed
rather than left as a decorative knob.

### Tests

`test_gate_passage_bonus` and `test_collision_penalty` asserted gate_bonus 100 /
collision -50, stale since commit 8fcb8c9 changed them to 200 / -150. They now read the
constants from `RewardConfig` so they track the config instead of drifting from it.

Added `TestProgressIsTelescoping`: reward per metre is independent of distance to the gate,
an out-and-back nets zero, and a summed traverse equals the total distance closed. These
would all have failed against the old relative form.

`pytest tests/ -q`: **36 passed**, 0 failed (was 31 passed / 2 failed).

### Verification before the retrain

BC-init policy under the new reward, 50 episodes per level:

| level | mean gates | full laps | crashes | mean reward |
|---|---|---|---|---|
| full_track | 4.00 / 4 | 50 / 50 | 0 | 913.9 |
| full_track_wide_start | 3.44 / 4 | 42 / 50 | 4 | 752.1 |
| full_track_noise | 3.72 / 4 | 46 / 50 | 2 | 834.2 |

Level 0 sits at 1.00 (above the 0.9 promotion threshold, so it will move up); level 1 sits
at 0.86 (between the thresholds, so it stays and improves). The curriculum has somewhere to
go in both directions rather than oscillating.

40k-step smoke test: gate_success **97.00% -> 98.50%**, crash rate 0.0%, `R_gate` +788 of a
possible 800. PPO now improves on the BC init rather than degrading it.

### Retrain result (outputs/ppo/ppo_simple_oval_20260922_132558)

Curriculum promoted 0 -> 1 -> 750k -> 2 at 1.0M and stayed there. No oscillation (the
previous run flipped between levels five times in 2.5M steps).

Final `best_model.zip`, 100 episodes per level, deterministic, verified independently of
the pipeline's own report:

| level | model | mean gates | laps | crashes | lap time |
|---|---|---|---|---|---|
| full_track | BC init | 3.97 / 4 | 99 / 100 | 0 | 4.84 s |
| full_track | **PPO** | **4.00 / 4** | **100 / 100** | 0 | **3.70 s** |
| full_track_wide_start | BC init | 3.75 / 4 | 93 / 100 | 4 | 5.40 s |
| full_track_wide_start | **PPO** | **3.91 / 4** | **96 / 100** | 3 | **4.13 s** |
| full_track_noise | BC init | 3.61 / 4 | 89 / 100 | 6 | 5.83 s |
| full_track_noise | **PPO** | **3.93 / 4** | **97 / 100** | 2 | **4.08 s** |

Against the run that started this work (`ppo_simple_oval_20260323_150644`):

| | before | after |
|---|---|---|
| gate success | 28.0% | **100.0%** |
| completed laps | 3 / 100 | **100 / 100** |
| crash rate | 17% | **0%** |
| timeout rate | 80% | **0%** |
| lap time | 8.12 s | **3.71 s** |
| mean altitude | 0.79 m | 1.10 m (gates at z = 1.0) |
| R_progress | +0.81 | +74.0 |

PPO also beat its own teacher: 3.70 s against the expert demonstrations' ~4.7 s.

### Known instability -- not fixed

Evaluations oscillate rather than converging: 79% (250k) -> 100% (1.25M) -> 72.5% (2.0M)
-> 100% (2.25M) -> 96% (2.5M). `best_model` selection caught a 100% point, so the saved
artifact is good, but that is partly luck and a rerun with a different seed could save a
worse checkpoint. Two causes, both visible in the logs:

- `approx_kl` ran 0.02-0.06 with `clip_fraction` up to 0.40. `n_epochs=10` at
  `clip_range=0.2` with no `target_kl` over-fits each rollout batch.
- Policy sigma drifted 0.20 -> 0.38-0.46 over the run. `ent_coef=0.01` pushes exploration
  up steadily, which is counterproductive for a policy that starts near ceiling.

Suggested: set `target_kl≈0.02`, drop `ent_coef` to ~0.001, and wire the `lr_schedule:
linear` that `configs/default.yaml` declares but `train_ppo` ignores.
