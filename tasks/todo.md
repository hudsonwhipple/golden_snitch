# Make racing gates solid

## Problem

`_check_collision` tests only the ground (z < 0.05) and a +/-20 m box. No gate geometry
appears anywhere in collision detection, so clipping a gate frame costs the drone nothing
and the agent has no incentive to centre itself -- only to get its centre point through the
window. The 100/100-lap result is measured against that permissive environment.

Measured over 240 gate crossings, window half-extent 0.50 m:

| | mean lateral offset | p95 | max |
|---|---|---|---|
| expert controller | 0.088 m | 0.203 m | 0.382 m |
| trained PPO policy | 0.282 m | 0.459 m | 2.231 m |

The policy's p95 crossing sits 4 cm from the window edge. The expert is unaffected by a
frame (60/60 clean laps even at a 0.10 m radius), so expert data and BC do not need
regenerating -- only PPO does.

## Geometry

clean pass  d <= half - radius           = 0.42 m
FRAME HIT   0.42 < d <= half + thick + r = 0.63 m
clean miss  d  > 0.63 m

where d = max(|local_y|, |local_z|) at the swept segment's plane-crossing point.
frame_thickness 0.05 m, collision_radius 0.08 m (Crazyflie 2.x footprint).

## Tasks

- [x] `gate_detection.py`: `Gate.frame_thickness` field (defaulted, after `size`)
- [x] `gate_detection.py`: `check_gate_passage` margin -> clearance, so the window *shrinks*
      by the drone radius instead of growing by 0.05 m. Without this a crossing at d = 0.50
      scores a gate pass (+200) and a frame strike (-150) in the same step.
- [x] `gate_detection.py`: `check_gate_frame_collision`, reusing the passage skeleton; not
      direction-gated, since hitting a frame from behind is still a crash
- [x] `base_env.py`: `_check_gate_collision` on `RacingMixin` over ALL track gates
      (a curriculum-deactivated gate is still solid)
- [x] `racing_env.py`: `_check_collision` takes prev_pos, ORs in the gate check, reports cause
- [x] `pybullet_racing_env.py`: same check; `_draw_gate` shows the frame band
- [x] `airsim_racing_env.py`: opt out by default -- it reads simGetCollisionInfo(), so a
      geometric check would double-count against real gate meshes
- [x] `track_loader.py`: optional per-gate `frame_thickness`
- [x] `configs/default.yaml`: `env.collision_radius`, `env.gate_frame_thickness`; drop the
      dead `env.gate_urdf` (file does not exist, no code reads it)
- [x] Tests: three bands, rotated gate, hit from behind, and pass/hit mutual exclusivity
- [x] Verify: expert lap yield holds; re-baseline the saved model and report the honest drop

## Review

### What changed

- `gate_detection.py`: `Gate.frame_thickness` (defaulted, so existing track files load
  unchanged); a shared `_plane_crossing` helper; `check_gate_frame_collision`; and
  `check_gate_passage`'s `margin=0.05` replaced by `clearance=0.0`.
- `base_env.py`: `DEFAULT_COLLISION_RADIUS = 0.08` and `RacingMixin._check_gate_collision`,
  over ALL track gates -- a curriculum-deactivated gate is still a solid object.
- `racing_env.py` / `pybullet_racing_env.py`: `_check_collision` now takes the swept segment.
  The numpy env returns the *cause* (`ground` / `bounds` / `gate_frame`), surfaced in `info`.
- `pybullet_racing_env.py`: `_draw_gate` renders the usable aperture and both frame edges, so
  the GUI shows what collision actually enforces.
- `airsim_racing_env.py`: deliberately opted out -- it reads `simGetCollisionInfo()`, which
  already reports contact with real gate meshes, so both would double-count one strike.
- Removed the dead `env.gate_urdf` key: the file does not exist and no code reads it.

### Deviation from the plan

The plan called for `env.collision_radius` and `env.gate_frame_thickness` config keys. I did
not add them. Frame thickness belongs to the track (it is a property of the gate, and is now
a per-gate YAML field), and body radius is a property of the vehicle, not a training
hyperparameter -- a config knob would let training and evaluation drift apart, which is the
exact bug `DEFAULT_EVAL_PERTURBATION` exists to prevent. Both remain constructor arguments.

### The passage/collision conflict

`check_gate_passage` took `margin=0.05`, which *enlarged* every window to 0.55 m -- 10% wider
than the track file declared. Once gates are solid, a crossing at d = 0.50 would have scored
a gate pass (+200) and a frame strike (-150) in the same step. `clearance` shrinks the window
by the drone radius instead, making pass and strike complementary by construction. Both
existing margin tests pass unchanged under the new default (`y=0.49` True, `y=0.6` False).

### Verification

Band sweep over 401 offsets, gate size 1.0, radius 0.08:

    pass    0.000 .. 0.420 m
    strike  0.422 .. 0.630 m
    miss    0.633 .. 1.000 m     overlaps: 0

`pytest tests/ -q`: **46 passed** (36 existing + 10 new), 0 failed.

Expert over 60 episodes, clean and with `action_noise=0.1`: **60/60 complete laps, zero
collisions** in both. Expert data and BC need no regeneration, as predicted.

### Re-baseline -- the headline

100 episodes, perturbation 0.3:

| model | gate model | gate success | laps/100 | crashes |
|---|---|---|---|---|
| PPO trained (2.5M) | none (the old env) | 100% | **100/100** | 0 |
| PPO trained (2.5M) | frame only, point mass | 96.0% | 94/100 | 6 |
| PPO trained (2.5M) | **frame + 0.08 m body** | 74.0% | **54/100** | 46 |
| BC init (0 PPO steps) | frame + 0.08 m body | **96.2%** | **95/100** | 5 |

**The BC initialization beats the fully trained policy under solid gates, 95/100 against
54/100.** BC barely degrades (98 -> 95) because it clones an expert that flies centred; the
trained policy collapses (94 -> 54) because 2.5M steps taught it to cut corners, which was
free in the permissive environment and is fatal in a real one. Its 3.71 s lap time was bought
largely by clipping gates that could not punish it.

So the honest baseline on this track is **54/100**, and PPO needs re-running against solid
gates -- it currently destroys more capability than it adds.

### Known limitation

Collision only fires on a plane crossing, so a drone grazing the outer edge of a frame while
travelling parallel to it is not detected. That needs swept-volume contact against the frame
solid, which the point-mass dynamics do not model. Noted in the docstring.

### Retrain against solid gates (outputs/ppo/ppo_simple_oval_20260924_180410)

Curriculum promoted 0 -> 1 at 250k and 1 -> 2 at 750k, then held level 2 with no demotions.

All on SOLID gates, 100 episodes, perturbation 0.3:

| model | gate success | laps/100 | lap time | collisions |
|---|---|---|---|---|
| BC init (0 PPO steps) | 96.2% | 95/100 | 4.68 s | 5 gate strikes |
| PPO before solid gates | 74.0% | 54/100 | 3.71 s | 46 gate strikes |
| **PPO retrained on solid gates** | **100.0%** | **100/100** | **3.87 s** | **none** |

Retrained model by curriculum level: full_track 100/100, wide_start 92/100 (8 strikes),
noise 93/100 (7 strikes).

PPO now beats its own initialization on the honest task -- more reliable *and* 0.8 s faster
than BC -- which is exactly what the previous run failed to do once measured properly. The
100/100 here is a real number: same policy, same 100 episodes, gates that can be hit.

Training still oscillates (83.0% -> 98.5% -> 84.0% -> 95.5% across evals), so the underlying
instability is unchanged: an expert-level policy paired with a randomly initialized value
head. `best_model` selection caught a good checkpoint again. A critic warm-up remains the
outstanding fix.

Caveat: this run used `ent_coef=0.001` and `lr_schedule: linear`, which have still never been
isolated from each other.

### Re-measured for the README (2026-10-16)

`scripts/make_readme_figures.py` re-scores both committed policies on current HEAD, 100 episodes,
perturbation 0.3, seed 2000. The retrained policy reproduces: 100/100 laps, zero strikes, 3.86 s.
The pre-solid-gate policy (`models/ppo_before_solid_gates.zip`, the 3.71 s model) reproduces its
100/100 under the old thin-gate rules, but on solid gates it now scores **67/100 laps with 33 gate
strikes** (81.5% gate success), not the 54/100 and 46 strikes recorded above. The same numbers
come out of `learning.evaluation.evaluate_policy`, so the difference is not the figure script.
The README reports the reproducible figures; the cause of the gap is not yet known.
