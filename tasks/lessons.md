# Golden Snitch - Lessons Learned

## Patterns & Rules
<!-- Updated after corrections from the user -->

- **`pkg_resources` / TensorBoard:** There is no PyPI package named `pkg_resources`; it comes from **setuptools**. If TensorBoard fails with `No module named 'pkg_resources'`, you likely have **setuptools ≥ 82**, which removed `pkg_resources`. Fix: `pip install "setuptools>=68,<82"` (project pins this in `pyproject.toml`).

## Technical Insights
<!-- Insights discovered during implementation -->

- **Euler angle integration requires the Euler-rate Jacobian:** Never add body-frame angular velocity directly to Euler angles (`orientation += omega * dt`). The correct relation is `d(rpy)/dt = T(roll, pitch) @ omega_body` where T is the ZYX Euler-rate Jacobian. Without this, orientation diverges and the drone spirals/crashes within a few timesteps. This was the root cause of 100% crash rate.

- **Config passthrough matters:** If reward config is defined in YAML but never passed to the environment constructor, tuning the config has zero effect. Always verify the config values actually reach the code that uses them.

- **Reward clipping prevents instability:** Progress reward `(d_prev - d_curr) / d_prev` is unbounded when `d_prev` is small. Always clip dense reward terms to bounded ranges (e.g., `[-1, 1]`).

- **BC weights must be connected to PPO:** If the training pipeline generates BC weights but doesn't pass `--bc-model` to the PPO stage, the entire BC stage is wasted. Verify end-to-end pipeline connectivity.

- **Eval reports need reward breakdowns:** Without per-component reward logging, it's impossible to diagnose whether the policy is earning reward from progress (approaching gates) vs. actually passing them. Always log reward component breakdowns during training eval callbacks.

- **Curriculum eval must match curriculum level:** If the training eval callback evaluates against the full track (all gates) but curriculum only exposes 1 gate, `gate_success = 1/4 = 25%` which can never reach the promotion threshold. The eval callback must apply the same CurriculumWrapper and divide by active gates, not total track gates. This caused the policy to be stuck at level 0 forever.

- **Eval perturbation is essential:** Running eval with `deterministic=True` and `initial_perturbation=0` produces identical trajectories every episode. This hides all variance and makes the eval meaningless — 100 episodes of the same trajectory. Always use perturbation > 0 (default 0.3m) to test robustness.

- **Tilt commands act in the body frame, not the world frame:** A position controller that maps world-frame desired acceleration straight to roll/pitch is only correct at yaw = 0. Rotate the desired horizontal acceleration into the drone's yaw frame first (`fwd = cos(yaw)*ax + sin(yaw)*ay`, `left = -sin(yaw)*ax + cos(yaw)*ay`). This is especially easy to miss when the same controller also commands yaw to follow the velocity vector: the first gate is straight ahead so it always passes, and only later gates fail, which looks like an RL exploration problem rather than a controller bug. This was the real cause of the agent never getting past gate 1.

- **A P-controller's standing error is a hard geometric constraint:** holding an acceleration `a` with gain `kp` requires a steady-state position error of `a/kp`. At kp=3 a 4 m/s^2 turn means 1.33 m of lag — larger than a 0.55 m gate half-width, so the gate is unpassable no matter how long you train. Compare `a_max/kp` against the gate half-width before blaming the policy, and add reference-acceleration feedforward so the PD only has to correct error rather than generate the whole trajectory.

- **Tilting costs lift:** at 30 degrees a quadrotor loses `g*(1 - cos(30)) = 1.31 m/s^2` of vertical acceleration. Without `thrust / (cos(roll)*cos(pitch))` compensation every trajectory sags — demos sat at z=0.85 against gates at z=1.0.

- **"Success" metrics must mean what they say:** the expert generator's `successful_episodes` incremented on `gates_passed > 0`, so it reported 500/500 while *zero* episodes completed a lap. A headline metric that can't distinguish 1 gate from a finished lap will hide a total failure for months. Check the definition before trusting a green number.

- **Never clone partial demonstrations:** BC reproduces its teacher faithfully, including where the teacher stops. The expert passed a mean of 1.47/4 gates and the trained PPO agent passed 1.12/4 — that is not an RL failure, it is a correctly learned bad teacher. Filter the demo set to complete runs before training on it.

- **Stop recording on `truncated`, not just `terminated`:** `_get_next_gate` wraps modulo the gate count, so stepping past a finished lap silently retargets gate 0 and writes contradictory observations into the dataset.

- **A BC network must be architecturally identical to the policy it initializes:** transplanting Linear weights from a net with LayerNorm into SB3's `mlp_extractor` (which has none) is not a partial transfer, it is noise — LayerNorm re-normalizes activations at run time. Same for an output `tanh`: SB3's `action_net` output is the *mean of a Gaussian* (`squash_output=False`), so BC must regress the action itself, not a pre-tanh logit. Check `policy.mlp_extractor.policy_net` and `policy.squash_output` before writing any weight-copy code, and assert equivalence afterwards rather than trusting it.

- **Fold observation normalization into the first layer instead of adding a wrapper:** `W1 @ ((x - mu)/sigma) + b1 == (W1/sigma) @ x + (b1 - W1 @ (mu/sigma))`. This is exact and makes the model self-contained, so no caller (training, eval callback, evaluation harness) can forget to normalize — which is the same class of bug as the eval-perturbation one below.

- **SB3's default `log_std = 0` means sigma = 1 on a [-1, 1] action space:** that is close to uniform noise. Any BC initialization is destroyed within the first rollout unless log_std is lowered too. Loading good weights into a policy that then samples randomly looks exactly like "BC init doesn't help".

- **Open-loop BC loss does not predict closed-loop performance:** our best validation loss (4.2e-4) came from the *worst* policy (0/50 laps) and loss got monotonically worse as laps improved, up to 40/50 at 1.9e-3. Judge a cloned policy by rolling it out, never by its regression loss.

- **Demonstrations must cover the states the learner will actually visit:** randomizing only the episode *start* gives demos with no recovery behaviour, so a cloned policy's small per-step errors compound with nothing in the data to correct them — ours drifted from ~1 sigma to 44 sigma off the demo distribution in 300 steps and crashed 50/50. Inject noise into the *executed* action while recording the *clean* expert action as the label. A feedback-controller expert supplies the correct recovery action from any state for free.

- **`initial_perturbation=0` makes an eval meaningless, and the fix must be applied everywhere:** this lesson was already recorded and applied to `evaluate.py`, but the training callback kept building its own env with the default 0 — so the curriculum promoted and demoted on one identical replayed trajectory for a whole 2.5M-step run. When a constant matters in two places, share it (`DEFAULT_EVAL_PERTURBATION`) rather than fixing one site.

- **Fail loudly on shape mismatch:** `_load_bc_weights_into_ppo` used to print a warning and return, leaving PPO with random weights while the log said the pipeline ran fine. Prefer raising in setup code whose whole purpose is to transfer state.

- **Shaping rewards must telescope:** express progress as *metres closed* (`d_prev - d_curr`), never as a fraction of the remaining distance (`(d_prev - d_curr) / d_prev`). The relative form is scale-dependent — it pays ~100x more for a wiggle next to a gate than for real progress across the track — and does not sum to anything meaningful, so a whole perfect approach was worth ~4 against a 200 gate bonus. Sanity-check a dense term by integrating it over an ideal episode and comparing that total against the sparse rewards it is meant to guide toward.

- **Do not build a curriculum out of "how much of the task exists":** a level that truncates once N gates are passed only ever shows the agent the opening of the track, and its bootstrapped terminal value is read at an observation whose `next_gate_idx` has wrapped back to gate 0. Scale difficulty on robustness axes instead — start displacement, observation noise, disturbances — so every level is the whole task. (SB3 does bootstrap correctly on `truncated`, so the problem is what the agent never sees, not a mistaken terminal value of zero.)

- **`hasattr(self.env, ...)` through a wrapper is a silent no-op in Gymnasium 1.x:** `Wrapper.__getattr__` was removed, so a guard like `if hasattr(self.env, '_sim')` returns False whenever another wrapper sits in between, and the setting is quietly dropped. Always reach for `self.env.unwrapped`. This one pattern disabled physics randomization entirely and would have hit the curriculum wrapper too.

- **Assign every branch of a setting, including the "no limit" case:** `CurriculumWrapper` only wrote `_active_gate_count` when `n_gates > 0`, so promoting to an all-gates level left the previous level's limit in place and silently kept running the shorter track. When a wrapper owns a piece of state, it must set it on every path, not just the interesting one.

- **Delete dead config knobs rather than leaving them:** `max_speed` was parsed from YAML into `CurriculumLevel` and used by nothing, so the curriculum's headline speed progression (2.0 -> 3.5 -> 5.0) never happened. A knob that looks wired is worse than a missing one — it makes failed experiments look like the idea didn't work.

- **Tests should read constants from the config object, not hardcode them:** two reward tests asserted values that a commit had changed months earlier, so they sat red and stopped protecting anything. `assert info.gate == RewardConfig().gate_bonus` keeps tracking the code.

- **SB3's `target_kl` starves the value function — never use it with a BC-initialized policy:** PPO shares one optimizer for actor and critic (`loss = policy_loss + ent_coef*entropy + vf_coef*value_loss`), and the `target_kl` check breaks out of the minibatch loop *before* `optimizer.step()`, so the value gradient for that batch is discarded along with the policy's. Starting from BC weights the policy is already expert-level while the value head is random, so `approx_kl` is high from the very first update and the check fires almost every iteration. Measured: `explained_variance` pinned at ~0.00 for 115k steps (against 0.95 by 100k without it), `value_loss` stuck at ~6e3 (against 5.6e2), advantages therefore noise, and `ep_rew_mean` collapsed 888 → −70 with a 98% crash rate. The symptom `target_kl` was meant to treat — evals oscillating 72–100% — is itself caused by the random value head, not by oversized updates. The right fix is to warm up the critic on frozen-policy rollouts before letting the policy move.

- **Fixing the metric you measured is not the same as fixing the problem:** the hyperparameter change did exactly what it was designed to do — policy sigma held at 0.199 instead of drifting to 0.26, `approx_kl` fell from 0.06 to 0.003–0.024, the LR decayed on schedule — and the run still came out worse on every outcome that mattered (lap time 4.26s vs 3.71s, curriculum level 1 vs 2, a catastrophic early collapse). Diagnose against outcomes, and treat a clean-looking diagnostic trace as weak evidence when the outcome disagrees.
