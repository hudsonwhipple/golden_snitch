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
