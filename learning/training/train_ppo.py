"""PPO training pipeline with BC weight initialization, curriculum, and checkpointing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environments.racing_env import DroneRacingEnv
from environments.wrappers.curriculum_wrapper import CurriculumWrapper
from environments.wrappers.domain_randomization import PhysicsRandomizationWrapper, SensorNoiseWrapper
from learning.evaluation.evaluate import DEFAULT_EVAL_PERTURBATION
from learning.training.curriculum import CurriculumManager
from learning.policies.state_policy import RacingMLP
from learning.utils.model_io import save_ppo_model
from utils.config import load_config, generate_run_id, save_config
from utils.tracking import init_wandb, log_metrics, finish_wandb

# Initial policy sigma when starting from BC weights. SB3's default log_std=0 means
# sigma=1 on a [-1, 1] action space, which drowns the cloned behaviour immediately.
BC_INIT_LOG_STD = float(np.log(0.2))


class RacingCheckpointCallback(BaseCallback):
    """Custom callback for periodic checkpointing and curriculum evaluation."""

    def __init__(
        self,
        save_dir: Path,
        track_name: str = "simple_oval",
        checkpoint_freq: int = 250_000,
        eval_freq: int = 250_000,
        eval_episodes: int = 50,
        curriculum: CurriculumManager | None = None,
        reward_cfg: dict | None = None,
        eval_perturbation: float = DEFAULT_EVAL_PERTURBATION,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.track_name = track_name
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.curriculum = curriculum
        self.reward_cfg = reward_cfg or None
        self.eval_perturbation = eval_perturbation
        self.best_gate_success = -1.0
        self._last_checkpoint_step = 0
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        step = self.num_timesteps

        if step - self._last_checkpoint_step >= self.checkpoint_freq:
            self._save_checkpoint(step)
            self._last_checkpoint_step = step

        if step - self._last_eval_step >= self.eval_freq:
            self._evaluate_and_update(step)
            self._last_eval_step = step

        return True

    def _save_checkpoint(self, step: int) -> None:
        ckpt_dir = self.save_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{step}"
        self.model.save(str(path))
        if self.curriculum:
            self.curriculum.save(str(path) + "_curriculum.json")
        if self.verbose:
            print(f"  Checkpoint saved: step {step}")

    def _evaluate_and_update(self, step: int) -> None:
        # The start perturbation is what makes these episodes differ from one another.
        # Without it the dynamics, the policy (deterministic=True) and the reset are all
        # deterministic, so every episode replays one identical trajectory and the
        # curriculum promotes/demotes on a sample size of one. Reward config must match
        # training too, or the logged breakdown describes a different reward function.
        env = DroneRacingEnv(
            track_name=self.track_name,
            initial_perturbation=self.eval_perturbation,
            reward_cfg=self.reward_cfg,
        )
        if self.curriculum is not None:
            env = CurriculumWrapper(env, level=self.curriculum.current_level)

        active_gate_count = self._get_active_gate_count(env)

        gates_passed_list = []
        collided_list = []
        reward_sums = {"gate": [], "progress": [], "velocity": [],
                       "smoothness": [], "collision": [], "time": []}

        for ep in range(self.eval_episodes):
            obs, info = env.reset(seed=1000 + ep)
            done = False
            ep_rewards = {k: 0.0 for k in reward_sums}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rb = info.get("reward_breakdown", {})
                for k in ep_rewards:
                    ep_rewards[k] += rb.get(k, 0.0)

            gates_passed_list.append(info.get("gates_passed", 0))
            collided_list.append(info.get("collided", False))
            for k in reward_sums:
                reward_sums[k].append(ep_rewards[k])

        env.close()

        gate_success = np.mean(
            [gp / max(active_gate_count, 1) for gp in gates_passed_list]
        )
        crash_rate = np.mean([1.0 if c else 0.0 for c in collided_list])
        avg_gates = np.mean(gates_passed_list)

        mean_rewards = {k: float(np.mean(v)) for k, v in reward_sums.items()}

        metrics = {
            "eval/gate_success_rate": gate_success,
            "eval/crash_rate": crash_rate,
            "eval/avg_gates_passed": avg_gates,
        }
        for k, v in mean_rewards.items():
            metrics[f"eval/reward_{k}"] = v
        if self.curriculum:
            metrics["eval/curriculum_level"] = self.curriculum.level_index

        log_metrics(metrics, step=step)

        if self.verbose:
            level_str = f" [Level {self.curriculum.level_index}]" if self.curriculum else ""
            reward_str = ", ".join(f"R_{k}={v:+.1f}" for k, v in mean_rewards.items())
            print(f"  Eval @ step {step}{level_str}: gate_success={gate_success:.2%}, "
                  f"avg_gates={avg_gates:.1f}, crash_rate={crash_rate:.1%}")
            print(f"    Rewards: {reward_str}")

        if gate_success > self.best_gate_success:
            self.best_gate_success = gate_success
            self.model.save(str(self.save_dir / "best_model"))
            if self.verbose:
                print(f"  New best model! gate_success={gate_success:.2%}")

        if self.curriculum:
            result = self.curriculum.update(gate_success, step)
            if result:
                print(f"  Curriculum: {result} to level {self.curriculum.level_index} "
                      f"({self.curriculum.current_level.name})")
                self._apply_curriculum_to_envs()

    def _get_active_gate_count(self, env) -> int:
        """Return the number of gates the current curriculum level exposes."""
        if self.curriculum is not None:
            n = self.curriculum.current_level.n_gates
            if n > 0:
                return n
        base = env.unwrapped if hasattr(env, "unwrapped") else env
        return len(base.track.gates)

    def _apply_curriculum_to_envs(self) -> None:
        """Propagate the current curriculum level to all wrapped training envs."""
        if not self.curriculum:
            return
        level = self.curriculum.current_level
        vec_env = self.model.get_env()
        for i in range(vec_env.num_envs):
            env = vec_env.envs[i]
            while env is not None:
                if isinstance(env, CurriculumWrapper):
                    env.set_level(level)
                    break
                env = getattr(env, "env", None)


def _load_bc_weights_into_ppo(
    model: PPO, bc_model_path: str, init_log_std: float = BC_INIT_LOG_STD
) -> None:
    """Copy BC backbone + action head weights into the SB3 PPO actor network.

    ``RacingMLP`` is deliberately shaped like SB3's actor (see its docstring), so the
    Linear layers map across one-for-one:

        RacingMLP.backbone       -> policy.mlp_extractor.policy_net
        RacingMLP.action_head    -> policy.action_net

    Two things still have to be reconciled.

    **Observation normalization.** BC trains on ``(obs - mean) / std`` while the PPO env
    emits raw observations. Rather than making every caller remember to wrap the env, the
    normalization is folded into the first layer, which is exact::

        W1 @ ((x - mu) / sigma) + b1  ==  (W1 / sigma) @ x + (b1 - W1 @ (mu / sigma))

    **Policy standard deviation.** SB3 initializes ``log_std = 0``, i.e. sigma = 1 on an
    action space of [-1, 1]. Sampling from that is close to uniform noise and washes the
    cloned behaviour out within the first rollout, so it is lowered to keep exploration
    local to what BC learned.

    The value network is left with its default init -- BC has nothing to say about it.
    """
    bc = RacingMLP()
    state_dict = torch.load(bc_model_path, map_location="cpu", weights_only=True)
    bc.load_state_dict(state_dict)

    bc_linears = [m for m in bc.backbone if isinstance(m, torch.nn.Linear)]

    policy = model.policy
    ppo_linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]

    if len(bc_linears) != len(ppo_linears):
        raise ValueError(
            f"BC backbone has {len(bc_linears)} linear layers but the PPO actor has "
            f"{len(ppo_linears)}. RacingMLP must mirror the SB3 actor for the transfer "
            f"to be valid -- check net_arch and RacingMLP.hidden_dims."
        )
    for i, (bc_layer, ppo_layer) in enumerate(zip(bc_linears, ppo_linears)):
        if bc_layer.weight.shape != ppo_layer.weight.shape:
            raise ValueError(
                f"BC layer {i} has shape {tuple(bc_layer.weight.shape)} but the PPO actor "
                f"expects {tuple(ppo_layer.weight.shape)}."
            )
    if policy.action_net.weight.shape != bc.action_head.weight.shape:
        raise ValueError(
            f"BC action head has shape {tuple(bc.action_head.weight.shape)} but "
            f"action_net expects {tuple(policy.action_net.weight.shape)}."
        )

    obs_mean, obs_std = _load_bc_obs_stats(bc_model_path)

    with torch.no_grad():
        for bc_layer, ppo_layer in zip(bc_linears, ppo_linears):
            ppo_layer.weight.copy_(bc_layer.weight)
            ppo_layer.bias.copy_(bc_layer.bias)

        policy.action_net.weight.copy_(bc.action_head.weight)
        policy.action_net.bias.copy_(bc.action_head.bias)

        if obs_mean is not None:
            first = ppo_linears[0]
            # Match the policy's device, not the default one: the policy may be on mps/cuda.
            mean_t = torch.as_tensor(obs_mean, dtype=first.weight.dtype, device=first.weight.device)
            std_t = torch.as_tensor(obs_std, dtype=first.weight.dtype, device=first.weight.device)
            first.bias.sub_(first.weight @ (mean_t / std_t))
            first.weight.div_(std_t)

        policy.log_std.fill_(init_log_std)

    stats_note = "with obs normalization folded in" if obs_mean is not None else (
        "WITHOUT obs normalization (no obs_mean.npy/obs_std.npy beside the model)"
    )
    print(f"  BC weights loaded from {bc_model_path} ({stats_note}), "
          f"log_std init {init_log_std:.2f} (sigma={np.exp(init_log_std):.2f})")


def _make_lr_schedule(learning_rate: float, schedule: str):
    """Build the value SB3 wants for ``learning_rate``.

    SB3 accepts a float or a callable of ``progress_remaining`` (1.0 at the start, 0.0 at
    the end). ``configs/default.yaml`` has declared ``lr_schedule: linear`` since the
    project started, but a constant float was always passed, so the decay never happened.
    """
    if schedule in (None, "", "constant"):
        return learning_rate
    if schedule == "linear":
        return lambda progress_remaining: progress_remaining * learning_rate
    raise ValueError(f"Unknown lr_schedule {schedule!r}; expected 'linear' or 'constant'.")


_ACTIVATIONS = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "elu": torch.nn.ELU}


def _resolve_policy_kwargs(policy_kwargs: dict | None) -> dict:
    """Turn the config's policy_kwargs into SB3's form (activation_fn as a class).

    ``net_arch`` must stay in step with ``RacingMLP.hidden_dims`` or the BC transfer has
    nothing to map onto; `_load_bc_weights_into_ppo` raises if they diverge.
    """
    resolved = dict(policy_kwargs or {})
    resolved.setdefault("net_arch", [256, 256])
    activation = resolved.get("activation_fn", "tanh")
    if isinstance(activation, str):
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation_fn {activation!r}; expected one of {sorted(_ACTIVATIONS)}."
            )
        resolved["activation_fn"] = _ACTIVATIONS[activation]
    return resolved


def _load_bc_obs_stats(bc_model_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load the observation mean/std that ``train_bc`` saved beside the model."""
    bc_dir = Path(bc_model_path).parent
    mean_path, std_path = bc_dir / "obs_mean.npy", bc_dir / "obs_std.npy"
    if not (mean_path.exists() and std_path.exists()):
        return None, None
    return np.load(mean_path), np.load(std_path)


def train_ppo(
    track_name: str = "simple_oval",
    bc_model_path: str | None = None,
    output_dir: str = "outputs/ppo",
    total_timesteps: int = 5_000_000,
    n_envs: int = 4,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    clip_range: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.001,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float | None = 0.02,
    lr_schedule: str = "linear",
    policy_kwargs: dict | None = None,
    checkpoint_freq: int = 250_000,
    eval_freq: int = 250_000,
    eval_episodes: int = 50,
    seed: int = 42,
    use_curriculum: bool = True,
    resume_from: str | None = None,
    curriculum_state_path: str | None = None,
    domain_randomization: bool = True,
    cfg: dict | None = None,
    device: str = "cpu",
) -> Path:
    """Train a PPO policy for drone racing.

    Parameters
    ----------
    track_name : name of the track
    bc_model_path : optional path to BC model for weight initialization
    output_dir : base output directory
    total_timesteps : total training steps
    n_envs : number of parallel environments
    resume_from : optional checkpoint path to resume training
    cfg : full config dict for logging/saving
    device : torch device

    Returns
    -------
    Path to the output directory
    """
    run_id = generate_run_id("ppo", track_name)
    save_dir = Path(output_dir) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = init_wandb(
        run_id, cfg or {},
        tags=["stage:ppo", f"track:{track_name}"],
    )

    curriculum = None
    if use_curriculum and cfg:
        curriculum = CurriculumManager.from_config(cfg)
        if curriculum_state_path and Path(curriculum_state_path).exists():
            curriculum.load(curriculum_state_path)
            print(f"Curriculum state restored from {curriculum_state_path} "
                  f"(level {curriculum.level_index}: {curriculum.current_level.name})")

    reward_cfg = cfg.get("reward", {}) if cfg else {}

    def make_env(env_seed):
        def _init():
            env = DroneRacingEnv(
                track_name=track_name,
                initial_perturbation=0.3,
                reward_cfg=reward_cfg if reward_cfg else None,
            )
            if curriculum is not None:
                env = CurriculumWrapper(env, level=curriculum.current_level)
            if domain_randomization:
                env = PhysicsRandomizationWrapper(env)
                env = SensorNoiseWrapper(env)
            env = Monitor(env)
            env.reset(seed=env_seed)
            return env
        return _init

    env = DummyVecEnv([make_env(seed + i) for i in range(n_envs)])

    policy_kwargs = _resolve_policy_kwargs(policy_kwargs)

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        model = PPO.load(resume_from, env=env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=_make_lr_schedule(learning_rate, lr_schedule),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device=device,
            tensorboard_log=str(save_dir / "tb_logs"),
        )

        if bc_model_path:
            _load_bc_weights_into_ppo(model, bc_model_path)

    callback = RacingCheckpointCallback(
        save_dir=save_dir,
        track_name=track_name,
        checkpoint_freq=checkpoint_freq,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        curriculum=curriculum,
        reward_cfg=reward_cfg if reward_cfg else None,
    )

    save_config(cfg or {}, save_dir / "training_config.yaml")

    print(f"Starting PPO training: {total_timesteps} steps, {n_envs} envs")
    print(f"Domain randomization: {'enabled' if domain_randomization else 'disabled'}")
    print(f"Output: {save_dir}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    model.save(str(save_dir / "final_model"))
    if curriculum:
        curriculum.save(save_dir / "curriculum_history.json")

    finish_wandb()
    env.close()

    best = callback.best_gate_success
    summary = f"{best:.2%}" if best >= 0 else f"n/a (no eval ran; eval_freq={eval_freq})"
    print(f"\nPPO training complete. Best gate success: {summary}")
    print(f"Models saved to {save_dir}")

    return save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO policy for drone racing")
    parser.add_argument("--track", default="simple_oval")
    parser.add_argument("--bc-model", default=None, help="Path to BC model for initialization")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None,
                        help="steps between curriculum evaluations")
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--curriculum-state", default=None, help="Path to curriculum JSON to restore")
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--no-domain-rand", action="store_true", help="Disable physics/sensor domain randomization")
    args = parser.parse_args()

    cfg = load_config()
    ppo_cfg = cfg.get("ppo", {})
    env_cfg = cfg.get("env", {})

    def pick(cli_value, section: dict, key: str, fallback):
        """CLI flag wins when given, then configs/default.yaml, then the fallback.

        Everything the config declares under `ppo:` and `env:` has to be read here --
        train_ppo's own parameter defaults used to win silently, so editing the YAML did
        nothing and tuning experiments looked like the idea had failed.
        """
        return cli_value if cli_value is not None else section.get(key, fallback)

    train_ppo(
        track_name=args.track,
        bc_model_path=args.bc_model,
        output_dir=pick(args.output_dir, ppo_cfg, "output_dir", "outputs/ppo"),
        total_timesteps=pick(args.timesteps, ppo_cfg, "total_timesteps", 2_500_000),
        n_envs=pick(args.n_envs, env_cfg, "n_envs", 4),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 256),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        lr_schedule=ppo_cfg.get("lr_schedule", "linear"),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        ent_coef=ppo_cfg.get("ent_coef", 0.001),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        target_kl=ppo_cfg.get("target_kl", 0.02),
        policy_kwargs=ppo_cfg.get("policy_kwargs"),
        eval_freq=pick(args.eval_freq, ppo_cfg, "eval_freq", 250_000),
        checkpoint_freq=pick(args.checkpoint_freq, ppo_cfg, "checkpoint_freq", 250_000),
        eval_episodes=ppo_cfg.get("eval_episodes", 50),
        seed=pick(args.seed, cfg, "seed", 42),
        device=pick(args.device, cfg, "device", "cpu"),
        resume_from=args.resume,
        curriculum_state_path=args.curriculum_state,
        use_curriculum=not args.no_curriculum,
        domain_randomization=not args.no_domain_rand,
        cfg=cfg,
    )
