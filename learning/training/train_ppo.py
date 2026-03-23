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
from learning.training.curriculum import CurriculumManager
from learning.policies.state_policy import RacingMLP
from learning.utils.model_io import save_ppo_model
from utils.config import load_config, generate_run_id, save_config
from utils.tracking import init_wandb, log_metrics, finish_wandb


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
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.track_name = track_name
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.curriculum = curriculum
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
        env = DroneRacingEnv(track_name=self.track_name)
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


def _load_bc_weights_into_ppo(model: PPO, bc_model_path: str) -> None:
    """Copy BC backbone + action head weights into the SB3 PPO actor network.

    RacingMLP backbone: [Linear(24,256), LayerNorm, Tanh, Linear(256,256), LayerNorm, Tanh]
    SB3 MlpPolicy actor mlp_extractor: [Linear(24,256), Tanh, Linear(256,256), Tanh]
    action_net: Linear(256, 4)

    We copy only the Linear weight/bias tensors, skipping LayerNorm since SB3's
    policy doesn't have them. The value network is left with its default init.
    """
    bc = RacingMLP()
    state_dict = torch.load(bc_model_path, map_location="cpu", weights_only=True)
    bc.load_state_dict(state_dict)

    # Extract Linear layers from BC backbone (indices 0 and 3 in the Sequential)
    bc_linears = [m for m in bc.backbone if isinstance(m, torch.nn.Linear)]

    policy = model.policy
    # SB3 stores the shared MLP extractor layers in policy.mlp_extractor.policy_net
    ppo_linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]

    if len(bc_linears) != len(ppo_linears):
        print(f"  Warning: BC has {len(bc_linears)} linear layers but PPO actor has "
              f"{len(ppo_linears)}. Skipping BC weight transfer.")
        return

    with torch.no_grad():
        for bc_layer, ppo_layer in zip(bc_linears, ppo_linears):
            if bc_layer.weight.shape == ppo_layer.weight.shape:
                ppo_layer.weight.copy_(bc_layer.weight)
                ppo_layer.bias.copy_(bc_layer.bias)

        # Copy action head into policy.action_net
        if policy.action_net.weight.shape == bc.action_head.weight.shape:
            policy.action_net.weight.copy_(bc.action_head.weight)
            policy.action_net.bias.copy_(bc.action_head.bias)

    print(f"  BC weights loaded from {bc_model_path}")


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
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
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

    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.Tanh,
    )

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        model = PPO.load(resume_from, env=env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
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

    print(f"\nPPO training complete. Best gate success: {callback.best_gate_success:.2%}")
    print(f"Models saved to {save_dir}")

    return save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO policy for drone racing")
    parser.add_argument("--track", default="simple_oval")
    parser.add_argument("--bc-model", default=None, help="Path to BC model for initialization")
    parser.add_argument("--output-dir", default="outputs/ppo")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--curriculum-state", default=None, help="Path to curriculum JSON to restore")
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--no-domain-rand", action="store_true", help="Disable physics/sensor domain randomization")
    args = parser.parse_args()

    cfg = load_config()
    train_ppo(
        track_name=args.track,
        bc_model_path=args.bc_model,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
        resume_from=args.resume,
        curriculum_state_path=args.curriculum_state,
        use_curriculum=not args.no_curriculum,
        domain_randomization=not args.no_domain_rand,
        cfg=cfg,
    )
