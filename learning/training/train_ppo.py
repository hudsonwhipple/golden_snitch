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
from learning.training.curriculum import CurriculumManager
from learning.utils.model_io import save_ppo_model
from utils.config import load_config, generate_run_id, save_config
from utils.tracking import init_wandb, log_metrics, finish_wandb


class RacingCheckpointCallback(BaseCallback):
    """Custom callback for periodic checkpointing and curriculum evaluation."""

    def __init__(
        self,
        save_dir: Path,
        checkpoint_freq: int = 250_000,
        eval_freq: int = 250_000,
        eval_episodes: int = 50,
        curriculum: CurriculumManager | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
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
        env = DroneRacingEnv(track_name="simple_oval")
        gates_passed_list = []
        total_gates_list = []

        for ep in range(self.eval_episodes):
            obs, info = env.reset(seed=1000 + ep)
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            gates_passed_list.append(info.get("gates_passed", 0))
            total_gates_list.append(len(env.track.gates))

        env.close()

        gate_success = np.mean(
            [gp / max(tg, 1) for gp, tg in zip(gates_passed_list, total_gates_list)]
        )
        crash_rate = np.mean([1 if gp == 0 else 0 for gp in gates_passed_list])
        avg_gates = np.mean(gates_passed_list)

        metrics = {
            "eval/gate_success_rate": gate_success,
            "eval/crash_rate": crash_rate,
            "eval/avg_gates_passed": avg_gates,
        }
        if self.curriculum:
            metrics["eval/curriculum_level"] = self.curriculum.level_index

        log_metrics(metrics, step=step)

        if self.verbose:
            level_str = f" [Level {self.curriculum.level_index}]" if self.curriculum else ""
            print(f"  Eval @ step {step}{level_str}: gate_success={gate_success:.2%}, "
                  f"avg_gates={avg_gates:.1f}, crash_rate={crash_rate:.1%}")

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

    def make_env(env_seed):
        def _init():
            env = DroneRacingEnv(track_name=track_name, initial_perturbation=0.3)
            env = Monitor(env)
            env.reset(seed=env_seed)
            return env
        return _init

    env = DummyVecEnv([make_env(seed + i) for i in range(n_envs)])

    curriculum = None
    if use_curriculum and cfg:
        curriculum = CurriculumManager.from_config(cfg)

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

    callback = RacingCheckpointCallback(
        save_dir=save_dir,
        checkpoint_freq=checkpoint_freq,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        curriculum=curriculum,
    )

    save_config(cfg or {}, save_dir / "training_config.yaml")

    print(f"Starting PPO training: {total_timesteps} steps, {n_envs} envs")
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
    parser.add_argument("--no-curriculum", action="store_true")
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
        use_curriculum=not args.no_curriculum,
        cfg=cfg,
    )
