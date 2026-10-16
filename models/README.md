# Models

The two Stable-Baselines3 PPO policies compared in the [README](../README.md). Both fly the `simple_oval` track.

| File | Trained against | Source run |
|---|---|---|
| `ppo_before_solid_gates.zip` | Thin gates (no frame collision, aperture enlarged by 5 cm) | `ppo_simple_oval_20260922_132558`, best checkpoint |
| `ppo_solid_gates.zip` | Solid gate frames with a 0.08 m drone body radius | `ppo_simple_oval_20260924_180410`, best checkpoint |

Re-score both and redraw the README figures:

```bash
python scripts/make_readme_figures.py
```

Load one directly:

```python
from stable_baselines3 import PPO
model = PPO.load("models/ppo_solid_gates.zip")
```
