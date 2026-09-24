"""Load race track definitions from YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from environments.gate_detection import DEFAULT_FRAME_THICKNESS, Gate
from utils.config import TRACKS_DIR


@dataclass
class Track:
    """A racing track with start pose and a sequence of gates."""

    name: str
    description: str
    start_position: np.ndarray
    start_orientation: np.ndarray
    gates: list[Gate]

    def get_gate(self, index: int) -> Gate:
        return self.gates[index % len(self.gates)]

    @property
    def n_gates(self) -> int:
        return len(self.gates)


def load_track(track_name: str, tracks_dir: str | Path | None = None) -> Track:
    """Load a track by name from the tracks config directory.

    Parameters
    ----------
    track_name : name of the track (without .yaml extension)
    tracks_dir : override directory to search for track files

    Gate frame thickness is optional and falls back per gate -> per track -> the default,
    so existing track files that predate solid gates keep working unchanged.
    """
    tracks_dir = Path(tracks_dir) if tracks_dir else TRACKS_DIR
    path = tracks_dir / f"{track_name}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Track file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    gates = []
    for g in data["gates"]:
        gates.append(
            Gate(
                id=g["id"],
                position=np.array(g["position"], dtype=np.float64),
                orientation=np.array(g["orientation"], dtype=np.float64),
                size=np.array(g["size"], dtype=np.float64),
                frame_thickness=float(
                    g.get("frame_thickness", data.get("frame_thickness", DEFAULT_FRAME_THICKNESS))
                ),
            )
        )

    return Track(
        name=data["name"],
        description=data.get("description", ""),
        start_position=np.array(data["start_position"], dtype=np.float64),
        start_orientation=np.array(data["start_orientation"], dtype=np.float64),
        gates=gates,
    )


def get_available_tracks(tracks_dir: str | Path | None = None) -> list[str]:
    """List all available track names."""
    tracks_dir = Path(tracks_dir) if tracks_dir else TRACKS_DIR
    return [p.stem for p in sorted(tracks_dir.glob("*.yaml"))]
