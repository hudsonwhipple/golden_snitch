"""Tests for the PyBullet racing environment."""

import numpy as np
import pytest

from environments.tracks.track_loader import load_track, get_available_tracks


class TestTrackLoader:
    def test_load_simple_oval(self):
        track = load_track("simple_oval")
        assert track.name == "simple_oval"
        assert track.n_gates == 4
        assert track.start_position.shape == (3,)

    def test_load_figure_eight(self):
        track = load_track("figure_eight")
        assert track.name == "figure_eight"
        assert track.n_gates == 6

    def test_load_competition(self):
        track = load_track("competition")
        assert track.name == "competition"
        assert track.n_gates == 8

    def test_get_available_tracks(self):
        tracks = get_available_tracks()
        assert "simple_oval" in tracks
        assert "figure_eight" in tracks
        assert "competition" in tracks

    def test_gate_properties(self):
        track = load_track("simple_oval")
        for gate in track.gates:
            assert gate.position.shape == (3,)
            assert gate.orientation.shape == (3,)
            assert gate.size.shape == (2,)
            assert np.all(gate.size > 0)

    def test_missing_track_raises(self):
        with pytest.raises(FileNotFoundError):
            load_track("nonexistent_track")
