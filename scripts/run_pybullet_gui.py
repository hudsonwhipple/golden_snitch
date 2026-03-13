#!/usr/bin/env python3
"""
Minimal PyBullet GUI demo: opens a 3D window with a ground plane and a box.
Run with: python scripts/run_pybullet_gui.py
Requires: pybullet installed (e.g. pip install pybullet, or conda install -c conda-forge pybullet).
"""
import time

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("PyBullet is not installed in this Python environment.")
    print("You're likely in the project .venv; PyBullet may be in conda base instead.")
    print("")
    print("Options:")
    print("  1. Run with conda's Python (if you installed pybullet via conda):")
    print("     conda run -n base python scripts/run_pybullet_gui.py")
    print("  2. Or deactivate the venv and run with base Python:")
    print("     deactivate")
    print("     python scripts/run_pybullet_gui.py")
    print("  3. Or install PyBullet into the venv (may fail on Apple Silicon):")
    print("     pip install pybullet")
    raise SystemExit(1)

# Connect to the GUI (this opens the window)
client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# Ground plane
p.loadURDF("plane.urdf")
# A simple box above the ground
box = p.loadURDF("cube.urdf", [0, 0, 0.5])

print("PyBullet GUI opened. You should see a 3D window with a ground plane and a cube.")
print("Close the window or press Ctrl+C to exit.")

try:
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 60.0)
except KeyboardInterrupt:
    pass
finally:
    p.disconnect()
