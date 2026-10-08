import os
import sys

# Headless plotting in tests: with DISPLAY set (WSLg), pyplot may otherwise pick a GUI backend and abort.
os.environ.setdefault("MPLBACKEND", "Agg")

# Make `import modules.*` and `import anarci` work from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
