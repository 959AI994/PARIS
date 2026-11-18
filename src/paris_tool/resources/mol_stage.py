"""Run a native recipe with the selected EDA Python runtime."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paris_tool.mol_profile import main

if __name__ == "__main__":
    main()
