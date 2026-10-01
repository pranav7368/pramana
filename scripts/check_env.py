"""Compatibility entry point; configuration is read from the working directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pramana.cli.check_env import main

if __name__ == "__main__":
    raise SystemExit(main())
