"""Run the project test suite.

From the repository root:

    python code/run_tests.py
"""

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent


def main() -> int:
    sys.path.insert(0, str(CODE_ROOT))
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CODE_ROOT / "tests"),
        pattern="*test*.py",
        top_level_dir=str(CODE_ROOT),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
