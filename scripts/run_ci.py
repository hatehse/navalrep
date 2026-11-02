"""Run the complete CPU suite; missing integration prerequisites fail CI."""

import os
from pathlib import Path
import sys
import unittest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun == 0:
        print("CI requires a nonempty test suite.", file=sys.stderr)
        return 1
    if result.skipped:
        print("CI requires all integration tests to execute; skipped tests are failures.", file=sys.stderr)
        return 1
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
