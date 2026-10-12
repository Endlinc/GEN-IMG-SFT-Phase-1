#!/usr/bin/env python3
"""Convenience wrapper to run data cleaning filters from the project root.

Defaults:
- Output directory: <repo>/output_default
- Report filename: filter_report_YYYYMMDD_HHMMSS.json
"""

import os
import sys
from datetime import datetime

from src.cli import main

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_DIR, "output_default")


def _build_argv():
    argv = list(sys.argv[1:])
    if "--output" not in argv:
        argv.extend(["--output", DEFAULT_OUTPUT_DIR])
    if "--report-name" not in argv:
        # Named runs write a stable report under <output>/<run-name>/.
        # Without --run-name, timestamp the filename so repeated dumps
        # into the same --output do not overwrite each other.
        if "--resume" in argv or "--run-name" in argv:
            argv.extend(["--report-name", "filter_report.json"])
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            argv.extend(["--report-name", f"filter_report_{timestamp}.json"])
    return argv


if __name__ == "__main__":
    sys.exit(main(_build_argv()))
