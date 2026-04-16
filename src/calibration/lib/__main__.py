#!/usr/bin/env python3
"""
CLI entry point for consistency_fit module.

Usage:
    python -m calibration.lib.consistency_fit [options] [csv_dir]
"""
import sys
from pathlib import Path

# Add src to path for absolute imports
src_root = Path(__file__).parent.parent.parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from calibration.lib.consistency_fit.consistency_fit import main
main()
