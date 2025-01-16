#!/usr/bin/env python
from facetselfcal.main import main as run_facetselfcal
import os
import sys

if __name__ == "__main__":
    # Required for running with relative imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)

    run_facetselfcal()
