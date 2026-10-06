#!/usr/bin/env python3
"""CLI wrapper: python scripts/nim_probe.py (needs NVIDIA_API_KEY + models)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.probe import main

if __name__ == "__main__":
    sys.exit(main())
