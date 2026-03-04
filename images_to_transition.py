#!/usr/bin/env python3
"""Compatibility wrapper for the old unified CLI name.

The active implementation lives in ``preprocess_images.py`` with subcommands:
``preprocess``, ``blend``, ``all``.
"""

import os
import subprocess
import sys

print("WARNING: images_to_transition.py is deprecated; use 'preprocess_images.py <subcommand>' instead.")

here = os.path.dirname(__file__)
new_script = os.path.join(here, "preprocess_images.py")
if not os.path.isfile(new_script):
    new_script = os.path.join(os.getcwd(), "preprocess_images.py")

cmd = [sys.executable, new_script] + sys.argv[1:]
sys.exit(subprocess.call(cmd))
