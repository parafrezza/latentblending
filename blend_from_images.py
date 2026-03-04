#!/usr/bin/env python3
"""Deprecated wrapper for the old blend_from_images utility.

The actual functionality now lives in ``preprocess_images.py`` under the
``blend`` subcommand. This stub forwards the arguments and emits a warning.
"""

import os
import subprocess
import sys

print("WARNING: blend_from_images.py is deprecated; use 'preprocess_images.py blend' instead.")

here = os.path.dirname(__file__)
new_script = os.path.join(here, "preprocess_images.py")
if not os.path.isfile(new_script):
    new_script = os.path.join(os.getcwd(), "preprocess_images.py")

cmd = [sys.executable, new_script, "blend"] + sys.argv[1:]
return_code = subprocess.call(cmd)
sys.exit(return_code)
