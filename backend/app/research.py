"""Imports the research modules (model6.py, extra_feats.py, nli_features.py) from backend/research WITHOUT copying or editing them.
model6.py also imports synth.py, matplotlib and scikit-learn at module level, so those must be present too."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def load(research_dir: Path):
    p = str(Path(research_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    return importlib.import_module("model6"), importlib.import_module("extra_feats"), importlib.import_module("nli_features")
