import numpy as np
from pathlib import Path

def mkdir_recursive(p: Path):
    if not p.exists():
        p.mkdir(parents=True)

def get_or_create_tmp(name=".tmp"):
    p = Path(name)
    if not p.exists():
        mkdir_recursive(p)
    return p.absolute()