"""Translate the authors' Windows project root (C:\\Users\\musab\\Desktop\\SAR) to this machine's copy of the package.

Applied in-process only (no file of the package is changed): pathlib paths, open() and torch.load() see the local path.
"""
import builtins
import os
import pathlib

WIN_ROOT = "C:\\Users\\musab\\Desktop\\SAR"
LOCAL_ROOT = os.environ.get("SAR_LOCAL_ROOT", "/home/claude/pkg/SAR")


def translate(p):
    if not isinstance(p, str):
        return p
    if p.startswith(WIN_ROOT) or p.startswith(WIN_ROOT.replace("\\", "/")):
        rest = p[len(WIN_ROOT):].replace("\\", "/").lstrip("/")
        return os.path.join(LOCAL_ROOT, rest)
    if "\\" in p and (p.startswith(".\\") or p.startswith(LOCAL_ROOT)):
        head = LOCAL_ROOT if p.startswith(LOCAL_ROOT) else ""
        rest = p[len(head):].replace("\\", "/").lstrip("/")
        return os.path.normpath(os.path.join(head or LOCAL_ROOT, rest))
    return p


_init = pathlib.PurePath.__init__


def _patched_init(self, *args):
    _init(self, *[translate(a) for a in args])


pathlib.PurePath.__init__ = _patched_init
_open = builtins.open
builtins.open = lambda file, *a, **k: _open(translate(file), *a, **k)
try:
    import torch
    _load = torch.load
    torch.load = lambda f, *a, **k: _load(translate(f), *a, **k)
except ImportError:
    pass
