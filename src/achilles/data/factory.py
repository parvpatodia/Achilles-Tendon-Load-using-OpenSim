"""Source selection with an automatic synthetic fallback.

Keeps the "which dataset" decision in one place so every stage script shares
the same behaviour: try the real Fukuchi data; if it is missing, fall back to
the clearly-labelled synthetic source rather than crashing.
"""
from __future__ import annotations

from achilles.data.base import GaitDataSource
from achilles.data.fukuchi import FukuchiDataSource
from achilles.data.grouvel import GrouvelDataSource
from achilles.data.synthetic import SyntheticGaitSource
from achilles.data.walking import WalkingDataSource


def make_source(name: str = "fukuchi", **kwargs) -> GaitDataSource:
    name = name.lower()
    if name == "fukuchi":
        return FukuchiDataSource(**kwargs)
    if name == "walking":
        return WalkingDataSource(**kwargs)
    if name == "grouvel":
        return GrouvelDataSource(**kwargs)
    if name == "synthetic":
        return SyntheticGaitSource(**kwargs)
    raise ValueError(
        f"unknown source {name!r} (use 'fukuchi', 'walking', 'grouvel', or 'synthetic')"
    )


def resolve_source(name: str = "fukuchi") -> tuple[GaitDataSource, str]:
    """Return (source, resolved_name), falling back to synthetic if real data
    is absent or unreadable."""
    if name == "synthetic":
        return SyntheticGaitSource(), "synthetic"
    try:
        src = FukuchiDataSource()
        if len(src.load_trials()) == 0:
            raise RuntimeError("no usable Fukuchi trials found")
        return src, "fukuchi"
    except Exception as e:  # noqa: BLE001
        print(f"[data] real dataset unavailable ({e}); using synthetic fallback.")
        return SyntheticGaitSource(), "synthetic"
