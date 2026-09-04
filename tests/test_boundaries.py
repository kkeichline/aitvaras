"""Architectural invariants, enforced as tests rather than left to discipline.

The claim in the plan is that swapping the replay environment for a broker's
paper API touches ``env/`` and nothing else. That claim decays silently unless
something checks it, so these tests fail the build the day it stops being true.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "aitvaras"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _referenced_symbols(path: Path) -> set[str]:
    """Identifiers the code actually uses -- imports, names, attributes.

    Deliberately AST-based rather than a text scan: a module docstring that
    *mentions* ReplayDataSource to explain the design is documentation, not
    coupling, and a grep-based version of this test would forbid explaining the
    architecture in the code that implements it.
    """
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.rsplit(".", 1)[-1])
    return names


def test_schemas_import_nothing_from_sibling_packages():
    """``schemas`` is the contract. Everything depends on it; it depends on
    nobody. This is what keeps the boundary real rather than aspirational --
    the moment schemas imports from ``env`` or ``rules``, the dependency graph
    has a cycle and the layers stop being separable."""
    offenders: list[str] = []
    for path in (SRC / "schemas").rglob("*.py"):
        for mod in _imports(path):
            if mod.startswith("aitvaras.") and not mod.startswith("aitvaras.schemas"):
                offenders.append(f"{path.name} imports {mod}")
    assert not offenders, f"schemas must not import siblings: {offenders}"


def test_concrete_data_source_is_not_referenced_outside_env():
    """Nothing outside ``env`` may name a concrete implementation.

    If ``rules`` or ``agent`` ever mentions ``ReplayDataSource`` by name, they
    have learned which world they are in, and the phase 4 broker swap becomes a
    rewrite instead of a config change.
    """
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.is_relative_to(SRC / "env"):
            continue
        used = _referenced_symbols(path)
        for symbol in ("ReplayDataSource", "LeakyReplaySource"):
            if symbol in used:
                offenders.append(f"{path.relative_to(SRC)} references {symbol}")
    assert not offenders, f"concrete source leaked out of env/: {offenders}"


def test_leaky_source_is_confined_to_the_test_suite():
    """The deliberately-broken twin must never be reachable from a real run."""
    for path in SRC.rglob("*.py"):
        if path.name == "replay.py":
            continue
        assert "LeakyReplaySource" not in _referenced_symbols(path), (
            f"leaky twin reachable from {path}"
        )
