"""Scaffold smoke test: every package and subpackage imports cleanly.

This guarantees the   0 repo structure is importable before any feature
code lands. Feature-specific tests live in the other test modules.
"""

import importlib

import pytest

MODULES = [
    "chess_robot",
    "chess_robot.chess",
    "chess_robot.data",
    "chess_robot.robot",
    "chess_robot.perception",
    "chess_robot.policies",
    "chess_robot.rl",
    "chess_robot.safety",
    "chess_robot.eval",
    "chess_robot.utils",
]


@pytest.mark.parametrize("module", MODULES)
def test_subpackage_imports(module: str) -> None:
    importlib.import_module(module)


def test_version_exposed() -> None:
    import chess_robot

    assert isinstance(chess_robot.__version__, str)
