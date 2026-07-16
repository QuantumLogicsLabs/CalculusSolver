"""
Root conftest.py — shared fixtures for all test suites.

Provides:
  reset_solver_singleton — autouse fixture that:
    1. Clears the api._shared solver singleton before every test
    2. Removes GROQ_API_KEY and MODEL_PATH from the environment
    3. Monkeypatches _resolve_model_path() to always return (None, None)

  The monkeypatch on _resolve_model_path() is the critical part of Fix 2.
  Without it, tests that assert solver_mode == "fallback" only pass because
  no checkpoint files happen to exist on disk. If anyone commits a checkpoint
  file to any of the priority locations, those tests would silently start
  hitting the neural path. The monkeypatch makes the fallback path explicit
  and unconditional — tests pass regardless of what files exist on disk.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def reset_solver_singleton(monkeypatch):
    """
    Reset solver singleton, strip credential env vars, and force the
    fallback path for every test via monkeypatching.

    autouse=True means this runs automatically for every test in every
    test file with no need to import or reference it explicitly.

    Uses pytest's built-in monkeypatch fixture so that all patches are
    automatically undone after each test — no manual cleanup needed.

    Why monkeypatch _resolve_model_path:
      After Fix 1, get_solver() calls _resolve_model_path() to find a
      checkpoint before attempting neural load. If any checkpoint file
      exists on disk (or MODEL_PATH is set), _resolve_model_path() returns
      a non-None path and the neural solver is attempted. Patching it to
      always return (None, None) guarantees the neural block is always
      skipped in tests, making fallback-mode assertions unconditionally
      correct rather than accidentally correct due to file absence.

    Why remove MODEL_PATH:
      MODEL_PATH is read by _resolve_model_path(). Even with the
      monkeypatch in place it cannot cause harm, but removing it makes
      the intent explicit — no checkpoint resolution of any kind should
      occur during tests.
    """
    # ── Strip environment variables that influence solver selection ───────────
    os.environ.pop("GROQ_API_KEY", None)
    os.environ.pop("MODEL_PATH", None)

    # ── Reset the singleton ───────────────────────────────────────────────────
    try:
        import api._shared as _shared

        # Reset cached singleton so every test starts with a fresh solver load
        _shared._solver = None
        _shared._solver_mode = "unloaded"
        _shared._solver_error = None
        monkeypatch.setattr(_shared, "_resolve_model_path", lambda: (None, None))

    except ImportError:
        # api._shared not importable (e.g. during unit-only runs where
        # the api package is not on sys.path). Safe to ignore — unit tests
        # do not touch the solver singleton.
        pass

    yield
