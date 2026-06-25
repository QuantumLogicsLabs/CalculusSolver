"""
Root conftest.py — shared fixtures for all test suites.

Provides:
  reset_solver_singleton — autouse fixture that clears the api._shared
  solver singleton before every test. This prevents solver state from
  leaking between test modules when pytest runs all suites in one process.

  Explicitly removes GROQ_API_KEY from the environment so that
  get_solver() always falls through to FallbackSolver in CI and locally.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def reset_solver_singleton():
    """
    Clear the solver singleton and remove GROQ_API_KEY before every test.

    autouse=True means this runs automatically for every test in every
    test file, with no need to import or reference it explicitly.

    Yields control to the test, then does nothing on teardown — the
    singleton stays reset for the next test because the next invocation
    of this fixture will reset it again before that test runs.
    """
    # Remove Groq key first so that any import of api._shared triggered
    # below does not accidentally instantiate GroqSolver
    os.environ.pop("GROQ_API_KEY", None)

    # Import here (not at module level) to avoid circular import issues
    # if conftest is loaded before the package is fully on sys.path
    try:
        import api._shared as _shared
        _shared._solver = None
        _shared._solver_mode = "unloaded"
        _shared._solver_error = None
    except ImportError:
        # api._shared not importable yet (e.g. during unit-only runs
        # where the api package is not needed). Safe to ignore.
        pass

    yield
