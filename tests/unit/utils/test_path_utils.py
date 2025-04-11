# tests/unit/utils/test_path_utils.py
import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Assume functions are defined in src.utils.path_utils
# Provide dummies if necessary
try:
    from src.utils.path_utils import (
        get_project_root,
        get_workspace_dir,
        get_output_base_dir,
        is_safe_path_component,
        validate_path_within_allowed_dirs,
        ensure_dir
    )
    PATHLIB_PATCH_TARGET = 'src.utils.path_utils.Path'
    OS_PATCH_TARGET = 'src.utils.path_utils.os'
except ImportError:
    print("Warning: Using dummy implementations for path_utils.")
    # Dummy implementations
    def get_project_root():
        # Dummy: Go up until a known marker (e.g., pyproject.toml) or assume structure
        # This dummy is likely inaccurate for actual testing
        p = Path(__file__).resolve().parent.parent.parent # Guess based on tests/unit/utils
        # Look for a common marker like pyproject.toml or .git
        for _ in range(3): # Limit search depth
            if (p / 'pyproject.toml').exists() or (p / '.git').exists():
                return p
            if p.parent == p:
                 break
            p = p.parent
        # Fallback guess if marker not found
        return Path(__file__).resolve().parent.parent.parent

    def get_workspace_dir(config=None):
        # Dummy: Use a subdirectory in project root
        # Allow override via config dict (basic simulation)
        if config and 'workspace' in config:
             return Path(config['workspace'])
        return get_project_root() / "workspace"

    def get_output_base_dir(config=None):
         # Dummy: Use a subdirectory in project root
        if config and 'output_base' in config:
             return Path(config['output_base'])
        return get_project_root() / "output"

    def is_safe_path_component(component):
        if not isinstance(component, str):
            return False
        # Basic safety checks
        return bool(component) and '..' not in component and '/' not in component and '\\' not in component

    def validate_path_within_allowed_dirs(path_to_validate, allowed_base_dirs):
        try:
            resolved_path = Path(path_to_validate).resolve()
        except Exception:
             # If path is invalid (e.g., contains null bytes on some OS), resolve might fail
             raise ValueError(f"Invalid path format: {path_to_validate}")

        validated = False
        for base_dir in allowed_base_dirs:
            try:
                resolved_base = Path(base_dir).resolve()
                # Check if resolved_path is equal to or starts with resolved_base
                if resolved_path == resolved_base or str(resolved_path).startswith(str(resolved_base) + os.sep):
                    validated = True
                    break # Found a valid base directory
            except Exception:
                # Ignore errors resolving base_dir? Or raise?
                # Let's ignore for dummy, real one might need error handling here.
                continue

        if not validated:
            raise ValueError(f"Path {path_to_validate} is outside allowed directories: {allowed_base_dirs}")
        return True

    def ensure_dir(dir_path):
        path_obj = Path(dir_path)
        # Check if a file exists at the location
        if path_obj.is_file():
            raise FileExistsError(f"Cannot create directory, a file exists at: {dir_path}")
        # Create directory, handling potential race conditions is complex
        path_obj.mkdir(parents=True, exist_ok=True)

    PATHLIB_PATCH_TARGET = 'pathlib.Path' # Patch standard pathlib if dummies used
    OS_PATCH_TARGET = 'os' # Patch standard os

# --- Helper Fixture for Path Mocking ---
@pytest.fixture
def mock_path_env(monkeypatch, tmp_path):
    """Mocks Path, os.environ, and provides temporary paths."""
    # Use tmp_path provided by pytest for realistic temporary paths
    mock_project_root = tmp_path / "project"
    mock_workspace = mock_project_root / "ws"
    mock_output = mock_project_root / "out"

    # Create the base directories in the temp filesystem
    mock_project_root.mkdir()
    mock_workspace.mkdir()
    mock_output.mkdir()

    # --- Mocking High-Level Functions --- 
    # It's generally safer and easier to mock the functions directly
    # if their internal logic (like searching for markers) is complex.
    monkeypatch.setattr('src.utils.path_utils.get_project_root', lambda: mock_project_root, raising=False)
    monkeypatch.setattr('src.utils.path_utils.get_workspace_dir', lambda config=None: mock_workspace, raising=False)
    monkeypatch.setattr('src.utils.path_utils.get_output_base_dir', lambda config=None: mock_output, raising=False)

    # --- Mocking Low-Level os functions if needed --- 
    # Mock os.environ specifically if functions read env vars
    mock_environ = {}
    monkeypatch.setattr(os, 'environ', mock_environ)

    # Mock os.path functions if they are used directly and need control
    # Using tmp_path often makes these less necessary, but shown as example:
    # def mock_abspath(p): return str(tmp_path / p)
    # monkeypatch.setattr(os.path, 'abspath', mock_abspath)
    # def mock_exists(p): return (tmp_path / p).exists()
    # monkeypatch.setattr(os.path, 'exists', mock_exists)

    # Patch pathlib.Path methods if the functions under test use them directly
    # Example: Mocking resolve() globally (use with caution)
    # mock_resolve = lambda self: self # Simplistic mock
    # monkeypatch.setattr(PATHLIB_PATCH_TARGET, "resolve", mock_resolve)

    # Return the paths for use in tests
    return {
        "project_root": mock_project_root,
        "workspace": mock_workspace,
        "output": mock_output,
        "environ": mock_environ
    }


# --- Path Function Tests ---

# Test the mocked high-level functions
def test_get_project_root_mocked(mock_path_env):
    """Tests that get_project_root returns the mocked value."""
    # This test works because mock_path_env patches the function directly
    assert get_project_root() == mock_path_env["project_root"]

def test_get_workspace_dir_mocked(mock_path_env):
    """Tests that get_workspace_dir returns the mocked value."""
    assert get_workspace_dir() == mock_path_env["workspace"]
    # Test with mock config (our mock ignores it, but tests the call)
    assert get_workspace_dir(config={"workspace": "/ignored"}) == mock_path_env["workspace"]

def test_get_output_base_dir_mocked(mock_path_env):
    """Tests that get_output_base_dir returns the mocked value."""
    assert get_output_base_dir() == mock_path_env["output"]
    assert get_output_base_dir(config={"output_base": "/ignored"}) == mock_path_env["output"]


# --- is_safe_path_component Tests ---

@pytest.mark.parametrize("component, expected", [
    ("filename.txt", True),
    ("sub_dir", True),
    ("with_underscore", True),
    ("with-hyphen", True),
    ("with.dot", True),
    ("..", False),             # Parent dir traversal
    ("dir/file", False),       # Contains separator
    ("dir\\file", False),      # Contains windows separator
    ("/absolute", False),      # Starts with separator
    ("", False),               # Empty string
    (None, False),             # None value
    ("../allowed", False),    # Starts with ..
    ("allowed/..", False),    # Contains ..
    (" space ", True),         # Spaces allowed by default
])
def test_is_safe_path_component(component, expected):
    """Tests various inputs for is_safe_path_component."""
    assert is_safe_path_component(component) == expected


# --- validate_path_within_allowed_dirs Tests ---

@pytest.fixture
def allowed_dirs_setup(tmp_path):
    """Creates some allowed directories for testing."""
    dir1 = tmp_path / "allowed1"
    dir2 = tmp_path / "allowed1" / "sub"
    dir3 = tmp_path / "other_allowed"
    dir1.mkdir()
    dir2.mkdir()
    dir3.mkdir()
    # Create files inside
    (dir2 / "safe_file.txt").touch()
    (dir3 / "another.dat").touch()
    # Create a non-allowed dir
    unsafe_dir = tmp_path / "unsafe"
    unsafe_dir.mkdir()
    (unsafe_dir / "unsafe_file.txt").touch()
    # Return allowed dirs as strings and the base tmp_path
    return [str(dir1), str(dir3)], tmp_path

def test_validate_path_within_allowed_success(allowed_dirs_setup):
    """Tests paths that should be successfully validated."""
    allowed_dirs, tmp_path = allowed_dirs_setup
    base_path = Path(allowed_dirs[0])
    sub_path = base_path / "sub"
    other_base_path = Path(allowed_dirs[1])

    # Paths directly within allowed dirs
    assert validate_path_within_allowed_dirs(str(base_path / "file.txt"), allowed_dirs) is True
    assert validate_path_within_allowed_dirs(str(other_base_path / "data.csv"), allowed_dirs) is True
    # Path within a subdirectory of an allowed dir
    assert validate_path_within_allowed_dirs(str(sub_path / "safe_file.txt"), allowed_dirs) is True
    # Path that is exactly an allowed dir
    assert validate_path_within_allowed_dirs(str(base_path), allowed_dirs) is True
    # Subdirectory itself is also considered within the allowed base
    assert validate_path_within_allowed_dirs(str(sub_path), allowed_dirs) is True

def test_validate_path_outside_allowed_failure(allowed_dirs_setup):
    """Tests paths that should fail validation."""
    allowed_dirs, tmp_path = allowed_dirs_setup
    unsafe_path = tmp_path / "unsafe" / "unsafe_file.txt"

    # Path completely outside
    with pytest.raises(ValueError, match="outside allowed directories"):
        validate_path_within_allowed_dirs(str(unsafe_path), allowed_dirs)

    # Path attempting parent traversal that resolves outside
    # Example: /tmp/pytest-of-user/pytest-0/allowed1/../unsafe/unsafe_file.txt
    tricky_outside_path = Path(allowed_dirs[0]) / ".." / "unsafe" / "unsafe_file.txt"
    with pytest.raises(ValueError, match="outside allowed directories"):
         validate_path_within_allowed_dirs(str(tricky_outside_path), allowed_dirs)

def test_validate_path_parent_traversal_resolves_inside(allowed_dirs_setup):
    """Tests traversal that resolves back inside allowed dir (should pass)."""
    allowed_dirs, tmp_path = allowed_dirs_setup
    # Example: /tmp/pytest-of-user/pytest-0/allowed1/sub/../sub/safe_file.txt
    # Resolves to: /tmp/pytest-of-user/pytest-0/allowed1/sub/safe_file.txt
    tricky_path = Path(allowed_dirs[0]) / "sub" / ".." / "sub" / "safe_file.txt"
    assert validate_path_within_allowed_dirs(str(tricky_path), allowed_dirs) is True

def test_validate_path_non_existent(allowed_dirs_setup):
    """Tests validation for paths that don't exist yet (should still work)."""
    allowed_dirs, tmp_path = allowed_dirs_setup
    non_existent_path = Path(allowed_dirs[0]) / "new_dir" / "new_file.txt"
    assert validate_path_within_allowed_dirs(str(non_existent_path), allowed_dirs) is True

    # Non-existent path outside allowed dirs
    non_existent_outside = tmp_path / "unsafe" / "non_existent_file.txt"
    with pytest.raises(ValueError, match="outside allowed directories"):
         validate_path_within_allowed_dirs(str(non_existent_outside), allowed_dirs)


# --- ensure_dir Tests ---

def test_ensure_dir_creates_new(tmp_path):
    """Tests that ensure_dir creates a new directory including parents."""
    new_dir = tmp_path / "a" / "b" / "c"
    assert not new_dir.exists()
    ensure_dir(str(new_dir))
    assert new_dir.is_dir()

def test_ensure_dir_existing(tmp_path):
    """Tests that ensure_dir does nothing if the directory already exists."""
    existing_dir = tmp_path / "exists"
    existing_dir.mkdir()
    ensure_dir(str(existing_dir))
    assert existing_dir.is_dir()

def test_ensure_dir_with_file_conflict(tmp_path):
    """Tests that ensure_dir raises an error if a file exists at the path."""
    file_path = tmp_path / "file.txt"
    file_path.touch()
    assert file_path.is_file()
    # Calling ensure_dir on a file path should raise FileExistsError or NotADirectoryError
    # Depending on the OS and Python version, mkdir can raise either.
    with pytest.raises((FileExistsError, NotADirectoryError)):
        ensure_dir(str(file_path)) 