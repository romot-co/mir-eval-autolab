import pytest
import os
from pathlib import Path
import json
import numpy as np

# Assuming utils are in src/utils/
from src.utils import path_utils, json_utils

# --- Test path_utils --- #

def test_get_project_root():
    """Test that get_project_root returns a valid directory Path."""
    root_path = path_utils.get_project_root()
    assert isinstance(root_path, Path)
    assert root_path.is_dir()
    # Check for a known file/dir in the project root (e.g., pyproject.toml)
    assert (root_path / "pyproject.toml").exists()

def test_ensure_dir(tmp_path):
    """Test ensure_dir creates a directory and handles existing ones."""
    new_dir = tmp_path / "new_test_dir"
    assert not new_dir.exists()
    path_utils.ensure_dir(new_dir)
    assert new_dir.exists()
    assert new_dir.is_dir()

    # Test with existing directory
    path_utils.ensure_dir(new_dir)
    assert new_dir.exists()

    # Test creating nested directories
    nested_dir = tmp_path / "nested" / "dir"
    assert not nested_dir.exists()
    path_utils.ensure_dir(nested_dir)
    assert nested_dir.exists()
    assert nested_dir.is_dir()

# TODO: Add tests for other path_utils functions (get_workspace_dir etc.)
# These might need mocking environment variables or config files.


# --- Test json_utils --- #

def test_numpy_encoder_basic():
    """Test NumpyEncoder with basic Python types."""
    data = {"a": 1, "b": "hello", "c": [1, 2, 3], "d": True, "e": None}
    encoded = json.dumps(data, cls=json_utils.NumpyEncoder)
    decoded = json.loads(encoded)
    assert decoded == data

def test_numpy_encoder_numpy_types():
    """Test NumpyEncoder with numpy arrays and scalars."""
    data = {
        "int64": np.int64(10),
        "float32": np.float32(3.14),
        "bool": np.bool_(True),
        "array_int": np.array([1, 2, 3]),
        "array_float": np.array([1.1, 2.2, 3.3]),
        "array_mixed": np.array([1, 2.5, True]) # Might convert bool to number
    }
    encoded = json.dumps(data, cls=json_utils.NumpyEncoder)
    decoded = json.loads(encoded)

    assert decoded["int64"] == 10
    assert isinstance(decoded["int64"], int) # Should be standard int
    assert np.isclose(decoded["float32"], 3.14, atol=1e-6)
    assert isinstance(decoded["float32"], float) # Should be standard float
    assert decoded["bool"] is True
    assert isinstance(decoded["bool"], bool) # Should be standard bool

    assert decoded["array_int"] == [1, 2, 3]
    assert isinstance(decoded["array_int"], list)
    assert np.allclose(decoded["array_float"], [1.1, 2.2, 3.3])
    assert isinstance(decoded["array_float"], list)
    # Note: np.array([1, 2.5, True]) -> [1. , 2.5, 1. ] (bool becomes float)
    assert np.allclose(decoded["array_mixed"], [1.0, 2.5, 1.0])
    assert isinstance(decoded["array_mixed"], list)

def test_numpy_encoder_unserializable():
    """Test NumpyEncoder falls back for unserializable types."""
    class Unserializable:
        pass

    data = {"obj": Unserializable()}
    # Default behavior of json.dumps for unsupported types depends on flags
    # The encoder might raise TypeError or convert to string based on default()
    # Let's check if it falls back to the default method which might raise TypeError
    with pytest.raises(TypeError):
        json.dumps(data, cls=json_utils.NumpyEncoder)

    # If the default method was overridden to return str(o):
    # encoded = json.dumps(data, cls=json_utils.NumpyEncoder)
    # decoded = json.loads(encoded)
    # assert isinstance(decoded["obj"], str)

# TODO: Add tests for other utils modules if present 