# tests/unit/detectors/test_init.py
import pytest
import sys
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import importlib # For mocking import_module

# テスト対象モジュールをインポート
try:
    from src.detectors import (
        register_detector,
        load_detector_from_file,
        get_detector_class,
        create_detector,
        BaseDetector,
        _detector_registry # Accessing internal for testing registration
    )
    from src.utils.exception_utils import DetectorError, ConfigError
except ImportError:
    pytest.skip("Skipping detectors/__init__ tests due to missing src modules", allow_module_level=True)
    # Dummy classes for static analysis
    class BaseDetector: pass
    class DetectorError(Exception): pass
    class ConfigError(Exception): pass
    _detector_registry = {}

# --- Fixtures ---

@pytest.fixture(autouse=True)
def clear_registry_between_tests():
    """Ensures the detector registry is clean before each test."""
    original_registry = _detector_registry.copy()
    _detector_registry.clear()
    yield
    # Restore original (though likely not strictly necessary if tests clean up)
    _detector_registry.clear()
    _detector_registry.update(original_registry)

@pytest.fixture
def mock_base_detector(monkeypatch):
    """Provides a mock BaseDetector class."""
    mock = MagicMock(spec=BaseDetector)
    mock.name = "MockBase"
    # Mock the __subclasses__ if needed for some tests, though direct registration is better
    return mock

# --- Test register_detector ---

def test_register_detector_success():
    """Tests successful registration of a valid detector class."""
    # Create a dummy class inheriting from BaseDetector (or use a mock)
    class DummyDetector(BaseDetector):
        name = "DummySuccess"
        version = "1.0"
        def detect(self, audio_data, sample_rate):
            pass # Dummy implementation

    register_detector(DummyDetector)

    assert "DummySuccess" in _detector_registry
    assert _detector_registry["DummySuccess"] == DummyDetector

def test_register_detector_not_subclass():
    """Tests registration attempt with a class not inheriting from BaseDetector."""
    class NotADetector:
        name = "InvalidDetector"
        version = "1.0"

    # Depending on implementation, this might raise TypeError or log a warning.
    # Assuming TypeError for stricter checking.
    with pytest.raises(TypeError, match="must inherit from BaseDetector"):
        register_detector(NotADetector)

    assert "InvalidDetector" not in _detector_registry

def test_register_detector_duplicate_name():
    """Tests registering a detector with a name that already exists."""
    class DetectorA(BaseDetector):
        name = "DuplicateName"
        version = "1.0"
        def detect(self, audio_data, sample_rate): pass

    class DetectorB(BaseDetector):
        name = "DuplicateName" # Same name
        version = "2.0"
        def detect(self, audio_data, sample_rate): pass

    register_detector(DetectorA) # First registration should succeed
    assert "DuplicateName" in _detector_registry

    # Second registration with the same name should fail
    with pytest.raises(ValueError, match="Detector name 'DuplicateName' already registered"):
        register_detector(DetectorB)

    # Ensure the registry still holds the first one
    assert _detector_registry["DuplicateName"] == DetectorA

# --- Test load_detector_from_file ---

@pytest.fixture
def mock_importlib():
    """Mocks importlib.import_module."""
    with patch('importlib.import_module') as mock:
        yield mock

@pytest.fixture
def mock_sys_path():
    """Temporarily adds a directory to sys.path for mocking imports."""
    test_dir = Path("/fake/detector/path")
    original_path = sys.path[:]
    sys.path.insert(0, str(test_dir.parent))
    yield test_dir
    # Cleanup sys.path
    sys.path = original_path

def test_load_detector_from_file_success(mock_importlib, mock_sys_path):
    """Tests successfully loading a detector class from a file."""
    detector_file = mock_sys_path / "my_detector.py"
    module_name = "my_detector" # Simplified module name based on path mock

    # Define the dummy detector class to be found in the mocked module
    class LoadedDetector(BaseDetector):
        name = "LoadedSuccess"
        version = "1.0"
        def detect(self, audio_data, sample_rate): pass

    # Mock the module that import_module will return
    mock_module = MagicMock()
    # Simulate dir() finding the class
    mock_module.LoadedDetector = LoadedDetector
    mock_importlib.return_value = mock_module

    # Mock Path.exists to return True
    with patch.object(Path, 'exists', return_value=True):
        # Mock Path.is_file to return True
        with patch.object(Path, 'is_file', return_value=True):
            loaded_classes = load_detector_from_file(detector_file)

    # Assertions
    mock_importlib.assert_called_once_with(module_name)
    assert "LoadedSuccess" in _detector_registry
    assert _detector_registry["LoadedSuccess"] == LoadedDetector
    assert loaded_classes == [LoadedDetector] # Function should return the loaded class(es)

def test_load_detector_from_file_not_found():
    """Tests loading when the specified Python file does not exist."""
    detector_file = Path("/non/existent/detector.py")

    # Mock Path.exists to return False
    with patch.object(Path, 'exists', return_value=False):
        with pytest.raises(DetectorError, match="Detector file not found"):
            load_detector_from_file(detector_file)

def test_load_detector_from_file_import_error(mock_importlib, mock_sys_path):
    """Tests loading when the Python file exists but causes an ImportError."""
    detector_file = mock_sys_path / "broken_detector.py"
    module_name = "broken_detector"

    # Configure import_module to raise ImportError
    mock_importlib.side_effect = ImportError("Module level error")

    # Mock Path.exists and is_file to return True
    with patch.object(Path, 'exists', return_value=True):
        with patch.object(Path, 'is_file', return_value=True):
            with pytest.raises(DetectorError, match="Failed to import detector module"):
                load_detector_from_file(detector_file)

    mock_importlib.assert_called_once_with(module_name)
    assert not _detector_registry # Registry should be empty

def test_load_detector_from_file_no_detector_class(mock_importlib, mock_sys_path):
    """Tests loading when the file is imported but contains no BaseDetector subclass."""
    detector_file = mock_sys_path / "nodetector_detector.py"
    module_name = "nodetector_detector"

    # Mock the module without a BaseDetector subclass
    mock_module = MagicMock()
    class SomeOtherClass: pass
    mock_module.SomeOtherClass = SomeOtherClass
    mock_importlib.return_value = mock_module

    # Mock Path.exists and is_file
    with patch.object(Path, 'exists', return_value=True):
        with patch.object(Path, 'is_file', return_value=True):
            # Expecting an error because no detector was found/registered
            with pytest.raises(DetectorError, match="No BaseDetector subclass found"):
                 load_detector_from_file(detector_file)
            # Or, if it just returns an empty list without error:
            # loaded_classes = load_detector_from_file(detector_file)
            # assert loaded_classes == []

    mock_importlib.assert_called_once_with(module_name)
    assert not _detector_registry # Registry should remain empty

# --- Test get_detector_class ---

def test_get_detector_class_registered():
    """Tests getting a class that was registered directly."""
    class MyRegisteredDetector(BaseDetector):
        name = "Registered"
        version = "1.0"
        def detect(self, *args): pass

    register_detector(MyRegisteredDetector)

    retrieved_class = get_detector_class("Registered")
    assert retrieved_class == MyRegisteredDetector

def test_get_detector_class_not_found():
    """Tests getting a class name that is not registered."""
    # Ensure the registry is empty or does not contain 'NotFound'
    assert "NotFound" not in _detector_registry

    with pytest.raises(DetectorError, match="Detector class 'NotFound' not found"):
        get_detector_class("NotFound")

# --- Test create_detector ---

def test_create_detector_success():
    """Tests successfully creating a detector instance."""
    # Mock the class returned by get_detector_class
    mock_class = MagicMock(spec=BaseDetector)
    # Mock the instance returned by calling the class
    mock_instance = MagicMock(spec=BaseDetector)
    mock_class.return_value = mock_instance # Calling the class returns the instance

    detector_name = "MyDetector"
    params = {"param1": 10, "param2": "value"}

    with patch('src.detectors.get_detector_class', return_value=mock_class) as mock_get_class:
        instance = create_detector(detector_name, params)

    mock_get_class.assert_called_once_with(detector_name)
    # Check if the class was instantiated with the params
    mock_class.assert_called_once_with(**params)
    assert instance == mock_instance # Ensure the created instance is returned

def test_create_detector_init_error():
    """Tests creating a detector when its __init__ raises an error."""
    # Mock the class whose __init__ will raise an error
    mock_class = MagicMock(spec=BaseDetector)
    init_error = TypeError("Invalid parameter type")
    mock_class.side_effect = init_error # Calling the class raises the error

    detector_name = "ErrorDetector"
    params = {"invalid_param": True}

    with patch('src.detectors.get_detector_class', return_value=mock_class) as mock_get_class:
        with pytest.raises(DetectorError, match=f"Failed to instantiate detector class {detector_name}") as excinfo:
            create_detector(detector_name, params)

    # Optionally check the original exception type if needed
    # assert isinstance(excinfo.value.__cause__, TypeError)

    mock_get_class.assert_called_once_with(detector_name)
    # Check that instantiation was attempted
    mock_class.assert_called_once_with(**params)

def test_create_detector_class_not_found():
    """Tests creating a detector when the class name is not found."""
    detector_name = "NonExistentDetector"
    params = {}
    error_msg = f"Detector class '{detector_name}' not found"

    # Mock get_detector_class to raise the error
    with patch('src.detectors.get_detector_class', side_effect=DetectorError(error_msg)) as mock_get_class:
        with pytest.raises(DetectorError, match=error_msg):
            create_detector(detector_name, params)

    mock_get_class.assert_called_once_with(detector_name) 