import pytest
import os
import yaml
import asyncio
import shutil
import logging # Import logging for caplog usage
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import time # Import time for utime tests
from typing import Optional # Add Optional for type hinting

# Assume core functions and ConfigError exist
try:
    from src.mcp_server_logic.core import (\
        ServerConfig, # Assuming config class/model
        load_config,
        cleanup_workspace_files,
        start_cleanup_task
    )
    from src.utils.exception_utils import ConfigError
    # Assume path utils are used for resolving paths
    import src.utils.path_utils as path_utils_module
except ImportError:
    print("Warning: Using dummy implementations for core.py and dependencies.")
    from pydantic import BaseModel, ValidationError # Use pydantic for dummy config model

    class ConfigError(Exception): pass

    class ServerConfig(BaseModel):
        db_path: str = "db/mcp_server.db"
        workspace_base: str = "workspace"
        output_base: str = "output"
        cleanup_interval_minutes: int = 60
        keep_duration_minutes: int = 10080 # 1 week
        log_level: str = "INFO"
        # Add other potential fields

    # Dummy path utils needed by dummy load_config
    class DummyPathUtils:
        @staticmethod
        def get_project_root(): return Path("/dummy/project")
        @staticmethod
        def ensure_dir(p):
             Path(p).mkdir(parents=True, exist_ok=True)

    path_utils_module = DummyPathUtils()


    def load_config(config_path: Optional[str] = None) -> ServerConfig:
        # Dummy load_config logic
        defaults = ServerConfig().dict()
        config_data = defaults.copy() # Use copy to avoid modifying class defaults

        # 1. Load from YAML if path provided
        if config_path:
             if Path(config_path).exists():
                 try:
                     with open(config_path, 'r') as f:
                         yaml_data = yaml.safe_load(f)
                         if isinstance(yaml_data, dict):
                             config_data.update(yaml_data)
                 except Exception as e:
                     print(f"Dummy load_config YAML error: {e}")

        # 2. Override with environment variables
        if 'MCP_DB_PATH' in os.environ: config_data['db_path'] = os.environ['MCP_DB_PATH']
        if 'MCP_WORKSPACE_BASE' in os.environ: config_data['workspace_base'] = os.environ['MCP_WORKSPACE_BASE']
        if 'MCP_OUTPUT_BASE' in os.environ: config_data['output_base'] = os.environ['MCP_OUTPUT_BASE']
        if 'MCP_LOG_LEVEL' in os.environ: config_data['log_level'] = os.environ['MCP_LOG_LEVEL']
        if 'MCP_CLEANUP_INTERVAL_MINUTES' in os.environ:
             try: config_data['cleanup_interval_minutes'] = int(os.environ['MCP_CLEANUP_INTERVAL_MINUTES'])
             except ValueError: pass
        if 'MCP_KEEP_DURATION_MINUTES' in os.environ:
             try: config_data['keep_duration_minutes'] = int(os.environ['MCP_KEEP_DURATION_MINUTES'])
             except ValueError: pass

        # Resolve paths
        root = path_utils_module.get_project_root()
        try:
             db_p = config_data.get('db_path')
             ws_p = config_data.get('workspace_base')
             out_p = config_data.get('output_base')

             if db_p and not Path(db_p).is_absolute(): config_data['db_path'] = str(root / db_p)
             if ws_p and not Path(ws_p).is_absolute(): config_data['workspace_base'] = str(root / ws_p)
             if out_p and not Path(out_p).is_absolute(): config_data['output_base'] = str(root / out_p)

             if config_data.get('workspace_base'): path_utils_module.ensure_dir(Path(config_data['workspace_base']).parent)
             if config_data.get('output_base'): path_utils_module.ensure_dir(Path(config_data['output_base']).parent)
             if config_data.get('db_path'): path_utils_module.ensure_dir(Path(config_data['db_path']).parent)

             return ServerConfig(**config_data)
        except ValidationError as e:
            # Catch Pydantic validation error specifically if using Pydantic model
             raise ConfigError(f"Invalid configuration: {e}") from e
        except Exception as e:
            raise ConfigError(f"Error processing config: {e}") from e


    async def cleanup_workspace_files(config: ServerConfig):
        # Dummy cleanup logic with logging simulation
        logger = logging.getLogger('src.mcp_server_logic.core') # Assume this logger name
        logger.info(f"Dummy cleanup called for workspace: {config.workspace_base}")
        if config.cleanup_interval_minutes <= 0 or config.keep_duration_minutes <= 0:
            logger.info("Cleanup disabled by config.")
            return

        workspace = Path(config.workspace_base)
        now = time.time()
        cutoff_time = now - (config.keep_duration_minutes * 60)
        logger.debug(f"Cleanup cutoff time: {cutoff_time}")

        if not workspace.exists():
             logger.warning("Workspace directory does not exist.")
             return

        deleted_files = []
        deleted_dirs = []
        errors = []
        for item in workspace.iterdir():
            try:
                item_stat = item.stat()
                item_mtime = item_stat.st_mtime
                logger.debug(f"Checking {item.name}, mtime: {item_mtime}")
                if item_mtime < cutoff_time:
                    if item.name == '.gitkeep':
                         logger.debug(f"Skipping protected file: {item.name}")
                         continue

                    if item.is_file():
                         logger.info(f"Dummy deleting file: {item}")
                         # In real test, mock os.remove(item)
                         # os.remove(item)
                         deleted_files.append(str(item))
                    elif item.is_dir():
                         logger.info(f"Dummy deleting dir: {item}")
                         # In real test, mock shutil.rmtree(item)
                         # shutil.rmtree(item)
                         deleted_dirs.append(str(item))
            except Exception as e:
                 logger.error(f"Error processing {item} during cleanup: {e}", exc_info=True)
                 errors.append(str(item))
        # Return info for testing mocks
        return {"deleted_files": deleted_files, "deleted_dirs": deleted_dirs, "errors": errors}


    async def start_cleanup_task(config: ServerConfig):
        # Dummy task starter
        logger = logging.getLogger('src.mcp_server_logic.core') # Assume logger
        if config.cleanup_interval_minutes > 0:
            logger.info("Dummy starting cleanup task.")
            mock_task = MagicMock(spec=asyncio.Task)
            mock_task.cancel = MagicMock()
            return mock_task
        else:
            logger.info("Cleanup task not started (interval <= 0).")
            return None


# --- Fixtures ---
@pytest.fixture
def mock_env(monkeypatch):
    """Clears relevant environment variables before test and restores after."""
    env_keys = ["MCP_DB_PATH", "MCP_WORKSPACE_BASE", "MCP_OUTPUT_BASE", "MCP_LOG_LEVEL",
                "MCP_CLEANUP_INTERVAL_MINUTES", "MCP_KEEP_DURATION_MINUTES"]
    original_env = {k: os.environ.get(k) for k in env_keys}
    for k in env_keys:
        if k in os.environ:
            monkeypatch.delenv(k, raising=False)
    yield monkeypatch
    for k, v in original_env.items():
        if v is None:
            if k in os.environ: monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)

@pytest.fixture
def mock_project_paths(monkeypatch, tmp_path):
    """Mocks path_utils functions to use tmp_path."""
    dummy_root = tmp_path / "project_root"
    dummy_root.mkdir()
    monkeypatch.setattr(path_utils_module, 'get_project_root', lambda: dummy_root)
    def mock_ensure(p):
        Path(p).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(path_utils_module, 'ensure_dir', mock_ensure, raising=False)
    return dummy_root


# --- load_config Tests ---

def test_load_config_defaults(mock_env, mock_project_paths):
    """Test loading default configuration when no file or env vars exist."""
    with patch('pathlib.Path.exists') as mock_exists, \
         patch("builtins.open", mock_open()) as mock_file:
         mock_exists.return_value = False
         config = load_config()

    assert isinstance(config, ServerConfig)
    default_model = ServerConfig()
    expected_db = mock_project_paths / default_model.db_path
    expected_ws = mock_project_paths / default_model.workspace_base
    expected_out = mock_project_paths / default_model.output_base
    assert Path(config.db_path).resolve() == expected_db.resolve()
    assert Path(config.workspace_base).resolve() == expected_ws.resolve()
    assert Path(config.output_base).resolve() == expected_out.resolve()
    assert config.log_level == default_model.log_level
    assert config.cleanup_interval_minutes == default_model.cleanup_interval_minutes

def test_load_config_from_yaml(mock_env, mock_project_paths, tmp_path):
    """Test loading configuration primarily from a YAML file."""
    config_file = tmp_path / "test_config.yaml"
    yaml_content = {
        "db_path": "/absolute/db/path.db",
        "workspace_base": "custom_ws",
        "log_level": "DEBUG",
        "cleanup_interval_minutes": 0
    }
    config_file.write_text(yaml.dump(yaml_content))

    original_exists = Path.exists
    def mock_exists(self):
         if str(self) == str(config_file): return True
         # Need to handle other exists checks, e.g., inside path_utils resolve/ensure?
         # For simplicity, assume other checks are not critical or mocked elsewhere.
         # A safer mock might check self.is_absolute() etc.
         return False # Default to False for other paths

    # Patch ensure_dir within the scope of this test to avoid side effects
    with patch('pathlib.Path.exists', side_effect=mock_exists, autospec=True), \
         patch.object(path_utils_module, 'ensure_dir', MagicMock()):
         config = load_config(str(config_file))

    assert config.db_path == "/absolute/db/path.db"
    expected_ws = mock_project_paths / "custom_ws"
    assert Path(config.workspace_base).resolve() == expected_ws.resolve()
    assert config.log_level == "DEBUG"
    assert config.cleanup_interval_minutes == 0
    assert Path(config.output_base).name == ServerConfig().output_base

@patch('yaml.safe_load')
def test_load_config_yaml_error(mock_safe_load, mock_env, mock_project_paths, tmp_path):
    """Test that YAML loading errors are handled (falls back to defaults)."""
    mock_safe_load.side_effect = yaml.YAMLError("Bad YAML")
    config_file = tmp_path / "bad_config.yaml"
    config_file.touch()

    original_exists = Path.exists
    def mock_exists(self):
         if str(self) == str(config_file): return True
         return False

    with patch('pathlib.Path.exists', side_effect=mock_exists, autospec=True), \
         patch.object(path_utils_module, 'ensure_dir', MagicMock()):
        config = load_config(str(config_file))
        # Dummy implementation prints error, check defaults
        assert Path(config.db_path).name == ServerConfig().db_path
        assert config.log_level == ServerConfig().log_level

def test_load_config_env_override(mock_env, mock_project_paths, tmp_path):
    """Test environment variables overriding YAML and defaults."""
    config_file = tmp_path / "test_config.yaml"
    yaml_content = {
        "db_path": "yaml/db.db",
        "workspace_base": "yaml_ws",
        "log_level": "WARNING",
        "cleanup_interval_minutes": 120
    }
    config_file.write_text(yaml.dump(yaml_content))

    mock_env.setenv("MCP_DB_PATH", "/env/db/override.db")
    mock_env.setenv("MCP_LOG_LEVEL", "ERROR")
    mock_env.setenv("MCP_CLEANUP_INTERVAL_MINUTES", "30")

    original_exists = Path.exists
    def mock_exists(self):
         if str(self) == str(config_file): return True
         return False

    with patch('pathlib.Path.exists', side_effect=mock_exists, autospec=True), \
         patch.object(path_utils_module, 'ensure_dir', MagicMock()):
        config = load_config(str(config_file))

    assert config.db_path == "/env/db/override.db"
    expected_ws = mock_project_paths / "yaml_ws"
    assert Path(config.workspace_base).resolve() == expected_ws.resolve()
    assert config.log_level == "ERROR"
    assert config.cleanup_interval_minutes == 30
    expected_out = mock_project_paths / ServerConfig().output_base
    assert Path(config.output_base).resolve() == expected_out.resolve()

def test_load_config_path_resolution(mock_env, mock_project_paths):
    """Test that relative paths in defaults are resolved correctly."""
    with patch('pathlib.Path.exists') as mock_exists, \
         patch.object(path_utils_module, 'ensure_dir', MagicMock()):
         mock_exists.return_value = False
         config = load_config()

    default_model = ServerConfig()
    expected_db_parent = (mock_project_paths / Path(default_model.db_path)).parent
    expected_ws_parent = (mock_project_paths / Path(default_model.workspace_base)).parent
    expected_out_parent = (mock_project_paths / Path(default_model.output_base)).parent

    assert Path(config.db_path).is_absolute()
    assert Path(config.db_path).parent.resolve() == expected_db_parent.resolve()
    assert Path(config.workspace_base).is_absolute()
    # workspace_base default is relative, parent should be project root
    assert Path(config.workspace_base).parent.resolve() == mock_project_paths.resolve()
    assert Path(config.output_base).is_absolute()
    # output_base default is relative, parent should be project root
    assert Path(config.output_base).parent.resolve() == mock_project_paths.resolve()

def test_load_config_invalid_data_raises_error(mock_env, mock_project_paths):
    """Test that ConfigError (or Pydantic's ValidationError) is raised for invalid final data."""
    # Example: Set cleanup_interval_minutes to a non-integer via env var
    mock_env.setenv("MCP_CLEANUP_INTERVAL_MINUTES", "not-an-int")

    with patch('pathlib.Path.exists') as mock_exists, \
         patch.object(path_utils_module, 'ensure_dir', MagicMock()):
        mock_exists.return_value = False # No config file
        # Pydantic validation should fail when creating ServerConfig instance
        with pytest.raises((ConfigError, ValidationError)): 
            load_config()


# --- cleanup_workspace_files Tests ---
pytestmark = pytest.mark.asyncio

async def test_cleanup_disabled(tmp_path):
    """Test that cleanup doesn't run if interval is zero or negative."""
    ws_path = tmp_path / "workspace"
    ws_path.mkdir()
    old_file = ws_path / "old_file.txt"
    old_file.touch()
    very_old_time = time.time() - (15 * 24 * 60 * 60)
    os.utime(str(old_file), (very_old_time, very_old_time))

    config_zero = ServerConfig(workspace_base=str(ws_path), cleanup_interval_minutes=0, keep_duration_minutes=1)
    config_neg = ServerConfig(workspace_base=str(ws_path), cleanup_interval_minutes=-10, keep_duration_minutes=1)

    with patch('shutil.rmtree') as mock_rmtree, \
         patch('os.remove') as mock_remove:
        await cleanup_workspace_files(config_zero)
        mock_rmtree.assert_not_called()
        mock_remove.assert_not_called()

        mock_rmtree.reset_mock()
        mock_remove.reset_mock()

        await cleanup_workspace_files(config_neg)
        mock_rmtree.assert_not_called()
        mock_remove.assert_not_called()

    assert old_file.exists()

async def test_cleanup_removes_old_items(tmp_path):
    """Test that old files/dirs are removed, respecting .gitkeep."""
    ws_path = tmp_path / "workspace"
    ws_path.mkdir()
    keep_minutes = 60
    now = time.time()
    old_time = now - (keep_minutes * 60 * 2)
    new_time = now - (keep_minutes * 60 / 2)

    old_file = ws_path / "old_file.txt"
    old_dir = ws_path / "old_dir"
    new_file = ws_path / "new_file.txt"
    new_dir = ws_path / "new_dir"
    gitkeep = ws_path / ".gitkeep"

    old_file.touch(); os.utime(str(old_file), (old_time, old_time))
    old_dir.mkdir(); os.utime(str(old_dir), (old_time, old_time))
    (old_dir / "inner.txt").touch(); os.utime(str(old_dir / "inner.txt"), (old_time, old_time))
    new_file.touch(); os.utime(str(new_file), (new_time, new_time))
    new_dir.mkdir(); os.utime(str(new_dir), (new_time, new_time))
    gitkeep.touch(); os.utime(str(gitkeep), (old_time, old_time))

    config = ServerConfig(workspace_base=str(ws_path), cleanup_interval_minutes=1, keep_duration_minutes=keep_minutes)

    with patch('shutil.rmtree') as mock_rmtree, \
         patch('os.remove') as mock_remove:
        await cleanup_workspace_files(config)

        mock_remove.assert_any_call(str(old_file))
        mock_rmtree.assert_any_call(str(old_dir))

        calls_args_list = [call.args[0] for call in mock_remove.call_args_list]
        calls_args_list.extend([call.args[0] for call in mock_rmtree.call_args_list])

        assert str(new_file) not in calls_args_list
        assert str(new_dir) not in calls_args_list
        assert str(gitkeep) not in calls_args_list

async def test_cleanup_handles_errors(tmp_path, caplog):
    """Test that cleanup logs errors but continues if deletion fails."""
    ws_path = tmp_path / "workspace"
    ws_path.mkdir()
    keep_minutes = 5
    old_time = time.time() - (keep_minutes * 60 * 2)

    old_file = ws_path / "old_file.txt"
    old_dir = ws_path / "old_dir"
    another_old_file = ws_path / "another_old.txt"

    old_file.touch(); os.utime(str(old_file), (old_time, old_time))
    old_dir.mkdir(); os.utime(str(old_dir), (old_time, old_time))
    another_old_file.touch(); os.utime(str(another_old_file), (old_time, old_time))

    config = ServerConfig(workspace_base=str(ws_path), cleanup_interval_minutes=1, keep_duration_minutes=keep_minutes)

    with patch('os.remove') as mock_remove, \
         patch('shutil.rmtree') as mock_rmtree:

        def rmtree_side_effect(path, ignore_errors=False, onerror=None):
            if Path(path) == old_dir:
                raise OSError("Permission denied")
        mock_rmtree.side_effect = rmtree_side_effect

        logger_name_to_capture = 'src.mcp_server_logic.core'
        # Ensure the logger exists and has a handler for caplog if using dummy
        logging.getLogger(logger_name_to_capture).addHandler(logging.NullHandler())
        with caplog.at_level(logging.ERROR, logger=logger_name_to_capture):
            await cleanup_workspace_files(config)

        mock_remove.assert_any_call(str(old_file))
        mock_remove.assert_any_call(str(another_old_file))
        mock_rmtree.assert_any_call(str(old_dir))

        # Check logs captured by caplog
        assert any(f"Error processing {old_dir}" in record.message and "Permission denied" in record.message 
                   for record in caplog.records if record.name == logger_name_to_capture)


# --- start_cleanup_task Tests ---

@patch('asyncio.create_task')
async def test_start_cleanup_task_creates_task_when_enabled(mock_create_task):
    """Test that a task is created when cleanup interval is positive."""
    config = ServerConfig(cleanup_interval_minutes=10)
    mock_task = MagicMock(spec=asyncio.Task)
    mock_create_task.return_value = mock_task

    returned_task = await start_cleanup_task(config)

    mock_create_task.assert_called_once()
    assert asyncio.iscoroutine(mock_create_task.call_args[0][0])
    assert returned_task is mock_task

@patch('asyncio.create_task')
async def test_start_cleanup_task_no_task_when_disabled(mock_create_task):
    """Test that no task is created when cleanup interval is zero or negative."""
    config_zero = ServerConfig(cleanup_interval_minutes=0)
    task_zero = await start_cleanup_task(config_zero)
    mock_create_task.assert_not_called()
    assert task_zero is None

    mock_create_task.reset_mock()

    config_neg = ServerConfig(cleanup_interval_minutes=-1)
    task_neg = await start_cleanup_task(config_neg)
    mock_create_task.assert_not_called()
    assert task_neg is None
