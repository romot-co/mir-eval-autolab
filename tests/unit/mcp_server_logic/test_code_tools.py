# tests/unit/mcp_server_logic/test_code_tools.py
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, ANY, call
from pathlib import Path
import subprocess
from concurrent.futures import Executor # For mocking the executor

# テスト対象モジュールと依存モジュールをインポート (存在しない場合はダミーを仮定)
try:
    from src.mcp_server_logic import code_tools
    from src.mcp_server_logic.code_tools import (
        get_detector_path,
        _run_get_code,
        _run_save_code,
        _run_get_code_async,
        _run_save_code_async,
        CodeToolsError, # 仮定
        TOOL_NAME_GET_CODE, # 仮定
        TOOL_NAME_SAVE_CODE # 仮定
    )
    from src.mcp_server_logic.schemas import (
        SessionInfoResponse, # _run_..._async に渡される
        GetCodeInput, GetCodeOutput,
        SaveCodeInput, SaveCodeOutput,
        ErrorInfo
    )
    from src.mcp_server_logic.session_manager import SessionManagerConfig # 設定が必要な場合
    import src.utils.path_utils as path_utils_module # path_utils を使う場合
except ImportError:
    print("Warning: Using dummy implementations for code_tools.py and dependencies.")
    from dataclasses import dataclass, field
    from typing import Dict, Any as TypingAny, Optional, List

    class CodeToolsError(Exception): pass
    TOOL_NAME_GET_CODE = "get_code"
    TOOL_NAME_SAVE_CODE = "save_code"

    @dataclass
    class SessionInfoResponse: # Simplified dummy
        session_id: str
        status: str
        workspace_path: str = "/dummy/workspace/session1" # Assume workspace path is available
        history: List[Dict] = field(default_factory=list)

    # --- Dummy Schemas ---
    @dataclass
    class GetCodeInput: params: Dict[str, TypingAny]
    @dataclass
    class GetCodeOutput: result: Dict[str, TypingAny]
    @dataclass
    class SaveCodeInput: params: Dict[str, TypingAny]
    @dataclass
    class SaveCodeOutput: result: Dict[str, TypingAny]
    @dataclass
    class ErrorInfo: error_type: str; message: str; details: Optional[str] = None

    @dataclass
    class SessionManagerConfig: db_path: str = ":memory:" # Dummy

    # Dummy path utils if needed by get_detector_path
    class DummyPathUtils:
        @staticmethod
        def get_workspace_dir(session_id: str, config): return Path(f"/dummy/workspace/{session_id}")
        @staticmethod
        def get_src_detectors_dir(config=None): return Path("/dummy/src/detectors")
        @staticmethod
        def validate_path_within_allowed_dirs(path, allowed_dirs): pass # Assume valid
    path_utils_module = DummyPathUtils()

    # --- Dummy code_tools functions ---
    def get_detector_path(session_info: SessionInfoResponse, detector_name: str, version: str, config) -> Path:
        # Basic dummy logic assuming workspace structure
        ws_path = path_utils_module.get_workspace_dir(session_info.session_id, config)
        if version == "latest":
             p = ws_path / f"{detector_name}.py"
        elif version == "base":
             p = path_utils_module.get_src_detectors_dir() / f"{detector_name}.py"
        else:
             raise ValueError(f"Invalid version: {version}")

        if not p.exists(): # Need to mock Path.exists() in tests
            raise FileNotFoundError(f"Detector file not found: {p}")
        return p

    def _run_get_code(session_info: SessionInfoResponse, params: Dict[str, TypingAny], config) -> Dict[str, TypingAny]:
        detector_name = params.get("detector_name")
        version = params.get("version", "latest")
        if not detector_name: raise CodeToolsError("Missing detector_name")

        try:
            file_path = get_detector_path(session_info, detector_name, version, config)
            # Simulate file read (mock Path.read_text in tests)
            code_content = file_path.read_text()
            return {"file_path": str(file_path), "code": code_content, "version": version}
        except FileNotFoundError as e:
            raise CodeToolsError(f"Cannot get code: {e}") from e
        except Exception as e:
            raise CodeToolsError(f"Error getting code: {e}") from e


    def _run_save_code(session_info: SessionInfoResponse, params: Dict[str, TypingAny], config, subprocess_mock) -> Dict[str, TypingAny]:
        detector_name = params.get("detector_name")
        code = params.get("code")
        commit_message = params.get("commit_message", f"Update {detector_name}")
        if not detector_name or code is None:
            raise CodeToolsError("Missing detector_name or code")

        try:
            # Only allow saving 'latest' version to workspace
            file_path = get_detector_path(session_info, detector_name, "latest", config)
            ws_path = path_utils_module.get_workspace_dir(session_info.session_id, config)
            path_utils_module.validate_path_within_allowed_dirs(file_path, [ws_path]) # Security check

            # Simulate file write (mock Path.write_text in tests)
            file_path.write_text(code)

            # Simulate git commands (mock subprocess.run)
            git_add_cmd = ["git", "add", str(file_path)]
            git_commit_cmd = ["git", "commit", "-m", commit_message]
            subprocess_mock.run(git_add_cmd, check=True, cwd=ws_path, capture_output=True, text=True)
            subprocess_mock.run(git_commit_cmd, check=True, cwd=ws_path, capture_output=True, text=True)

            return {"file_path": str(file_path), "version": "latest", "commit_message": commit_message}
        except FileNotFoundError as e:
            # This might occur if get_detector_path uses exists check for 'latest' which fails initially
             raise CodeToolsError(f"Cannot save code, path issue: {e}") from e
        except subprocess.CalledProcessError as e:
            raise CodeToolsError(f"Git command failed: {e.stderr}") from e
        except Exception as e:
            raise CodeToolsError(f"Error saving code: {e}") from e

    # Dummy async wrappers
    async def _run_async_wrapper(func, session_info, tool_input, tool_output_class, event_name, executor_mock, session_manager_mock, config, *args):
         tool_params = tool_input.params
         event_type = ""
         output = None
         try:
             # Simulate running sync function in executor
             loop = asyncio.get_running_loop()
             result_dict = await loop.run_in_executor(executor_mock, func, session_info, tool_params, config, *args)
             output = tool_output_class(result=result_dict)
             event_type = f"{event_name}_complete"
             details = {"input": tool_params, "output": result_dict}
         except Exception as e:
             error_info = ErrorInfo(error_type=type(e).__name__, message=str(e))
             event_type = f"{event_name}_failed"
             details = {"input": tool_params, "error": error_info.__dict__}
             # In real code, would likely re-raise or return an error state
             # For dummy, just record history and maybe raise
             await session_manager_mock.add_session_history(session_id=session_info.session_id, event_type=event_type, details=details, config=config, db_utils_mock=ANY, misc_utils_mock=ANY, validate_func_mock=ANY)
             raise CodeToolsError(f"Task {event_name} failed: {e}") from e # Re-raise wrapped error

         await session_manager_mock.add_session_history(session_id=session_info.session_id, event_type=event_type, details=details, config=config, db_utils_mock=ANY, misc_utils_mock=ANY, validate_func_mock=ANY)
         return output

    async def _run_get_code_async(session_info: SessionInfoResponse, tool_input: GetCodeInput, executor_mock, session_manager_mock, config):
        # Need to pass subprocess_mock=None for get_code dummy call signature
        # The executor_mock, session_manager_mock, config are the *args for the wrapper
        return await _run_async_wrapper(_run_get_code, session_info, tool_input, GetCodeOutput, TOOL_NAME_GET_CODE, executor_mock, session_manager_mock, config)

    async def _run_save_code_async(session_info: SessionInfoResponse, tool_input: SaveCodeInput, executor_mock, session_manager_mock, config, subprocess_mock):
        # Need to pass the subprocess_mock to the sync function via *args
        return await _run_async_wrapper(_run_save_code, session_info, tool_input, SaveCodeOutput, TOOL_NAME_SAVE_CODE, executor_mock, session_manager_mock, config, subprocess_mock)


# --- Fixtures ---
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_executor():
    """Mocks concurrent.futures.Executor."""
    # Simple mock that runs the function immediately in the current thread
    mock = MagicMock(spec=Executor)
    def run_sync(func, *args, **kwargs):
        # In a real executor mock, you might use a thread pool or similar,
        # but for unit tests, immediate execution is often sufficient.
        # Note: This bypasses actual async execution benefits/complexities.
        try:
             return func(*args, **kwargs)
        except Exception as e:
            # Simulate how run_in_executor might propagate exceptions
             raise e # Re-raise the exact exception

    mock.submit = MagicMock(side_effect=lambda func, *args, **kwargs: run_sync(func, *args, **kwargs)) # Mock submit if used
    # Mock the behavior expected by loop.run_in_executor
    # loop.run_in_executor typically returns a Future, but for simplicity,
    # we'll mock it to directly call our synchronous simulation.
    # The patch target for run_in_executor is tricky, patching within test is safer.
    return mock


@pytest.fixture
def mock_session_manager_for_code():
    """Mocks session_manager functions needed by code_tools."""
    mock = MagicMock()
    mock.add_session_history = AsyncMock()
    return mock

@pytest.fixture
def mock_config_for_code() -> SessionManagerConfig:
     """Provides dummy config potentially needed."""
     # Use the dummy SessionManagerConfig if imported
     return SessionManagerConfig()

@pytest.fixture
def mock_subprocess():
     """Mocks subprocess.run."""
     mock = MagicMock(spec=subprocess)
     # Default successful run
     mock.run.return_value = MagicMock(spec=subprocess.CompletedProcess, returncode=0, stdout="", stderr="")
     return mock


@pytest.fixture
def dummy_session_info_for_code(tmp_path) -> SessionInfoResponse:
     """Provides a basic SessionInfoResponse object with a real temp workspace path."""
     session_id = "code_test_sid"
     ws_path = tmp_path / "workspace" / session_id
     ws_path.mkdir(parents=True, exist_ok=True)
     # Use the dummy SessionInfoResponse if imported
     return SessionInfoResponse(session_id=session_id, status="running", workspace_path=str(ws_path))

@pytest.fixture
def mock_path_utils():
     """Mocks the path_utils module functions."""
     mock = MagicMock(spec=path_utils_module) # Use spec of imported module
     mock.get_workspace_dir = MagicMock(side_effect=lambda sid, cfg: Path(f"/mock_ws/{sid}")) # Basic mock
     mock.get_src_detectors_dir = MagicMock(return_value=Path("/mock_src/detectors"))
     mock.validate_path_within_allowed_dirs = MagicMock() # Assume valid by default
     return mock


# --- Tests for get_detector_path ---
# These tests might need adjustment depending on the *actual* implementation
# and how it uses pathlib and path_utils

def test_get_detector_path_latest(dummy_session_info_for_code, mock_config_for_code, mock_path_utils):
    """Test getting the 'latest' version path (in workspace)."""
    detector_name = "my_detector"
    expected_path = Path(dummy_session_info_for_code.workspace_path) / f"{detector_name}.py"

    # Patch Path.exists for this test
    with patch('pathlib.Path.exists', return_value=True):
         # Patch path_utils usage within the function if not using dummy
         with patch('src.mcp_server_logic.code_tools.path_utils', mock_path_utils):
             # Need to re-fetch workspace path from potentially patched util
             # Or rely on the fixture's setup if the dummy uses it directly
             # Recreating expected path based on mocked util:
             expected_path = mock_path_utils.get_workspace_dir(dummy_session_info_for_code.session_id, mock_config_for_code) / f"{detector_name}.py"

             path = get_detector_path(dummy_session_info_for_code, detector_name, "latest", mock_config_for_code)
             assert path == expected_path


def test_get_detector_path_base(dummy_session_info_for_code, mock_config_for_code, mock_path_utils):
    """Test getting the 'base' version path (in src)."""
    detector_name = "base_detector"
    expected_path = mock_path_utils.get_src_detectors_dir() / f"{detector_name}.py"

    with patch('pathlib.Path.exists', return_value=True):
         with patch('src.mcp_server_logic.code_tools.path_utils', mock_path_utils):
            path = get_detector_path(dummy_session_info_for_code, detector_name, "base", mock_config_for_code)
            assert path == expected_path

def test_get_detector_path_not_found(dummy_session_info_for_code, mock_config_for_code, mock_path_utils):
    """Test FileNotFoundError when the detector file doesn't exist."""
    detector_name = "nonexistent_detector"
    with patch('pathlib.Path.exists', return_value=False): # Simulate file not existing
        with patch('src.mcp_server_logic.code_tools.path_utils', mock_path_utils):
             with pytest.raises(FileNotFoundError):
                 get_detector_path(dummy_session_info_for_code, detector_name, "latest", mock_config_for_code)

# --- Tests for _run_get_code (Sync) ---

def test_run_get_code_success(dummy_session_info_for_code, mock_config_for_code, mock_path_utils):
     """Test successfully getting code content."""
     detector_name = "my_detector"
     version = "latest"
     file_content = "def detect():\n    print('hello')"
     params = {"detector_name": detector_name, "version": version}
     # Use the workspace path from the fixture
     ws_path = Path(dummy_session_info_for_code.workspace_path)
     expected_file_path = ws_path / f"{detector_name}.py"

     # Mock pathlib methods used by the function
     with patch('src.mcp_server_logic.code_tools.path_utils', mock_path_utils), \
          patch('pathlib.Path.exists', return_value=True), \
          patch('pathlib.Path.read_text', return_value=file_content) as mock_read:

         # Re-mock get_workspace_dir to use the real tmp path from fixture
         mock_path_utils.get_workspace_dir.return_value = ws_path
         expected_file_path = mock_path_utils.get_workspace_dir(ANY, ANY) / f"{detector_name}.py"


         result = _run_get_code(dummy_session_info_for_code, params, mock_config_for_code)

         assert result == {"file_path": str(expected_file_path), "code": file_content, "version": version}
         # Ensure read_text was called on the correct path object
         # This assertion is tricky because Path objects are created dynamically.
         # We check if *any* read_text call happened. A more robust check
         # might involve mocking Path.__new__ or __init__.
         mock_read.assert_called_once()


def test_run_get_code_file_not_found(dummy_session_info_for_code, mock_config_for_code, mock_path_utils):
     """Test getting code when the file doesn't exist."""
     params = {"detector_name": "no_such_detector", "version": "latest"}
     ws_path = Path(dummy_session_info_for_code.workspace_path)

     with patch('src.mcp_server_logic.code_tools.path_utils', mock_path_utils), \
          patch('pathlib.Path.exists', return_value=False): # Simulate file not found
         mock_path_utils.get_workspace_dir.return_value = ws_path
         with pytest.raises(CodeToolsError, match="Cannot get code: Detector file not found"):
             _run_get_code(dummy_session_info_for_code, params, mock_config_for_code)

# --- Tests for _run_save_code (Sync) ---

def test_run_save_code_success(dummy_session_info_for_code, mock_config_for_code, mock_path_utils, mock_subprocess):
     """Test successfully saving code and calling git."""
     detector_name = "save_detector"
     new_code = "def new_detect(): pass"
     commit_msg = "Save new detector"
     params = {"detector_name": detector_name, "code": new_code, "commit_message": commit_msg}
     ws_path = Path(dummy_session_info_for_code.workspace_path)
     expected_file_path = ws_path / f"{detector_name}.py"

     # Mock pathlib and subprocess
     with patch('src.mcp_server_logic.code_tools.path_utils', mock_path_utils), \
          patch('pathlib.Path.exists', return_value=True), \
          patch('pathlib.Path.write_text') as mock_write:

         mock_path_utils.get_workspace_dir.return_value = ws_path
         expected_file_path = mock_path_utils.get_workspace_dir(ANY, ANY) / f"{detector_name}.py"

         result = _run_save_code(dummy_session_info_for_code, params, mock_config_for_code, mock_subprocess)

         assert result == {"file_path": str(expected_file_path), "version": "latest", "commit_message": commit_msg}
         # Verify write_text call
         mock_write.assert_called_once_with(new_code)
         # Verify path validation call
         mock_path_utils.validate_path_within_allowed_dirs.assert_called_once_with(expected_file_path, [ws_path])
         # Verify subprocess calls for git
         expected_add_call = call(['git', 'add', str(expected_file_path)], check=True, cwd=ws_path, capture_output=True, text=True)
         expected_commit_call = call(['git', 'commit', '-m', commit_msg], check=True, cwd=ws_path, capture_output=True, text=True)
         mock_subprocess.run.assert_has_calls([expected_add_call, expected_commit_call], any_order=False)

def test_run_save_code_git_error(dummy_session_info_for_code, mock_config_for_code, mock_path_utils, mock_subprocess):
     """Test saving code when a git command fails."""
     params = {"detector_name": "git_fail_detector", "code": "code"}
     ws_path = Path(dummy_session_info_for_code.workspace_path)

     # Simulate CalledProcessError on git commit
     git_error = subprocess.CalledProcessError(1, ['git', 'commit'], stderr="commit failed")
     mock_subprocess.run.side_effect = [
         MagicMock(spec=subprocess.CompletedProcess, returncode=0), # git add succeeds
         git_error # git commit fails
     ]

     with patch('src.mcp_server_logic.code_tools.path_utils', mock_path_utils), \
          patch('pathlib.Path.exists', return_value=True), \
          patch('pathlib.Path.write_text'): # Mock write

         mock_path_utils.get_workspace_dir.return_value = ws_path

         with pytest.raises(CodeToolsError, match="Git command failed: commit failed"):
             _run_save_code(dummy_session_info_for_code, params, mock_config_for_code, mock_subprocess)

# --- Tests for _run_..._async Wrappers ---

async def test_run_get_code_async_success(dummy_session_info_for_code, mock_executor, mock_session_manager_for_code, mock_config_for_code):
     """Test the async wrapper for get_code successfully."""
     tool_input = GetCodeInput(params={"detector_name": "async_get", "version": "base"})
     sync_result = {"file_path": "/mock_src/detectors/async_get.py", "code": "base code", "version": "base"}

     # Patch the sync function directly or mock the executor's behavior more precisely
     with patch('src.mcp_server_logic.code_tools._run_get_code', return_value=sync_result) as mock_sync_get, \
          patch('asyncio.get_running_loop') as mock_loop: # Need to mock the loop

         # Configure run_in_executor mock on the loop mock
         mock_loop.return_value.run_in_executor = AsyncMock(return_value=sync_result)

         output = await _run_get_code_async(dummy_session_info_for_code, tool_input, mock_executor, mock_session_manager_for_code, mock_config_for_code)

         assert isinstance(output, GetCodeOutput)
         assert output.result == sync_result
         # Verify run_in_executor was called correctly
         mock_loop.return_value.run_in_executor.assert_called_once_with(
              mock_executor, ANY, dummy_session_info_for_code, tool_input.params, mock_config_for_code # Check args passed to sync func
         )
         # Verify history add
         mock_session_manager_for_code.add_session_history.assert_called_once()
         hist_args, hist_kwargs = mock_session_manager_for_code.add_session_history.call_args
         assert hist_kwargs["event_type"] == f"{TOOL_NAME_GET_CODE}_complete"
         assert hist_kwargs["details"]["output"] == sync_result


async def test_run_save_code_async_sync_error(dummy_session_info_for_code, mock_executor, mock_session_manager_for_code, mock_config_for_code, mock_subprocess):
     """Test the async wrapper when the underlying sync save function fails."""
     tool_input = SaveCodeInput(params={"detector_name": "async_save_fail", "code": "bad"})
     error_message = "Sync save failed"

     # Patch the sync function to raise an error
     with patch('src.mcp_server_logic.code_tools._run_save_code', side_effect=CodeToolsError(error_message)) as mock_sync_save, \
          patch('asyncio.get_running_loop') as mock_loop:

         # Configure run_in_executor to propagate the error
         mock_loop.return_value.run_in_executor = AsyncMock(side_effect=CodeToolsError(error_message))


         with pytest.raises(CodeToolsError, match=f"Task {TOOL_NAME_SAVE_CODE} failed: {error_message}"):
              await _run_save_code_async(dummy_session_info_for_code, tool_input, mock_executor, mock_session_manager_for_code, mock_config_for_code, mock_subprocess)

         # Verify run_in_executor was called
         mock_loop.return_value.run_in_executor.assert_called_once()
         # Verify history add for failure
         mock_session_manager_for_code.add_session_history.assert_called_once()
         hist_args, hist_kwargs = mock_session_manager_for_code.add_session_history.call_args
         assert hist_kwargs["event_type"] == f"{TOOL_NAME_SAVE_CODE}_failed"
         assert hist_kwargs["details"]["error"]["error_type"] == "CodeToolsError"
         assert hist_kwargs["details"]["error"]["message"] == error_message
