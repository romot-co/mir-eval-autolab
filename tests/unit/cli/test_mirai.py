# tests/unit/cli/test_mirai.py
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock, AsyncMock

# テスト対象のTyperアプリケーションをインポート
try:
    # Assuming your main Typer app instance is named 'app' in src.cli.mirai
    from src.cli.mirai import app
    from src.utils.exception_utils import MiraiError, ConfigError
    # Import other necessary modules if needed for mocking targets
    # from src.evaluation import evaluation_runner
    # from mcp.client import MCPClient, ToolError, MCPError # Assuming mcp library structure
except ImportError:
    pytest.skip("Skipping cli/mirai tests due to missing src modules or app instance", allow_module_level=True)
    # Dummy app for static analysis
    class DummyApp:
        def __call__(self, *args, **kwargs): pass
    app = DummyApp()
    class MiraiError(Exception): pass
    class ConfigError(Exception): pass
    class ToolError(Exception): pass
    class MCPError(Exception): pass
    class MCPClient: pass

# --- Fixtures ---

@pytest.fixture
def runner():
    """Provides a Typer CliRunner instance."""
    return CliRunner()

@pytest.fixture
def mock_evaluation_runner():
    """Mocks the evaluation_runner module used by standalone evaluate."""
    with patch('src.cli.mirai.evaluation_runner', autospec=True) as mock:
        # Setup mock return values if needed
        mock.evaluate_detector.return_value = {'valid': True, 'metrics': {'note.f_measure': 0.9}}
        mock.run_evaluation.return_value = [{"summary": "data"}]
        yield mock

@pytest.fixture
def mock_mcp_client():
    """Mocks the MCPClient used by server mode evaluate."""
    # Assuming the client is instantiated within the command function
    # We might need to patch the specific client class
    # For now, patch where it might be imported/used, e.g., 'src.cli.mirai.MCPClient'
    # Adjust the patch target based on actual usage in mirai.py
    try:
        client_path = 'src.cli.mirai.MCPClient' # Common place
        # Try to import to check path validity, avoid hard error if wrong
        from src.cli.mirai import MCPClient
    except ImportError:
        client_path = 'mcp.client.MCPClient' # Alternative if imported directly

    with patch(client_path, autospec=True) as mock_client_class:
        mock_instance = MagicMock()
        # Make methods async if they are awaitable
        mock_instance.start_session = AsyncMock(return_value="test-session-123")
        mock_instance.run_tool = AsyncMock(return_value="job-abc")
        mock_instance.get_job_status = AsyncMock(side_effect=[
            {'status': 'pending', 'result': None},
            {'status': 'completed', 'result': {'summary': 'server data'}}
        ]) # Simulate polling
        mock_client_class.return_value = mock_instance # Instantiating the client returns our mock
        yield mock_instance # Return the instance for direct assertion

# --- Test 'evaluate' command ---

# Standalone Mode Tests
def test_evaluate_standalone_help(runner):
    """Tests `mirai evaluate --help` in standalone mode (implicitly)."""
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--audio" in result.stdout
    assert "--ref" in result.stdout
    assert "--detector" in result.stdout
    assert "--server" not in result.stdout # Ensure server args aren't shown confusingly

def test_evaluate_standalone_basic(runner, mock_evaluation_runner):
    """Tests basic standalone evaluate call."""
    audio_file = "data/test.wav"
    ref_file = "data/test.csv"
    detector_name = "SimplePeakDetector"

    result = runner.invoke(app, [
        "evaluate",
        "--audio", audio_file,
        "--ref", ref_file,
        "--detector", detector_name,
        # Add other necessary standalone options if any, e.g., --output
    ])

    assert result.exit_code == 0, f"CLI Error: {result.stdout}"
    # Assuming standalone mode calls evaluate_detector for single files
    # Adjust if it calls run_evaluation based on args
    mock_evaluation_runner.evaluate_detector.assert_called_once()
    # Check args more specifically if possible (need detector instance mapping)
    # call_args, call_kwargs = mock_evaluation_runner.evaluate_detector.call_args
    # assert call_kwargs['audio_path'].name == Path(audio_file).name
    # assert call_kwargs['reference_path'].name == Path(ref_file).name

    # Check for expected output (e.g., metrics)
    assert "Evaluation successful" in result.stdout # Or check for specific metric output
    assert "note.f_measure: 0.9" in result.stdout # Based on mock return value

def test_evaluate_standalone_missing_args(runner):
    """Tests standalone evaluate missing required arguments."""
    # Missing --audio
    result_no_audio = runner.invoke(app, [
        "evaluate",
        "--ref", "data/test.csv",
        "--detector", "Dummy"
    ])
    assert result_no_audio.exit_code != 0
    assert "Missing option '--audio'" in result_no_audio.stdout

    # Missing --ref
    result_no_ref = runner.invoke(app, [
        "evaluate",
        "--audio", "data/test.wav",
        "--detector", "Dummy"
    ])
    assert result_no_ref.exit_code != 0
    assert "Missing option '--ref'" in result_no_ref.stdout

    # Missing --detector
    result_no_detector = runner.invoke(app, [
        "evaluate",
        "--audio", "data/test.wav",
        "--ref", "data/test.csv"
    ])
    assert result_no_detector.exit_code != 0
    assert "Missing option '--detector'" in result_no_detector.stdout

def test_evaluate_standalone_evaluation_error(runner, mock_evaluation_runner):
    """Tests standalone evaluate when evaluation_runner raises an error."""
    audio_file = "data/error.wav"
    ref_file = "data/error.csv"
    detector_name = "ErrorDetector"
    error_message = "Something went wrong during evaluation"

    # Configure the mock to raise an error
    mock_evaluation_runner.evaluate_detector.side_effect = MiraiError(error_message)

    result = runner.invoke(app, [
        "evaluate",
        "--audio", audio_file,
        "--ref", ref_file,
        "--detector", detector_name,
    ])

    assert result.exit_code != 0
    mock_evaluation_runner.evaluate_detector.assert_called_once() # It should still be called
    assert f"Error: {error_message}" in result.stdout

# Server Mode Tests
def test_evaluate_server_help(runner):
    """Tests `mirai evaluate --server --help`."""
    result = runner.invoke(app, ["evaluate", "--server", "http://localhost:8000", "--help"])
    assert result.exit_code == 0
    assert "--detector" in result.stdout
    assert "--dataset" in result.stdout # or --audio-list ?
    assert "--session-id" in result.stdout
    assert "--audio" not in result.stdout # Ensure standalone args aren't shown

def test_evaluate_server_basic(runner, mock_mcp_client):
    """Tests basic server mode evaluate call, simulating job completion."""
    server_url = "http://mockserver:1234"
    detector_name = "ServerDetector"
    dataset = "full_dataset"
    expected_job_id = "job-abc" # From fixture
    expected_session_id = "test-session-123" # From fixture (assuming implicit start)

    # Reset mock call counts specifically for this test if needed
    mock_mcp_client.start_session.reset_mock()
    mock_mcp_client.run_tool.reset_mock()
    mock_mcp_client.get_job_status.reset_mock()
    # Reconfigure side effect for get_job_status for this specific test run
    mock_mcp_client.get_job_status.side_effect=[
            {'status': 'pending', 'result': None, 'error': None},
            {'status': 'completed', 'result': {'summary': 'server data'}, 'error': None}
    ]

    result = runner.invoke(app, [
        "evaluate",
        "--server", server_url,
        "--detector", detector_name,
        "--dataset", dataset,
        # Implicitly start a new session
    ])

    assert result.exit_code == 0, f"CLI Error: {result.stdout}"

    # Verify interactions
    # Assuming default behavior is to start a session if no ID provided
    mock_mcp_client.start_session.assert_called_once()
    mock_mcp_client.run_tool.assert_called_once()
    run_tool_args, run_tool_kwargs = mock_mcp_client.run_tool.call_args
    assert run_tool_kwargs['session_id'] == expected_session_id
    assert run_tool_kwargs['tool_name'] == 'run_evaluation' # Assuming tool name
    assert run_tool_kwargs['params']['detector_params_list'][0]['class_name'] == detector_name
    assert run_tool_kwargs['params']['dataset_dir'] == dataset

    # Check polling calls (at least pending and completed)
    assert mock_mcp_client.get_job_status.call_count >= 2
    mock_mcp_client.get_job_status.assert_any_call(session_id=expected_session_id, job_id=expected_job_id)

    assert f"Started evaluation job: {expected_job_id}" in result.stdout
    assert "Polling job status..." in result.stdout
    assert "Job completed successfully." in result.stdout
    assert "'summary': 'server data'" in result.stdout # Check result output

def test_evaluate_server_start_session(runner, mock_mcp_client):
    """Tests server mode evaluate starting a new session (explicit check)."""
    server_url = "http://mockserver:1234"
    detector_name = "StartSessionDetector"
    dataset = "start_dataset"

    # Reset mocks and configure side effects
    mock_mcp_client.start_session.reset_mock()
    mock_mcp_client.start_session.return_value = "newly-started-session-456"
    mock_mcp_client.run_tool.reset_mock()
    mock_mcp_client.run_tool.return_value = "job-def"
    mock_mcp_client.get_job_status.reset_mock()
    mock_mcp_client.get_job_status.side_effect=[
            {'status': 'pending', 'result': None, 'error': None},
            {'status': 'completed', 'result': {'summary': 'new session data'}, 'error': None}
    ]

    result = runner.invoke(app, [
        "evaluate",
        "--server", server_url,
        "--detector", detector_name,
        "--dataset", dataset,
        # NO --session-id provided
    ])

    assert result.exit_code == 0
    mock_mcp_client.start_session.assert_called_once()
    mock_mcp_client.run_tool.assert_called_once()
    run_tool_args, run_tool_kwargs = mock_mcp_client.run_tool.call_args
    assert run_tool_kwargs['session_id'] == "newly-started-session-456"
    assert "Job completed successfully." in result.stdout

def test_evaluate_server_reuse_session(runner, mock_mcp_client):
    """Tests server mode evaluate reusing an existing session ID."""
    server_url = "http://mockserver:1234"
    detector_name = "ReuseSessionDetector"
    dataset = "reuse_dataset"
    existing_session_id = "existing-session-789"

    # Reset mocks and configure side effects
    mock_mcp_client.start_session.reset_mock()
    mock_mcp_client.run_tool.reset_mock()
    mock_mcp_client.run_tool.return_value = "job-ghi"
    mock_mcp_client.get_job_status.reset_mock()
    mock_mcp_client.get_job_status.side_effect=[
            {'status': 'pending', 'result': None, 'error': None},
            {'status': 'completed', 'result': {'summary': 'reused session data'}, 'error': None}
    ]

    result = runner.invoke(app, [
        "evaluate",
        "--server", server_url,
        "--detector", detector_name,
        "--dataset", dataset,
        "--session-id", existing_session_id # Provide existing ID
    ])

    assert result.exit_code == 0
    mock_mcp_client.start_session.assert_not_called() # Should not start a new one
    mock_mcp_client.run_tool.assert_called_once()
    run_tool_args, run_tool_kwargs = mock_mcp_client.run_tool.call_args
    assert run_tool_kwargs['session_id'] == existing_session_id # Correct ID used
    assert "Job completed successfully." in result.stdout

def test_evaluate_server_tool_error(runner, mock_mcp_client):
    """Tests server mode evaluate when the MCP server returns a ToolError on run_tool."""
    server_url = "http://mockserver:1234"
    detector_name = "ToolErrorDetector"
    dataset = "tool_error_dataset"
    tool_error_message = "Invalid detector parameters provided"

    # Reset mocks and configure run_tool to raise error
    mock_mcp_client.start_session.reset_mock()
    mock_mcp_client.start_session.return_value = "tool-error-session"
    mock_mcp_client.run_tool.reset_mock()
    # Use the dummy ToolError defined in the skip block or import the real one
    mock_mcp_client.run_tool.side_effect = ToolError(tool_error_message)
    mock_mcp_client.get_job_status.reset_mock()

    result = runner.invoke(app, [
        "evaluate",
        "--server", server_url,
        "--detector", detector_name,
        "--dataset", dataset,
    ])

    assert result.exit_code != 0
    mock_mcp_client.start_session.assert_called_once() # Session start is attempted
    mock_mcp_client.run_tool.assert_called_once() # Tool run is attempted
    mock_mcp_client.get_job_status.assert_not_called() # Polling should not happen
    assert "Tool execution failed:" in result.stdout
    assert tool_error_message in result.stdout

def test_evaluate_server_job_fails(runner, mock_mcp_client):
    """Tests server mode evaluate when the job status polling indicates failure."""
    server_url = "http://mockserver:1234"
    detector_name = "JobFailDetector"
    dataset = "job_fail_dataset"
    job_id = "job-fail-xyz"
    session_id = "job-fail-session"
    fail_error_details = "{'type': 'ValueError', 'message': 'Something broke'}"

    # Reset mocks and configure side effects
    mock_mcp_client.start_session.reset_mock()
    mock_mcp_client.start_session.return_value = session_id
    mock_mcp_client.run_tool.reset_mock()
    mock_mcp_client.run_tool.return_value = job_id
    mock_mcp_client.get_job_status.reset_mock()
    mock_mcp_client.get_job_status.side_effect=[
            {'status': 'pending', 'result': None, 'error': None},
            {'status': 'running', 'result': None, 'error': None},
            {'status': 'failed', 'result': None, 'error': fail_error_details}
    ]

    result = runner.invoke(app, [
        "evaluate",
        "--server", server_url,
        "--detector", detector_name,
        "--dataset", dataset,
    ])

    assert result.exit_code != 0
    mock_mcp_client.start_session.assert_called_once()
    mock_mcp_client.run_tool.assert_called_once()
    assert mock_mcp_client.get_job_status.call_count == 3 # Pending, running, failed
    mock_mcp_client.get_job_status.assert_any_call(session_id=session_id, job_id=job_id)
    assert f"Job {job_id} failed." in result.stdout
    assert f"Error details: {fail_error_details}" in result.stdout

# --- Test other commands (if any) ---
# e.g., test_config_command, test_server_command 