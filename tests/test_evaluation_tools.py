import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY

# Target module
from mcp_server_logic import evaluation_tools
from src.utils.exception_utils import EvaluationError, GridSearchError

# --- Fixtures --- #

@pytest.fixture
def mock_config_paths():
    # Provide dummy paths used by the refactored functions or registration
    return {
        "paths": {
            "workspace": "/fake/workspace",
            "evaluation_results": "/fake/workspace/evaluation_results",
            "grid_search_results": "/fake/workspace/grid_search_results",
            "db": "/fake/workspace/db" # Needed if db interaction tested
        }
        # Add other necessary config parts if needed by the tools
    }

@pytest.fixture
def mock_add_history_sync():
    # Mock the synchronous history adding function
    return MagicMock()

# --- Test _run_evaluate --- #

@patch("mcp_server_logic.evaluation_tools.run_evaluation_core")
@patch("mcp_server_logic.evaluation_tools.ensure_dir")
@patch("mcp_server_logic.evaluation_tools.get_output_dir", return_value=Path("/fake/workspace/evaluation_results/session_test_job_123"))
def test_run_evaluate_success(mock_get_out_dir, mock_ensure_dir, mock_run_core, mock_add_history_sync):
    job_id = "job_123"
    eval_results_dir = Path("/fake/workspace/evaluation_results")
    kwargs = {
        "detector_name": "TestDetector",
        "dataset_name": "TestData",
        "session_id": "session_test",
        # Other args passed to run_evaluation_core
        "num_procs": 2
    }
    expected_core_result = {"TestDetector": {"overall_metrics": {"note": {"f_measure": 0.85}}}}
    mock_run_core.return_value = expected_core_result

    # Run the task function
    result = evaluation_tools._run_evaluate(
        job_id, eval_results_dir, mock_add_history_sync, **kwargs
    )

    # Assertions
    mock_ensure_dir.assert_called_once_with(Path("/fake/workspace/evaluation_results/session_test_job_123"))
    mock_run_core.assert_called_once_with(**kwargs)
    assert result == expected_core_result

    # Check history calls
    expected_calls = [
        call("session_test", "evaluation_started", {"detector": "TestDetector", "params": kwargs}),
        call("session_test", "evaluation_complete", {"detector": "TestDetector", "f_measure": 0.85, "summary": expected_core_result["TestDetector"]["overall_metrics"]})
    ]
    mock_add_history_sync.assert_has_calls(expected_calls)

@patch("mcp_server_logic.evaluation_tools.run_evaluation_core", side_effect=Exception("Core eval failed"))
@patch("mcp_server_logic.evaluation_tools.ensure_dir")
@patch("mcp_server_logic.evaluation_tools.get_output_dir", return_value=Path("/fake/workspace/evaluation_results/session_fail_job_456"))
def test_run_evaluate_failure(mock_get_out_dir, mock_ensure_dir, mock_run_core, mock_add_history_sync):
    job_id = "job_456"
    eval_results_dir = Path("/fake/workspace/evaluation_results")
    kwargs = {"detector_name": "FailDetector", "session_id": "session_fail"}

    with pytest.raises(EvaluationError, match="Core eval failed"):
        evaluation_tools._run_evaluate(job_id, eval_results_dir, mock_add_history_sync, **kwargs)

    # Check history calls (only started and failed)
    expected_calls = [
        call("session_fail", "evaluation_started", {"detector": "FailDetector", "params": kwargs}),
        call("session_fail", "evaluation_failed", {"detector": "FailDetector", "error": "Core eval failed"})
    ]
    mock_add_history_sync.assert_has_calls(expected_calls)
    assert mock_add_history_sync.call_count == 2

# --- Test _execute_grid_search --- #

@patch("mcp_server_logic.evaluation_tools.run_grid_search_core")
@patch("mcp_server_logic.evaluation_tools.ensure_dir")
@patch("mcp_server_logic.evaluation_tools.get_output_dir", return_value=Path("/fake/workspace/grid_results/session_grid_job_789"))
@patch("builtins.open", new_callable=MagicMock)
@patch("yaml.safe_load", return_value={"detector_name": "GridDetector"})
def test_execute_grid_search_success(mock_yaml_load, mock_open, mock_get_out_dir, mock_ensure_dir, mock_run_grid_core, mock_add_history_sync):
    job_id = "job_789"
    grid_results_dir = Path("/fake/workspace/grid_search_results")
    kwargs = {
        "config_path": "/path/to/grid_config.yaml",
        "session_id": "session_grid",
        "num_procs": 4
    }
    expected_core_result = {"best_params": {"p1": 1}, "best_score": 0.9}
    mock_run_grid_core.return_value = expected_core_result

    result = evaluation_tools._execute_grid_search(
        job_id, grid_results_dir, mock_add_history_sync, **kwargs
    )

    # Assertions
    mock_ensure_dir.assert_called_once_with(Path("/fake/workspace/grid_results/session_grid_job_789"))
    mock_run_grid_core.assert_called_once_with(**kwargs)
    # Check that output_directory is added to the result
    expected_result_with_path = {**expected_core_result, "output_directory": "/fake/workspace/grid_results/session_grid_job_789"}
    assert result == expected_result_with_path

    # Check history calls
    expected_calls = [
        call("session_grid", "grid_search_started", {"config_path": kwargs["config_path"], "detector": "GridDetector"}),
        call("session_grid", "grid_search_complete", {"detector": "GridDetector", "best_params": {"p1": 1}, "best_score": 0.9})
    ]
    mock_add_history_sync.assert_has_calls(expected_calls)

@patch("mcp_server_logic.evaluation_tools.run_grid_search_core", side_effect=Exception("Core grid failed"))
@patch("mcp_server_logic.evaluation_tools.ensure_dir")
@patch("mcp_server_logic.evaluation_tools.get_output_dir", return_value=Path("/fake/workspace/grid_results/session_gfail_job_101"))
@patch("builtins.open", new_callable=MagicMock)
@patch("yaml.safe_load", return_value={"detector_name": "GridFailDetector"})
def test_execute_grid_search_failure(mock_yaml_load, mock_open, mock_get_out_dir, mock_ensure_dir, mock_run_grid_core, mock_add_history_sync):
    job_id = "job_101"
    grid_results_dir = Path("/fake/workspace/grid_search_results")
    kwargs = {"config_path": "/path/to/fail_config.yaml", "session_id": "session_gfail"}

    with pytest.raises(GridSearchError, match="Core grid failed"):
        evaluation_tools._execute_grid_search(job_id, grid_results_dir, mock_add_history_sync, **kwargs)

    # Check history calls
    expected_calls = [
        call("session_gfail", "grid_search_started", {"config_path": kwargs["config_path"], "detector": "GridFailDetector"}),
        call("session_gfail", "grid_search_failed", {"detector": "GridFailDetector", "error": "Core grid failed"})
    ]
    mock_add_history_sync.assert_has_calls(expected_calls)
    assert mock_add_history_sync.call_count == 2

# TODO: Add tests for register_evaluation_tools (more complex, involves mocking MCP and start_async_job_func) 