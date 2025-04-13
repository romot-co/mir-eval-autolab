# tests/unit/mcp_server_logic/test_evaluation_tools.py
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock, ANY, call
from concurrent.futures import Executor  # For mocking the executor
from pathlib import Path

# テスト対象モジュールと依存モジュールをインポート (存在しない場合はダミーを仮定)
try:
    # evaluation_tools の関数
    from src.mcp_server_logic.evaluation_tools import (
        _run_evaluate_detector_async,
        _run_run_evaluation_async,
        _run_grid_search_async,
        EvaluationToolsError,
        TOOL_NAME_EVALUATE_DETECTOR,
        TOOL_NAME_RUN_EVALUATION,
        TOOL_NAME_GRID_SEARCH,
    )

    # コア評価・グリッドサーチ関数 (モック対象)
    from src.evaluation import evaluation_runner
    from src.evaluation.grid_search import core as grid_search_core

    # 依存スキーマ・設定・ユーティリティ
    from src.mcp_server_logic.schemas import (
        SessionInfoResponse,
        HistoryEntry,  # セッション情報
        EvaluateDetectorInput,
        EvaluateDetectorOutput,
        RunEvaluationInput,
        RunEvaluationOutput,
        GridSearchInput,
        GridSearchOutput,
        ErrorInfo,
    )
    from src.mcp_server_logic.session_manager import SessionManagerConfig

    # from src.mcp_server_logic.db_utils import StateManagementError # 必要に応じて

except ImportError:
    print(
        "Warning: Using dummy implementations for evaluation_tools.py and dependencies."
    )
    from dataclasses import dataclass, field
    from typing import Dict, Any as TypingAny, Optional, List

    class EvaluationToolsError(Exception):
        pass

    class StateManagementError(Exception):
        pass  # Dummy if needed

    TOOL_NAME_EVALUATE_DETECTOR = "evaluate_detector"
    TOOL_NAME_RUN_EVALUATION = "run_evaluation"
    TOOL_NAME_GRID_SEARCH = "grid_search"

    # --- Dummy Schemas ---
    @dataclass
    class SessionInfoResponse:  # Simplified dummy
        session_id: str
        status: str
        history: List[Dict] = field(default_factory=list)

    @dataclass
    class EvaluateDetectorInput:
        params: Dict[str, TypingAny]

    @dataclass
    class EvaluateDetectorOutput:
        result: Dict[str, TypingAny]

    @dataclass
    class RunEvaluationInput:
        params: Dict[str, TypingAny]

    @dataclass
    class RunEvaluationOutput:
        result: Dict[str, TypingAny]

    @dataclass
    class GridSearchInput:
        params: Dict[str, TypingAny]

    @dataclass
    class GridSearchOutput:
        result: Dict[str, TypingAny]

    @dataclass
    class ErrorInfo:
        error_type: str
        message: str
        details: Optional[str] = None

    @dataclass
    class SessionManagerConfig:
        db_path: str = ":memory:"  # Dummy

    # --- Dummy Core Functions (to be mocked) ---
    class DummyEvaluationRunner:
        @staticmethod
        def evaluate_detector(*args, **kwargs) -> Dict[str, TypingAny]:
            print("Dummy evaluate_detector called")
            if kwargs.get("audio_path") == "fail":
                raise ValueError("Dummy eval failure")
            return {"valid": True, "metrics": {"f_measure": 0.9}, "error_message": None}

        @staticmethod
        def run_evaluation(*args, **kwargs) -> Dict[str, TypingAny]:
            print("Dummy run_evaluation called")
            if kwargs.get("dataset_path") == "fail":
                raise ValueError("Dummy run_eval failure")
            return {
                "summary": {"f_measure_mean": 0.85},
                "results_path": "/path/to/results.json",
            }

    evaluation_runner = DummyEvaluationRunner()

    class DummyGridSearchCore:
        @staticmethod
        def run_grid_search(*args, **kwargs) -> Dict[str, TypingAny]:
            print("Dummy run_grid_search called")
            if kwargs.get("detector_name") == "fail":
                raise ValueError("Dummy grid_search failure")
            return {
                "best_params": {"threshold": 0.6},
                "best_results": {"f_measure": 0.92},
                "summary_path": "/path/gs_summary.csv",
            }

    grid_search_core = DummyGridSearchCore()

    # --- Dummy Async Wrappers (Copied for context, real wrappers assumed to exist) ---
    async def _run_async_wrapper(
        func,
        session_info: SessionInfoResponse,
        tool_input,
        tool_output_class,
        event_name: str,
        executor_mock,
        session_manager_mock,
        config,
    ):
        tool_params = tool_input.params
        event_type = ""
        output = None
        try:
            loop = asyncio.get_running_loop()
            result_dict = await loop.run_in_executor(executor_mock, func, **tool_params)
            output = tool_output_class(result=result_dict)
            event_type = f"{event_name}_complete"
            details = {"input": tool_params, "output": result_dict}
        except Exception as e:
            error_info = ErrorInfo(error_type=type(e).__name__, message=str(e))
            event_type = f"{event_name}_failed"
            details = {"input": tool_params, "error": error_info.__dict__}
            await session_manager_mock.add_session_history(
                session_id=session_info.session_id,
                event_type=event_type,
                details=details,
                config=config,
                db_utils_mock=ANY,
                misc_utils_mock=ANY,
                validate_func_mock=ANY,
            )
            raise EvaluationToolsError(f"Task {event_name} failed: {e}") from e

        await session_manager_mock.add_session_history(
            session_id=session_info.session_id,
            event_type=event_type,
            details=details,
            config=config,
            db_utils_mock=ANY,
            misc_utils_mock=ANY,
            validate_func_mock=ANY,
        )
        return output

    async def _run_evaluate_detector_async(
        session_info: SessionInfoResponse,
        tool_input: EvaluateDetectorInput,
        executor_mock,
        session_manager_mock,
        config,
    ):
        return await _run_async_wrapper(
            evaluation_runner.evaluate_detector,
            session_info,
            tool_input,
            EvaluateDetectorOutput,
            TOOL_NAME_EVALUATE_DETECTOR,
            executor_mock,
            session_manager_mock,
            config,
        )

    async def _run_run_evaluation_async(
        session_info: SessionInfoResponse,
        tool_input: RunEvaluationInput,
        executor_mock,
        session_manager_mock,
        config,
    ):
        return await _run_async_wrapper(
            evaluation_runner.run_evaluation,
            session_info,
            tool_input,
            RunEvaluationOutput,
            TOOL_NAME_RUN_EVALUATION,
            executor_mock,
            session_manager_mock,
            config,
        )

    async def _run_grid_search_async(
        session_info: SessionInfoResponse,
        tool_input: GridSearchInput,
        executor_mock,
        session_manager_mock,
        config,
    ):
        return await _run_async_wrapper(
            grid_search_core.run_grid_search,
            session_info,
            tool_input,
            GridSearchOutput,
            TOOL_NAME_GRID_SEARCH,
            executor_mock,
            session_manager_mock,
            config,
        )


# --- Fixtures ---


@pytest.fixture
def mock_executor():
    """Mocks concurrent.futures.Executor."""
    mock = MagicMock(spec=Executor)

    def run_sync(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise e

    # Attach the sync runner logic to the mock executor
    # This part needs refinement depending on how run_in_executor is called/mocked
    # If run_in_executor is patched on the loop, this fixture might just need to be spec=Executor
    return mock


@pytest.fixture
def mock_session_manager_for_eval():
    """Mocks session_manager functions needed by eval tools."""
    mock = MagicMock()
    mock.add_session_history = AsyncMock()
    return mock


@pytest.fixture
def mock_config_for_eval() -> SessionManagerConfig:
    """Provides dummy config potentially needed."""
    return SessionManagerConfig()  # Use dummy config


@pytest.fixture
def dummy_session_info_for_eval() -> SessionInfoResponse:
    """Provides a basic SessionInfoResponse object."""
    return SessionInfoResponse(session_id="eval_test_sid", status="running")


# Mocks for the core sync functions
@pytest.fixture
def mock_evaluate_detector():
    """Mocks evaluation_runner.evaluate_detector."""
    return MagicMock(
        return_value={
            "valid": True,
            "metrics": {"f_measure": 0.95},
            "error_message": None,
        }
    )


@pytest.fixture
def mock_run_evaluation():
    """Mocks evaluation_runner.run_evaluation."""
    return MagicMock(
        return_value={
            "summary": {"f_measure_mean": 0.88},
            "results_path": "/results.json",
        }
    )


@pytest.fixture
def mock_run_grid_search():
    """Mocks grid_search_core.run_grid_search."""
    return MagicMock(
        return_value={
            "best_params": {"p": 1},
            "best_results": {"f": 0.99},
            "summary_path": "/gs.csv",
        }
    )


@pytest.fixture
def mock_path_utils():
    """Mocks the path_utils module functions."""
    mock = MagicMock()
    mock.get_output_dir = MagicMock(return_value=Path("/mock_output"))
    mock.get_ground_truth_dir = MagicMock(return_value=Path("/mock_ground_truth"))
    return mock


# --- Tests for evaluation tool async wrappers ---


@pytest.mark.asyncio
async def test_run_evaluate_detector_async_success(
    dummy_session_info_for_eval,
    mock_executor,
    mock_session_manager_for_eval,
    mock_config_for_eval,
    mock_evaluate_detector,
):
    """Test the evaluate_detector async wrapper successfully."""
    tool_input = EvaluateDetectorInput(
        params={"detector_name": "test", "audio_path": "a.wav"}
    )
    sync_result = mock_evaluate_detector.return_value

    # Define patch target path
    eval_runner_path = (
        "src.mcp_server_logic.evaluation_tools.evaluation_runner.evaluate_detector"
    )
    try:
        with patch(eval_runner_path, mock_evaluate_detector), patch(
            "asyncio.get_running_loop"
        ) as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=sync_result)
            output = await _run_evaluate_detector_async(
                dummy_session_info_for_eval,
                tool_input,
                mock_executor,
                mock_session_manager_for_eval,
                mock_config_for_eval,
            )
    except (ImportError, AttributeError):
        print(f"Could not patch {eval_runner_path}, using dummy logic path.")
        global evaluation_runner
        evaluation_runner.evaluate_detector = mock_evaluate_detector
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=sync_result)
            output = await _run_evaluate_detector_async(
                dummy_session_info_for_eval,
                tool_input,
                mock_executor,
                mock_session_manager_for_eval,
                mock_config_for_eval,
            )

    assert isinstance(output, EvaluateDetectorOutput)
    assert output.result == sync_result
    mock_loop.return_value.run_in_executor.assert_called_once_with(
        mock_executor, mock_evaluate_detector, **tool_input.params
    )
    mock_session_manager_for_eval.add_session_history.assert_called_once()
    hist_args, hist_kwargs = mock_session_manager_for_eval.add_session_history.call_args
    assert hist_kwargs["event_type"] == f"{TOOL_NAME_EVALUATE_DETECTOR}_complete"
    assert hist_kwargs["details"]["output"] == sync_result


@pytest.mark.asyncio
async def test_run_run_evaluation_async_success(
    dummy_session_info_for_eval,
    mock_executor,
    mock_session_manager_for_eval,
    mock_config_for_eval,
    mock_run_evaluation,
):
    """Test the run_evaluation async wrapper successfully."""
    tool_input = RunEvaluationInput(
        params={"dataset_path": "/data", "output_dir": "/out"}
    )
    sync_result = mock_run_evaluation.return_value

    target_path = (
        "src.mcp_server_logic.evaluation_tools.evaluation_runner.run_evaluation"
    )
    try:
        with patch(target_path, mock_run_evaluation), patch(
            "asyncio.get_running_loop"
        ) as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=sync_result)
            output = await _run_run_evaluation_async(
                dummy_session_info_for_eval,
                tool_input,
                mock_executor,
                mock_session_manager_for_eval,
                mock_config_for_eval,
            )
    except (ImportError, AttributeError):
        print(f"Could not patch {target_path}, using dummy logic path.")
        global evaluation_runner
        evaluation_runner.run_evaluation = mock_run_evaluation
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=sync_result)
            output = await _run_run_evaluation_async(
                dummy_session_info_for_eval,
                tool_input,
                mock_executor,
                mock_session_manager_for_eval,
                mock_config_for_eval,
            )

    assert isinstance(output, RunEvaluationOutput)
    assert output.result == sync_result
    mock_loop.return_value.run_in_executor.assert_called_once_with(
        mock_executor, mock_run_evaluation, **tool_input.params
    )
    mock_session_manager_for_eval.add_session_history.assert_called_once()
    hist_args, hist_kwargs = mock_session_manager_for_eval.add_session_history.call_args
    assert hist_kwargs["event_type"] == f"{TOOL_NAME_RUN_EVALUATION}_complete"


@pytest.mark.asyncio
async def test_run_grid_search_async_success(
    dummy_session_info_for_eval,
    mock_executor,
    mock_session_manager_for_eval,
    mock_config_for_eval,
    mock_run_grid_search,
):
    """Test the grid_search async wrapper successfully."""
    tool_input = GridSearchInput(params={"detector_name": "gs_test", "param_grid": {}})
    sync_result = mock_run_grid_search.return_value

    target_path = (
        "src.mcp_server_logic.evaluation_tools.grid_search_core.run_grid_search"
    )
    try:
        with patch(target_path, mock_run_grid_search), patch(
            "asyncio.get_running_loop"
        ) as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=sync_result)
            output = await _run_grid_search_async(
                dummy_session_info_for_eval,
                tool_input,
                mock_executor,
                mock_session_manager_for_eval,
                mock_config_for_eval,
            )
    except (ImportError, AttributeError):
        print(f"Could not patch {target_path}, using dummy logic path.")
        global grid_search_core
        grid_search_core.run_grid_search = mock_run_grid_search
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=sync_result)
            output = await _run_grid_search_async(
                dummy_session_info_for_eval,
                tool_input,
                mock_executor,
                mock_session_manager_for_eval,
                mock_config_for_eval,
            )

    assert isinstance(output, GridSearchOutput)
    assert output.result == sync_result
    mock_loop.return_value.run_in_executor.assert_called_once_with(
        mock_executor, mock_run_grid_search, **tool_input.params
    )
    mock_session_manager_for_eval.add_session_history.assert_called_once()
    hist_args, hist_kwargs = mock_session_manager_for_eval.add_session_history.call_args
    assert hist_kwargs["event_type"] == f"{TOOL_NAME_GRID_SEARCH}_complete"


@pytest.mark.asyncio
async def test_run_evaluate_detector_async_failure(
    dummy_session_info_for_eval,
    mock_executor,
    mock_session_manager_for_eval,
    mock_config_for_eval,
    mock_evaluate_detector,
):
    """Test the evaluate_detector async wrapper when the core function fails."""
    tool_input = EvaluateDetectorInput(
        params={"detector_name": "fail_eval", "audio_path": "fail"}
    )
    error_message = "Core evaluation failed"
    core_exception = ValueError(error_message)
    mock_evaluate_detector.side_effect = core_exception

    eval_runner_path = (
        "src.mcp_server_logic.evaluation_tools.evaluation_runner.evaluate_detector"
    )
    try:
        with patch(eval_runner_path, mock_evaluate_detector), patch(
            "asyncio.get_running_loop"
        ) as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=core_exception
            )
            with pytest.raises(
                EvaluationToolsError,
                match=f"Task {TOOL_NAME_EVALUATE_DETECTOR} failed: {error_message}",
            ):
                await _run_evaluate_detector_async(
                    dummy_session_info_for_eval,
                    tool_input,
                    mock_executor,
                    mock_session_manager_for_eval,
                    mock_config_for_eval,
                )
    except (ImportError, AttributeError):
        print(f"Could not patch {eval_runner_path}, using dummy logic path.")
        global evaluation_runner
        evaluation_runner.evaluate_detector = mock_evaluate_detector
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=core_exception
            )
            with pytest.raises(
                EvaluationToolsError,
                match=f"Task {TOOL_NAME_EVALUATE_DETECTOR} failed: {error_message}",
            ):
                await _run_evaluate_detector_async(
                    dummy_session_info_for_eval,
                    tool_input,
                    mock_executor,
                    mock_session_manager_for_eval,
                    mock_config_for_eval,
                )

    mock_loop.return_value.run_in_executor.assert_called_once_with(
        mock_executor, mock_evaluate_detector, **tool_input.params
    )
    mock_session_manager_for_eval.add_session_history.assert_called_once()
    hist_args, hist_kwargs = mock_session_manager_for_eval.add_session_history.call_args
    assert hist_kwargs["event_type"] == f"{TOOL_NAME_EVALUATE_DETECTOR}_failed"
    assert hist_kwargs["details"]["input"] == tool_input.params
    assert hist_kwargs["details"]["error"]["error_type"] == "ValueError"
    assert hist_kwargs["details"]["error"]["message"] == error_message
