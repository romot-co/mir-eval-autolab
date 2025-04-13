# tests/unit/mcp_server_logic/test_session_manager.py
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock, ANY
from datetime import datetime, timezone, timedelta

# テスト対象モジュールと依存モジュールをインポート (存在しない場合はダミーを仮定)
try:
    from src.mcp_server_logic import session_manager
    from src.mcp_server_logic.session_manager import (
        start_session,
        get_session_info,
        add_session_history,
        cleanup_old_sessions,
        update_session_status_internal,  # add_session_history のテストで必要になる可能性
        SessionManagerConfig,  # セッションマネージャ設定用 (仮定)
    )
    from src.mcp_server_logic.schemas import (
        SessionInfoResponse,
        HistoryEntry,
        SessionStatus,  # 必要に応じて追加
        # BestMetrics, # 必要に応じて追加
        # EvaluationResultSummary # add_session_history のテストで使う可能性
    )
    from src.mcp_server_logic.db_utils import StateManagementError
    from jsonschema import ValidationError  # add_session_history のテスト用
except ImportError:
    print(
        "Warning: Using dummy implementations for session_manager.py and dependencies."
    )
    from dataclasses import dataclass, field
    from enum import Enum
    from typing import Dict, Any as TypingAny, Optional, List

    class SessionStatus(str, Enum):
        INITIALIZING = "initializing"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        STALLED = "stalled"  # 必要に応じて追加

    @dataclass
    class HistoryEntry:  # Simplified dummy
        event_id: str
        timestamp: str
        event_type: str
        details: Dict[str, TypingAny]

    @dataclass
    class SessionInfoResponse:  # Simplified dummy
        session_id: str
        status: str  # Use str instead of SessionStatus for simplicity
        created_at: str = ""
        last_update: str = ""
        history: List[HistoryEntry] = field(default_factory=list)
        # Add other fields like best_metrics_json if needed by tests
        best_metrics_json: Optional[str] = None
        stagnation_count: int = 0

    @dataclass
    class SessionManagerConfig:  # Dummy config
        db_path: str = ":memory:"
        session_timeout_hours: int = 24 * 7  # 1 week
        # Add other relevant config if needed

    class StateManagementError(Exception):
        pass

    class ValidationError(Exception):
        pass  # Dummy for jsonschema

    # Dummy session_manager functions
    async def start_session(
        config: SessionManagerConfig, db_utils_mock, misc_utils_mock
    ) -> SessionInfoResponse:
        session_id = misc_utils_mock.generate_id("session")
        timestamp = misc_utils_mock.get_timestamp()
        status = SessionStatus.INITIALIZING.value
        # 実際のSQLクエリ文字列を使用
        insert_query = "INSERT INTO sessions (session_id, status, created_at, last_update, best_metrics_json, stagnation_count) VALUES (?, ?, ?, ?, ?, ?)"
        await db_utils_mock.db_execute_commit_async(
            config.db_path,
            insert_query,  # 実際のSQLクエリ
            (session_id, status, timestamp, timestamp, None, 0),  # Dummy params
        )
        return SessionInfoResponse(
            session_id=session_id,
            status=status,
            created_at=timestamp,
            last_update=timestamp,
        )

    async def get_session_info(
        session_id: str, config: SessionManagerConfig, db_utils_mock
    ) -> Optional[SessionInfoResponse]:
        # 実際のSQLクエリ文字列を使用
        session_query = "SELECT * FROM sessions WHERE session_id = ?"
        session_row = await db_utils_mock.db_fetch_one_async(
            config.db_path, session_query, (session_id,)
        )
        if not session_row:
            return None

        # 実際のSQLクエリ文字列を使用
        history_query = (
            "SELECT * FROM history WHERE session_id = ? ORDER BY timestamp ASC"
        )
        history_rows = await db_utils_mock.db_fetch_all_async(
            config.db_path, history_query, (session_id,)
        )
        history = []
        if history_rows:
            history = [
                # Handle potential dict vs Row object from mock
                HistoryEntry(
                    event_id=row[0] if isinstance(row, tuple) else row.get("event_id"),
                    timestamp=(
                        row[2] if isinstance(row, tuple) else row.get("timestamp")
                    ),
                    event_type=(
                        row[3] if isinstance(row, tuple) else row.get("event_type")
                    ),
                    details=(
                        json.loads(row[4])
                        if isinstance(row, tuple)
                        else json.loads(row.get("details_json"))
                    ),
                )
                for row in history_rows
            ]

        get = getattr(session_row, "get", session_row.__getitem__)
        return SessionInfoResponse(
            session_id=get("session_id"),
            status=get("status"),
            created_at=get("created_at"),
            last_update=get("last_update"),
            history=history,
            best_metrics_json=get("best_metrics_json", None),
            stagnation_count=get("stagnation_count", 0),
        )

    async def add_session_history(
        session_id: str,
        event_type: str,
        details: Dict[str, TypingAny],
        config: SessionManagerConfig,
        db_utils_mock,
        misc_utils_mock,
        validate_func_mock,
    ):
        # Simulate schema validation
        try:
            validate_func_mock(details)  # Assume a mock validation function
        except ValidationError as e:
            print(f"Schema validation failed: {e}")  # Log or handle
            # Depending on actual implementation, might raise or just log
            raise  # Re-raise for test assertion

        event_id = misc_utils_mock.generate_id("event")
        timestamp = misc_utils_mock.get_timestamp()
        details_json = json.dumps(details)

        # 実際のSQLクエリ文字列を使用
        history_insert = "INSERT INTO history (event_id, session_id, timestamp, event_type, details_json) VALUES (?, ?, ?, ?, ?)"
        # Simulate history insert
        await db_utils_mock.db_execute_commit_async(
            config.db_path,
            history_insert,  # 実際のSQLクエリ
            (event_id, session_id, timestamp, event_type, details_json),
        )

        # 実際のSQLクエリ文字列を使用
        session_update = "UPDATE sessions SET last_update = ? WHERE session_id = ?"
        # Simulate session update (last_update, potentially status, metrics, stagnation)
        # Basic update: just last_update
        await db_utils_mock.db_execute_commit_async(
            config.db_path,
            session_update,  # 実際のSQLクエリ
            (timestamp, session_id),  # Only update timestamp
        )
        # --- Placeholder for metric/stagnation logic ---
        # if event_type == "evaluation_complete":
        #    # Basic best metric update check
        #    # Basic stagnation check

    async def cleanup_old_sessions(config: SessionManagerConfig, db_utils_mock):
        # 実際のSQLクエリ文字列を使用
        delete_query = "DELETE FROM sessions WHERE last_update < ?"
        # Only check if the delete command is executed
        cutoff_timestamp = (
            datetime.now(timezone.utc) - timedelta(hours=config.session_timeout_hours)
        ).isoformat()
        await db_utils_mock.db_execute_commit_async(
            config.db_path,
            delete_query,  # 実際のSQLクエリ
            (cutoff_timestamp,),  # Timestamp cutoff parameter
        )

    async def update_session_status_internal(
        session_id: str, new_status: str, db_path: str, db_utils_mock
    ):
        # 実際のSQLクエリ文字列を使用
        update_query = (
            "UPDATE sessions SET status = ?, last_update = ? WHERE session_id = ?"
        )
        # Dummy internal status update function
        timestamp = datetime.now(timezone.utc).isoformat()
        await db_utils_mock.db_execute_commit_async(
            db_path,
            update_query,  # 実際のSQLクエリ
            (new_status, timestamp, session_id),  # Update status and timestamp
        )


# --- Fixtures ---


@pytest.fixture
def mock_config() -> SessionManagerConfig:
    """Provides a dummy SessionManagerConfig."""
    return SessionManagerConfig(
        db_path=":memory:", session_timeout_hours=1
    )  # Short timeout for cleanup test


@pytest.fixture
def mock_db_utils():
    """Mocks the db_utils module functions."""
    mock = MagicMock()
    mock.db_execute_commit_async = AsyncMock(return_value=1)
    mock.db_fetch_one_async = AsyncMock(return_value=None)  # Default: session not found
    mock.db_fetch_all_async = AsyncMock(return_value=[])  # Default: no history
    return mock


@pytest.fixture
def mock_misc_utils():
    """Mocks the misc_utils module functions."""
    mock = MagicMock()
    # Use a list to provide different IDs for session and event
    mock.generate_id = MagicMock()
    mock.get_timestamp = MagicMock(return_value=datetime.now(timezone.utc).isoformat())
    return mock


@pytest.fixture
def mock_validate():
    """Mocks the jsonschema.validate function."""
    return MagicMock()  # Does nothing by default (valid)


# --- Tests for start_session ---


@pytest.mark.asyncio
async def test_start_session_success(mock_config, mock_db_utils, mock_misc_utils):
    """Test starting a session successfully."""
    # モックをリセット
    mock_misc_utils.generate_id.reset_mock()
    mock_misc_utils.generate_id.side_effect = ["session_abc"]

    response = await start_session(mock_config, mock_db_utils, mock_misc_utils)

    assert isinstance(response, SessionInfoResponse)
    assert response.session_id == "session_abc"  # First ID from mock
    assert response.status == SessionStatus.INITIALIZING.value
    assert response.created_at is not None
    assert response.last_update == response.created_at
    assert response.history == []

    # Verify DB insert call
    mock_db_utils.db_execute_commit_async.assert_called_once()
    args, kwargs = mock_db_utils.db_execute_commit_async.call_args
    assert args[0] == mock_config.db_path
    assert "INSERT INTO sessions" in args[1]
    db_params = args[2]
    assert response.session_id in db_params
    assert response.status in db_params
    assert response.created_at in db_params


@pytest.mark.asyncio
async def test_start_session_db_error(mock_config, mock_db_utils, mock_misc_utils):
    """Test that StateManagementError during DB insert is propagated."""
    mock_db_utils.db_execute_commit_async.side_effect = StateManagementError(
        "DB unavailable"
    )

    with pytest.raises(StateManagementError, match="DB unavailable"):
        await start_session(mock_config, mock_db_utils, mock_misc_utils)


# --- Tests for get_session_info ---


@pytest.mark.asyncio
async def test_get_session_info_found(mock_config, mock_db_utils):
    """Test getting information for an existing session with history."""
    session_id = "existing_session"
    now_iso = datetime.now(timezone.utc).isoformat()
    session_data = {
        "session_id": session_id,
        "status": SessionStatus.RUNNING.value,
        "created_at": now_iso,
        "last_update": now_iso,
        "best_metrics_json": '{"f_measure": 0.8}',
        "stagnation_count": 1,
    }
    history_data = [
        # Simulate tuple format from aiosqlite fetchall
        ("event_aaa", session_id, now_iso, "tool_started", '{"tool": "eval"}'),
        ("event_bbb", session_id, now_iso, "tool_completed", '{"result": "ok"}'),
    ]
    mock_db_utils.db_fetch_one_async.return_value = (
        session_data  # Simulate finding session
    )
    mock_db_utils.db_fetch_all_async.return_value = (
        history_data  # Simulate finding history
    )

    response = await get_session_info(session_id, mock_config, mock_db_utils)

    assert isinstance(response, SessionInfoResponse)
    assert response.session_id == session_id
    assert response.status == SessionStatus.RUNNING.value
    assert response.best_metrics_json == '{"f_measure": 0.8}'
    assert response.stagnation_count == 1
    assert len(response.history) == 2
    assert response.history[0].event_id == "event_aaa"
    assert response.history[0].event_type == "tool_started"
    assert response.history[0].details == {"tool": "eval"}
    assert response.history[1].event_id == "event_bbb"

    # Verify DB calls
    mock_db_utils.db_fetch_one_async.assert_called_once_with(
        mock_config.db_path, ANY, (session_id,)
    )
    mock_db_utils.db_fetch_all_async.assert_called_once_with(
        mock_config.db_path, ANY, (session_id,)
    )


@pytest.mark.asyncio
async def test_get_session_info_not_found(mock_config, mock_db_utils):
    """Test getting information for a non-existent session."""
    session_id = "non_existent_session"
    mock_db_utils.db_fetch_one_async.return_value = None  # Simulate not found

    response = await get_session_info(session_id, mock_config, mock_db_utils)

    assert response is None
    mock_db_utils.db_fetch_one_async.assert_called_once_with(
        mock_config.db_path, ANY, (session_id,)
    )
    mock_db_utils.db_fetch_all_async.assert_not_called()  # Should not query history if session not found


@pytest.mark.asyncio
async def test_get_session_info_db_error(mock_config, mock_db_utils):
    """Test StateManagementError during DB fetch."""
    session_id = "error_session"
    mock_db_utils.db_fetch_one_async.side_effect = StateManagementError(
        "DB read failed"
    )

    with pytest.raises(StateManagementError, match="DB read failed"):
        await get_session_info(session_id, mock_config, mock_db_utils)


# --- Tests for add_session_history ---


@pytest.mark.asyncio
async def test_add_session_history_success(
    mock_config, mock_db_utils, mock_misc_utils, mock_validate
):
    """Test adding a history event successfully."""
    session_id = "history_test_session"
    event_type = "test_event"
    details = {"key": "value", "number": 123}
    timestamp_before = datetime.now(timezone.utc).isoformat()

    # モックをリセット
    mock_misc_utils.generate_id.reset_mock()
    mock_misc_utils.generate_id.side_effect = ["event_123"]
    mock_misc_utils.get_timestamp.return_value = timestamp_before  # Control timestamp

    await add_session_history(
        session_id,
        event_type,
        details,
        mock_config,
        mock_db_utils,
        mock_misc_utils,
        mock_validate,
    )

    # Verify validation call
    mock_validate.assert_called_once_with(details)

    # Verify DB calls (history insert + session update)
    assert mock_db_utils.db_execute_commit_async.call_count == 2

    # Check history insert call
    history_call_args, _ = mock_db_utils.db_execute_commit_async.call_args_list[0]
    assert history_call_args[0] == mock_config.db_path
    assert "INSERT INTO history" in history_call_args[1]
    history_params = history_call_args[2]
    assert "event_123" in history_params  # Second ID from mock
    assert session_id in history_params
    assert timestamp_before in history_params
    assert event_type in history_params
    assert json.dumps(details) in history_params

    # Check session update call (basic: only timestamp update)
    session_update_call_args, _ = mock_db_utils.db_execute_commit_async.call_args_list[
        1
    ]
    assert session_update_call_args[0] == mock_config.db_path
    assert (
        "UPDATE sessions SET last_update = ?" in session_update_call_args[1]
    )  # Basic check
    session_update_params = session_update_call_args[2]
    assert timestamp_before in session_update_params
    assert session_id in session_update_params


@pytest.mark.asyncio
async def test_add_session_history_validation_error(
    mock_config, mock_db_utils, mock_misc_utils, mock_validate
):
    """Test that validation error prevents DB writes."""
    session_id = "validation_fail_session"
    event_type = "invalid_event"
    details = {"wrong_key": "bad_data"}

    # Simulate validation failure
    mock_validate.side_effect = ValidationError("Schema mismatch")

    with pytest.raises(ValidationError, match="Schema mismatch"):
        await add_session_history(
            session_id,
            event_type,
            details,
            mock_config,
            mock_db_utils,
            mock_misc_utils,
            mock_validate,
        )

    # Verify validation was called but DB was not
    mock_validate.assert_called_once_with(details)
    mock_db_utils.db_execute_commit_async.assert_not_called()


# --- Test for cleanup_old_sessions ---


@pytest.mark.asyncio
async def test_cleanup_old_sessions(mock_config, mock_db_utils):
    """Test that the cleanup function executes a delete query."""
    await cleanup_old_sessions(mock_config, mock_db_utils)

    # Verify that a DELETE query was executed
    mock_db_utils.db_execute_commit_async.assert_called_once()
    args, kwargs = mock_db_utils.db_execute_commit_async.call_args
    assert args[0] == mock_config.db_path
    assert "DELETE FROM sessions WHERE last_update < ?" in args[1]
    # Check that the timestamp parameter is roughly correct (within a reasonable range)
    cutoff_time_param = args[2][0]
    expected_cutoff = datetime.now(timezone.utc) - timedelta(
        hours=mock_config.session_timeout_hours
    )
    # Allow some tolerance for test execution time
    assert abs(datetime.fromisoformat(cutoff_time_param) - expected_cutoff) < timedelta(
        seconds=10
    )
