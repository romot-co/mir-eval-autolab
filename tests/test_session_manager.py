import pytest
import sqlite3
from pathlib import Path
import json

# Assuming manager modules are in mcp_server_logic/
from mcp_server_logic import session_manager, db_utils
from mcp_server_logic.utils import generate_id, get_timestamp

# Re-use the in-memory DB fixture from test_db_utils
# (or define it here if test files are independent)

@pytest.fixture
def in_memory_db_session(monkeypatch):
    """Fixture for in-memory DB for session tests."""
    db_path_id = ":memory:"
    # Create the in-memory connection
    conn = sqlite3.connect(db_path_id, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # --- Directly create schema in the in-memory DB --- #
    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY, base_algorithm TEXT, start_time REAL,
            last_update REAL, status TEXT, history TEXT, config TEXT,
            current_metrics TEXT, best_metrics TEXT, best_code_version TEXT,
            best_code_path TEXT, baseline_metrics TEXT, cycle_state TEXT
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY, session_id TEXT, tool_name TEXT, status TEXT,
            start_time REAL, end_time REAL, result TEXT, error_details TEXT,
            task_args TEXT, worker_id TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
        """)
        conn.commit()
    except Exception as e:
        conn.close()
        pytest.fail(f"DB schema creation failed in fixture: {e}")
    # --- Schema creation done --- #

    # Mock get_db_connection to *always* return the in-memory connection
    def mock_get_conn_always(path: Path):
        return conn
    monkeypatch.setattr(db_utils, 'get_db_connection', mock_get_conn_always)

    # Mock init_database to do nothing
    monkeypatch.setattr(db_utils, 'init_database', lambda *args, **kwargs: None)

    yield db_path_id, conn
    conn.close()

# Mock config dictionary for tests
@pytest.fixture
def mock_config():
    return {
        "paths": {"db": ":memory:"}, # Use memory path
        # Add other minimal config keys if needed by manager functions
        "cleanup": {}
    }

@pytest.mark.asyncio
async def test_start_session(in_memory_db_session, mock_config):
    """Test starting a new session."""
    db_path, _ = in_memory_db_session
    base_algo = "test_algo"

    session_info = await session_manager.start_session(mock_config, Path(db_path), base_algo)

    assert "session_id" in session_info
    assert session_info["base_algorithm"] == base_algo
    assert session_info["status"] == "active"
    assert isinstance(session_info["cycle_state"], dict)
    assert len(session_info["cycle_state"]) == 0 # Initial state is empty

    # Verify in DB
    row = await db_utils.db_fetch_one_async(db_path, "SELECT * FROM sessions WHERE session_id = ?", (session_info["session_id"],))
    assert row is not None
    assert row['base_algorithm'] == base_algo
    assert row['status'] == 'active'
    assert json.loads(row['cycle_state']) == {}

@pytest.mark.asyncio
async def test_get_session_info(in_memory_db_session, mock_config):
    """Test getting session information."""
    db_path, _ = in_memory_db_session
    base_algo = "test_algo_get"

    # Start a session first
    start_info = await session_manager.start_session(mock_config, Path(db_path), base_algo)
    session_id = start_info["session_id"]

    # Get the session info
    get_info = await session_manager.get_session_info(mock_config, Path(db_path), session_id)

    assert get_info["session_id"] == session_id
    assert get_info["base_algorithm"] == base_algo
    assert get_info["status"] == "active"
    assert isinstance(get_info["history"], list)
    assert isinstance(get_info["config"], dict)
    assert isinstance(get_info["cycle_state"], dict)

@pytest.mark.asyncio
async def test_get_session_info_not_found(in_memory_db_session, mock_config):
    """Test getting info for a non-existent session."""
    db_path, _ = in_memory_db_session
    with pytest.raises(ValueError, match="Session nonexistent_session が見つかりません"):
        await session_manager.get_session_info(mock_config, Path(db_path), "nonexistent_session")

@pytest.mark.asyncio
async def test_add_session_history(in_memory_db_session, mock_config):
    """Test adding events to session history."""
    db_path, _ = in_memory_db_session
    start_info = await session_manager.start_session(mock_config, Path(db_path), "hist_test")
    session_id = start_info["session_id"]

    event_type1 = "test_event_1"
    event_data1 = {"key": "value1"}
    updated_info_1 = await session_manager.add_session_history(mock_config, Path(db_path), session_id, event_type1, event_data1)

    assert len(updated_info_1["history"]) == 1
    assert updated_info_1["history"][0]["type"] == event_type1
    assert updated_info_1["history"][0]["data"] == event_data1 # Assumes data is serializable

    event_type2 = "test_event_2"
    event_data2 = {"count": 5}
    updated_info_2 = await session_manager.add_session_history(mock_config, Path(db_path), session_id, event_type2, event_data2)

    assert len(updated_info_2["history"]) == 2
    assert updated_info_2["history"][1]["type"] == event_type2

@pytest.mark.asyncio
async def test_add_session_history_with_cycle_update(in_memory_db_session, mock_config):
    """Test adding history with cycle_state update."""
    db_path, _ = in_memory_db_session
    start_info = await session_manager.start_session(mock_config, Path(db_path), "cycle_update_test")
    session_id = start_info["session_id"]

    update_data = {"iteration": 1, "best_f": 0.5}
    updated_info = await session_manager.add_session_history(
        mock_config, Path(db_path), session_id, "iteration_complete", {"f_measure": 0.5}, cycle_state_update=update_data
    )

    assert updated_info["cycle_state"]["iteration"] == 1
    assert updated_info["cycle_state"]["best_f"] == 0.5

    # Verify in DB
    row = await db_utils.db_fetch_one_async(db_path, "SELECT cycle_state FROM sessions WHERE session_id = ?", (session_id,))
    db_cycle_state = json.loads(row['cycle_state'])
    assert db_cycle_state["iteration"] == 1
    assert db_cycle_state["best_f"] == 0.5

# Add tests for cleanup_old_sessions if needed (requires manipulating timestamps) 