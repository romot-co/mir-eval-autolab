import pytest
import sqlite3
from pathlib import Path
import os
import time

# Assuming db_utils.py is in mcp_server_logic/
from mcp_server_logic import db_utils
from mcp_server_logic.utils import generate_id, get_timestamp # Assuming these utils exist

# --- Fixture for in-memory database ---

@pytest.fixture
def in_memory_db(monkeypatch):
    """Creates an in-memory SQLite database and sets up the schema directly."""
    db_path_id = ":memory:"

    # Create the in-memory connection
    conn = sqlite3.connect(db_path_id, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # --- Directly create schema in the in-memory DB --- #
    try:
        cursor = conn.cursor()
        # Copy table creation SQL from db_utils.init_database
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            base_algorithm TEXT,
            start_time REAL,
            last_update REAL,
            status TEXT,
            history TEXT,
            config TEXT,
            current_metrics TEXT,
            best_metrics TEXT,
            best_code_version TEXT,
            best_code_path TEXT,
            baseline_metrics TEXT,
            cycle_state TEXT
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            session_id TEXT,
            tool_name TEXT,
            status TEXT,
            start_time REAL,
            end_time REAL,
            result TEXT,
            error_details TEXT,
            task_args TEXT,
            worker_id TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
        """)
        # Add indices if necessary for tests, though likely not critical for in-memory
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_last_update ON sessions(last_update);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);")
        # ... add other indices if needed ...
        conn.commit()
    except Exception as e:
        conn.close()
        pytest.fail(f"DB schema creation failed in fixture: {e}")
    # --- Schema creation done --- #

    # Mock get_db_connection to *always* return the in-memory connection
    def mock_get_conn_always(path: Path):
        return conn
    monkeypatch.setattr(db_utils, 'get_db_connection', mock_get_conn_always)

    # Mock init_database to do nothing, as we created the schema manually
    monkeypatch.setattr(db_utils, 'init_database', lambda *args, **kwargs: None)

    yield db_path_id, conn # Provide identifier and connection

    # Cleanup
    conn.close()

# --- Test db_utils functions ---

@pytest.mark.asyncio
async def test_db_execute_commit_async(in_memory_db):
    """Test asynchronous DB execute and commit."""
    db_path, conn = in_memory_db
    test_id = generate_id()
    test_value = "test_data"

    # Test INSERT
    rows_affected = await db_utils.db_execute_commit_async(
        db_path,
        "INSERT INTO jobs (job_id, tool_name, status, request_data, created_at) VALUES (?, ?, ?, ?, ?)",
        (test_id, "test_tool", "pending", test_value, get_timestamp())
    )
    assert rows_affected == 1

    # Verify INSERT using synchronous fetch
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (test_id,))
    row = cursor.fetchone()
    assert row is not None
    assert row['job_id'] == test_id
    assert row['request_data'] == test_value

    # Test UPDATE
    new_status = "running"
    rows_affected_update = await db_utils.db_execute_commit_async(
        db_path,
        "UPDATE jobs SET status = ? WHERE job_id = ?",
        (new_status, test_id)
    )
    assert rows_affected_update == 1

    # Verify UPDATE
    cursor.execute("SELECT status FROM jobs WHERE job_id = ?", (test_id,))
    row_update = cursor.fetchone()
    assert row_update['status'] == new_status

@pytest.mark.asyncio
async def test_db_fetch_one_async(in_memory_db):
    """Test asynchronous DB fetch one."""
    db_path, conn = in_memory_db
    test_id = generate_id()
    test_tool = "fetch_one_test"

    # Insert test data
    await db_utils.db_execute_commit_async(
        db_path,
        "INSERT INTO jobs (job_id, tool_name, status, created_at) VALUES (?, ?, ?, ?)",
        (test_id, test_tool, "pending", get_timestamp())
    )

    # Fetch the inserted row
    row = await db_utils.db_fetch_one_async(db_path, "SELECT * FROM jobs WHERE job_id = ?", (test_id,))
    assert row is not None
    assert row['job_id'] == test_id
    assert row['tool_name'] == test_tool

    # Fetch non-existent row
    row_none = await db_utils.db_fetch_one_async(db_path, "SELECT * FROM jobs WHERE job_id = ?", ("nonexistent",))
    assert row_none is None

@pytest.mark.asyncio
async def test_db_fetch_all_async(in_memory_db):
    """Test asynchronous DB fetch all."""
    db_path, conn = in_memory_db
    ids = [generate_id() for _ in range(3)]
    tool_name = "fetch_all_test"

    # Insert multiple rows
    ts = get_timestamp()
    await db_utils.db_execute_commit_async(
        db_path,
        "INSERT INTO jobs (job_id, tool_name, status, created_at) VALUES (?, ?, ?, ?), (?, ?, ?, ?), (?, ?, ?, ?)",
        (ids[0], tool_name, "pending", ts,
         ids[1], tool_name, "running", ts,
         ids[2], tool_name, "completed", ts)
    )

    # Fetch all rows for this tool
    rows = await db_utils.db_fetch_all_async(db_path, "SELECT * FROM jobs WHERE tool_name = ? ORDER BY created_at", (tool_name,))
    assert len(rows) == 3
    assert rows[0]['job_id'] == ids[0]
    assert rows[1]['job_id'] == ids[1]
    assert rows[2]['job_id'] == ids[2]

    # Fetch with no results
    rows_none = await db_utils.db_fetch_all_async(db_path, "SELECT * FROM jobs WHERE tool_name = ?", ("nonexistent_tool",))
    assert len(rows_none) == 0

# Add tests for synchronous versions (_db_execute_commit, _db_fetch_one, _db_fetch_all) if needed
# Add tests for initialize_database (check if tables exist)
# Add tests for locking mechanism (might be complex) 