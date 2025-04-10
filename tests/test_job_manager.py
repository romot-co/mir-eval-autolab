import pytest
import asyncio
import time
from pathlib import Path
import json
import sqlite3

# Assuming job_manager and core modules are in mcp_server_logic/
from mcp_server_logic import job_manager, core, db_utils
from mcp_server_logic.utils import generate_id, get_timestamp

# Re-use the in-memory DB fixture
@pytest.fixture
def in_memory_db_job(monkeypatch):
    """Fixture for in-memory DB for job tests."""
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

# Mock config for job manager tests
@pytest.fixture
def mock_config_job():
    return {
        "paths": {"db": ":memory:"},
        "resource_limits": {"max_jobs_history": 100}, # For cleanup test
        "cleanup": {"job_completed_retention_seconds": 3600, "job_stuck_timeout_seconds": 3600, "max_jobs_count": 100}
    }

# Dummy async task for testing
async def dummy_task(job_id: str, delay: float = 0.1, should_fail: bool = False):
    """A simple async task for testing the job worker."""
    print(f"Dummy task {job_id} started.")
    await asyncio.sleep(delay)
    if should_fail:
        print(f"Dummy task {job_id} failing intentionally.")
        raise ValueError("Intentional failure")
    print(f"Dummy task {job_id} finished.")
    return {"job_id": job_id, "result": "success", "slept_for": delay}

@pytest.mark.asyncio
async def test_start_async_job(in_memory_db_job, mock_config_job):
    """Test registering a new job."""
    db_path, _ = in_memory_db_job
    tool_name = "test_tool_start"
    session_id = "test_session"
    request_data = {"param": "value"}

    # Clear the queue and active jobs for isolated test
    job_manager.job_queue = asyncio.Queue() # Use the one from job_manager
    job_manager.active_jobs = {} # Use the one from job_manager

    # Use the start_async_job from core.py
    job_id = core.start_async_job(
        config=mock_config_job,
        task_coroutine_func=dummy_task, # Pass the coroutine function itself
        tool_name=tool_name,
        session_id=session_id,
        # Other args for dummy_task
        delay=0.01
    )

    assert isinstance(job_id, str)
    assert job_id in job_manager.active_jobs
    assert job_manager.active_jobs[job_id]["status"] == "pending"
    assert job_manager.active_jobs[job_id]["tool_name"] == tool_name
    assert job_manager.active_jobs[job_id]["session_id"] == session_id

    # Check queue
    assert not job_manager.job_queue.empty()
    queued_job_id, _, _, queued_kwargs = await job_manager.job_queue.get()
    assert queued_job_id == job_id
    assert queued_kwargs['tool_name'] == tool_name
    assert queued_kwargs['session_id'] == session_id
    assert queued_kwargs['delay'] == 0.01

    # Check DB
    row = await db_utils.db_fetch_one_async(db_path, "SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    assert row is not None
    assert row['tool_name'] == tool_name
    assert row['status'] == 'pending'
    assert row['session_id'] == session_id

@pytest.mark.asyncio
async def test_get_job_status(in_memory_db_job, mock_config_job):
    """Test retrieving job status."""
    db_path, _ = in_memory_db_job
    job_manager.active_jobs = {} # Clear active jobs

    # Test case 1: Job pending in DB
    job_id_pending = generate_id()
    await db_utils.db_execute_commit_async(
        db_path,
        "INSERT INTO jobs (job_id, tool_name, status, created_at) VALUES (?, ?, ?, ?)",
        (job_id_pending, "tool1", "pending", get_timestamp())
    )
    status_pending = await job_manager.get_job_status(db_path, job_id_pending)
    assert status_pending["status"] == "pending"
    assert status_pending["job_id"] == job_id_pending

    # Test case 2: Job active in memory
    job_id_active = generate_id()
    job_manager.active_jobs[job_id_active] = {
        "status": "running",
        "tool_name": "tool2",
        "session_id": "sess2",
        "created_at": get_timestamp(),
        "start_time": get_timestamp(),
        "worker_id": 1
    }
    status_active = await job_manager.get_job_status(db_path, job_id_active)
    assert status_active["status"] == "running"
    assert status_active["worker_id"] == 1

    # Test case 3: Job completed in DB
    job_id_completed = generate_id()
    result_data = {"output": "done"}
    await db_utils.db_execute_commit_async(
        db_path,
        "INSERT INTO jobs (job_id, tool_name, status, created_at, end_time, result) VALUES (?, ?, ?, ?, ?, ?)",
        (job_id_completed, "tool3", "completed", get_timestamp(), get_timestamp(), json.dumps(result_data))
    )
    status_completed = await job_manager.get_job_status(db_path, job_id_completed)
    assert status_completed["status"] == "completed"
    assert status_completed["result"] == result_data # Should be deserialized

    # Test case 4: Job failed in DB
    job_id_failed = generate_id()
    error_data = {"error": "it broke", "traceback": "..."}
    await db_utils.db_execute_commit_async(
        db_path,
        "INSERT INTO jobs (job_id, tool_name, status, created_at, end_time, result) VALUES (?, ?, ?, ?, ?, ?)",
        (job_id_failed, "tool4", "failed", get_timestamp(), get_timestamp(), json.dumps(error_data))
    )
    status_failed = await job_manager.get_job_status(db_path, job_id_failed)
    assert status_failed["status"] == "failed"
    assert status_failed["error"] == error_data["error"]
    assert status_failed["traceback"] == error_data["traceback"]

    # Test case 5: Job not found
    status_not_found = await job_manager.get_job_status(db_path, "nonexistent_job")
    assert status_not_found["status"] == "not_found"
    assert status_not_found["error"] is not None

# Add tests for job_worker (more complex, requires running the worker)
# Add tests for cleanup_old_jobs 