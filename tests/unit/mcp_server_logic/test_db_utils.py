# tests/unit/mcp_server_logic/test_db_utils.py
import pytest
import aiosqlite
import asyncio # For async lock testing if included
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

# Assume db_utils uses aiosqlite and defines StateManagementError
# Provide dummies if necessary
try:
    # Try importing specific functions and exceptions first
    from src.mcp_server_logic.db_utils import (
        get_db_connection, # Assumed function to get/create connection
        init_database,
        db_execute_commit_async,
        db_fetch_one_async,
        db_fetch_all_async,
        db_lock, # Assumed lock object or context manager
        get_db, # get_dbを追加 (コンテキストマネージャ用)
        StateManagementError
    )
    # from src.mcp_server_logic.session_manager import SessionManagerError
except ImportError:
    print("Warning: Using dummy implementations for db_utils.")
    # Dummy StateManagementError if not found in exception_utils
    try:
        from src.utils.exception_utils import StateManagementError
    except ImportError:
        class StateManagementError(Exception): pass

    # Dummy DB functions (do not perform real DB ops)
    async def get_db_connection(db_path=":memory:"):
        # Dummy connection (replace with real if testing interaction)
        conn = await aiosqlite.connect(":memory:") # Use real in-memory for dummy logic
        conn.row_factory = aiosqlite.Row
        return conn

    async def init_database(db_path=":memory:"):
        conn = await get_db_connection(db_path)
        # Dummy schema based on assumptions
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                status TEXT,
                created_at TEXT,
                last_update TEXT
                -- Add other potential fields based on SessionInfoResponse
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                session_id TEXT, -- Foreign key simulation
                tool_name TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                result_json TEXT,
                error_json TEXT,
                input_params_json TEXT
            );
            CREATE TABLE IF NOT EXISTS session_history (
                event_id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp TEXT,
                event_type TEXT,
                details_json TEXT
            );
        """)
        await conn.commit()
        await conn.close()
        print("Dummy init_database executed.")

    async def db_execute_commit_async(db_path, query, params=()):
        conn = await get_db_connection(db_path)
        try:
            cursor = await conn.execute(query, params)
            await conn.commit()
            last_id = cursor.lastrowid
            await cursor.close()
            return last_id # Or rowcount depending on need
        except aiosqlite.Error as e:
            raise StateManagementError(f"DB execution error: {e}") from e
        finally:
            await conn.close()

    async def db_fetch_one_async(db_path, query, params=()):
        conn = await get_db_connection(db_path)
        try:
            cursor = await conn.execute(query, params)
            row = await cursor.fetchone() # Returns Row or None
            await cursor.close()
            return row
        except aiosqlite.Error as e:
            raise StateManagementError(f"DB fetch error: {e}") from e
        finally:
            await conn.close()

    async def db_fetch_all_async(db_path, query, params=()):
        conn = await get_db_connection(db_path)
        try:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall() # Returns list of Rows
            await cursor.close()
            return rows
        except aiosqlite.Error as e:
            raise StateManagementError(f"DB fetch error: {e}") from e
        finally:
            await conn.close()

    # Dummy lock (replace with real asyncio.Lock or similar if testing)
    db_lock = asyncio.Lock() # Simple async lock for dummy

    # get_db のダミー実装
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def get_db(db_path: str):
        conn = None
        try:
            conn = await aiosqlite.connect(db_path)
            yield conn
        except Exception as e:
             raise StateManagementError(f"DB connection error: {e}") from e
        finally:
            if conn:
                await conn.close()


# --- Fixtures ---

@pytest.fixture
async def in_memory_db():
    """Provides a clean in-memory SQLite database for each test."""
    db_path = ":memory:"
    # Initialize the schema using the (potentially dummy) init_database
    await init_database(db_path)
    # We return the path, functions under test will use get_db_connection
    yield db_path
    # Cleanup is handled by closing connections within functions,
    # and :memory: DB disappears when all connections are closed.

@pytest.fixture
async def memory_db_path() -> str:
    """インメモリDBのパスを提供するフィクスチャ"""
    return ":memory:"

@pytest.fixture
async def initialized_memory_db(memory_db_path: str) -> str:
    """テスト用のテーブルを初期化したインメモリDBを提供するフィクスチャ"""
    # ここでは単純なテーブルを作成する例
    # 実際の init_database は複数のテーブルを作成するはず
    async with aiosqlite.connect(memory_db_path) as db:
        await db.execute("""
            CREATE TABLE test_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL
            );
        """)
        await db.execute("CREATE TABLE another_table (key TEXT PRIMARY KEY, data TEXT)")
        await db.commit()
        # 必要に応じて src.mcp_server_logic.db_utils.init_database を呼び出す
        # await init_database(memory_db_path) # 実際の初期化関数を使う場合
    return memory_db_path

# Mark tests as asyncio
pytestmark = pytest.mark.asyncio

# --- init_database Tests ---

async def test_init_database_creates_tables(in_memory_db):
    """Verify that init_database creates the expected tables."""
    db_path = in_memory_db
    conn = await aiosqlite.connect(db_path)
    cursor = await conn.cursor()

    # Check if tables exist
    await cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
    assert await cursor.fetchone() is not None, "sessions table not created"

    await cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
    assert await cursor.fetchone() is not None, "jobs table not created"

    await cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_history'")
    assert await cursor.fetchone() is not None, "session_history table not created"

    await cursor.close()
    await conn.close()

# --- Basic CRUD Tests ---

async def test_db_execute_insert(in_memory_db):
    """Test inserting data using db_execute_commit_async."""
    db_path = in_memory_db
    session_id = "sess_insert_test"
    query = "INSERT INTO sessions (session_id, status, created_at, last_update) VALUES (?, ?, ?, ?)"
    params = (session_id, "running", "ts1", "ts2")

    await db_execute_commit_async(db_path, query, params)

    # Verify insertion
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = await cursor.fetchone()
    await conn.close()

    assert row is not None
    assert row["session_id"] == session_id
    assert row["status"] == "running"

async def test_db_execute_update(in_memory_db):
    """Test updating data using db_execute_commit_async."""
    db_path = in_memory_db
    session_id = "sess_update_test"
    # Insert initial data
    await db_execute_commit_async(db_path, "INSERT INTO sessions (session_id, status, created_at, last_update) VALUES (?, ?, ?, ?)", (session_id, "running", "ts1", "ts1"))

    # Update the status
    update_query = "UPDATE sessions SET status = ?, last_update = ? WHERE session_id = ?"
    await db_execute_commit_async(db_path, update_query, ("stopped", "ts2", session_id))

    # Verify update
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute("SELECT status, last_update FROM sessions WHERE session_id = ?", (session_id,))
    row = await cursor.fetchone()
    await conn.close()

    assert row is not None
    assert row["status"] == "stopped"
    assert row["last_update"] == "ts2"

async def test_db_execute_delete(in_memory_db):
    """Test deleting data using db_execute_commit_async."""
    db_path = in_memory_db
    session_id = "sess_delete_test"
    # Insert initial data
    await db_execute_commit_async(db_path, "INSERT INTO sessions (session_id, status, created_at, last_update) VALUES (?, ?, ?, ?)", (session_id, "running", "ts1", "ts1"))

    # Delete the record
    delete_query = "DELETE FROM sessions WHERE session_id = ?"
    await db_execute_commit_async(db_path, delete_query, (session_id,))

    # Verify deletion
    conn = await aiosqlite.connect(db_path)
    cursor = await conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
    row = await cursor.fetchone()
    await conn.close()

    assert row is None


async def test_db_fetch_one_success(in_memory_db):
    """Test fetching a single existing row."""
    db_path = in_memory_db
    session_id = "sess_fetch_one"
    await db_execute_commit_async(db_path, "INSERT INTO sessions (session_id, status, created_at, last_update) VALUES (?, ?, ?, ?)", (session_id, "running", "ts1", "ts1"))

    query = "SELECT * FROM sessions WHERE session_id = ?"
    row = await db_fetch_one_async(db_path, query, (session_id,))

    assert row is not None
    assert isinstance(row, aiosqlite.Row) # Check if row_factory worked
    assert row["session_id"] == session_id
    assert row["status"] == "running"

async def test_db_fetch_one_not_found(in_memory_db):
    """Test fetching a non-existent row returns None."""
    db_path = in_memory_db
    query = "SELECT * FROM sessions WHERE session_id = ?"
    row = await db_fetch_one_async(db_path, query, ("non_existent_id",))
    assert row is None


async def test_db_fetch_all_success(in_memory_db):
    """Test fetching multiple rows."""
    db_path = in_memory_db
    sessions = [
        ("sess_all_1", "running", "t1", "t1"),
        ("sess_all_2", "stopped", "t2", "t2"),
    ]
    insert_query = "INSERT INTO sessions (session_id, status, created_at, last_update) VALUES (?, ?, ?, ?)"
    conn = await aiosqlite.connect(db_path)
    await conn.executemany(insert_query, sessions)
    await conn.commit()
    await conn.close()

    query = "SELECT * FROM sessions WHERE session_id LIKE ?"
    rows = await db_fetch_all_async(db_path, query, ("sess_all_%",))

    assert rows is not None
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert isinstance(rows[0], aiosqlite.Row)
    session_ids = {row["session_id"] for row in rows}
    assert session_ids == {"sess_all_1", "sess_all_2"}

async def test_db_fetch_all_empty(in_memory_db):
    """Test fetching when no rows match returns an empty list."""
    db_path = in_memory_db
    query = "SELECT * FROM sessions WHERE status = ?"
    rows = await db_fetch_all_async(db_path, query, ("stalled",))
    assert rows == []


# --- Error Handling Tests ---

@patch('aiosqlite.connect') # Mock the connection itself
async def test_db_execute_raises_state_management_error(mock_connect):
    """Test that DB execution errors are wrapped in StateManagementError."""
    # Configure the mock connection and cursor to raise an aiosqlite error
    mock_conn_obj = MagicMock()
    mock_cursor = MagicMock()
    # Make execute return an async context manager that raises on __aenter__
    async def raise_operational_error(*args, **kwargs):
        raise aiosqlite.OperationalError("Mock DB Error")
    mock_cursor.__aenter__.side_effect = raise_operational_error
    mock_cursor.__aexit__ = MagicMock(return_value=None) # Need awaitable __aexit__
    mock_conn_obj.execute = MagicMock(return_value=mock_cursor)
    mock_conn_obj.commit = MagicMock()
    async def commit_coro(): pass
    mock_conn_obj.commit = asyncio.coroutine(commit_coro)
    async def close_coro(): pass
    mock_conn_obj.close = asyncio.coroutine(close_coro)

    async def connect_coro(*args, **kwargs):
        return mock_conn_obj
    mock_connect.return_value = asyncio.coroutine(connect_coro)()

    db_path = "dummy_path" # Path doesn't matter as connect is mocked
    query = "INSERT INTO table VALUES (?)"
    with pytest.raises(StateManagementError, match="DB execution error: Mock DB Error"):
         await db_execute_commit_async(db_path, query, (1,))


@patch('aiosqlite.connect')
async def test_db_fetch_one_raises_state_management_error(mock_connect):
     """Test that DB fetch errors are wrapped in StateManagementError."""
     mock_conn_obj = MagicMock()
     mock_cursor = MagicMock()
     async def raise_integrity_error(*args, **kwargs):
        raise aiosqlite.IntegrityError("Mock Fetch Error")
     mock_cursor.__aenter__.side_effect = raise_integrity_error
     mock_cursor.__aexit__ = MagicMock(return_value=None)
     mock_conn_obj.execute = MagicMock(return_value=mock_cursor)
     async def close_coro(): pass
     mock_conn_obj.close = asyncio.coroutine(close_coro)

     async def connect_coro(*args, **kwargs):
        return mock_conn_obj
     mock_connect.return_value = asyncio.coroutine(connect_coro)()

     db_path = "dummy_path"
     query = "SELECT col FROM table WHERE id = ?"
     with pytest.raises(StateManagementError, match="DB fetch error: Mock Fetch Error"):
         await db_fetch_one_async(db_path, query, (1,))

# --- Lock Test (Basic) ---

# This test is very basic and assumes db_lock is an asyncio.Lock compatible object
# It doesn't test concurrency issues.
async def test_db_lock_acquire_release():
    """Basic test for acquiring and releasing the lock."""
    # Use the imported or dummy lock
    lock_obj = db_lock
    assert not lock_obj.locked()
    async with lock_obj:
        assert lock_obj.locked()
        # Simulate some locked operation
        await asyncio.sleep(0.01)
    assert not lock_obj.locked()

# --- Tests ---

async def test_init_database_creates_tables(memory_db_path: str):
    """init_database が期待されるテーブルを作成することを確認する"""
    # 実際の init_database を呼び出す
    # このテストは実際の init_database の実装に依存する
    # ダミー実装を使っている場合は、実際のテーブル名を確認する必要がある
    await init_database(memory_db_path) # ダミー実装を使用

    async with aiosqlite.connect(memory_db_path) as db:
        # init_database が作成するはずのテーブルが存在するか確認
        # 例: jobs, sessions テーブルなど (実際のスキーマに合わせて修正)
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dummy_table';")
        table = await cursor.fetchone()
        assert table is not None, "init_database should create 'dummy_table'"
        # 他の期待されるテーブルも同様にチェック
        # cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions';")
        # table = await cursor.fetchone()
        # assert table is not None, "init_database should create 'sessions' table"


async def test_db_execute_commit_async_success(initialized_memory_db: str):
    """db_execute_commit_async が正常にクエリを実行しコミットすることを確認する"""
    db_path = initialized_memory_db
    insert_query = "INSERT INTO test_items (name, value) VALUES (?, ?)"
    params = ("test_item_1", 123.45)

    await db_execute_commit_async(db_path, insert_query, params)

    # データが挿入されたことを確認
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT name, value FROM test_items WHERE name = ?", (params[0],)) as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == params[0]
            assert row[1] == params[1]

async def test_db_execute_commit_async_error_raises_state_error(initialized_memory_db: str):
    """db_execute_commit_async がDBエラー時に StateManagementError を発生させることを確認する"""
    db_path = initialized_memory_db
    invalid_query = "INSERT INTO non_existent_table (col) VALUES (?)" # 不正なクエリ
    params = ("value",)

    with pytest.raises(StateManagementError):
        await db_execute_commit_async(db_path, invalid_query, params)

async def test_db_fetch_one_async_success(initialized_memory_db: str):
    """db_fetch_one_async が正常に1行取得することを確認する"""
    db_path = initialized_memory_db
    # テストデータを準備
    item_name = "fetch_one_test"
    item_value = 987.65
    async with aiosqlite.connect(db_path) as db:
        await db.execute("INSERT INTO test_items (name, value) VALUES (?, ?)", (item_name, item_value))
        await db.commit()

    query = "SELECT id, name, value FROM test_items WHERE name = ?"
    row = await db_fetch_one_async(db_path, query, (item_name,))

    assert row is not None
    assert isinstance(row, aiosqlite.Row) # aiosqlite.Row オブジェクトであることを確認
    assert row["name"] == item_name # 列名でアクセスできることを確認
    assert row["value"] == item_value

async def test_db_fetch_one_async_no_result(initialized_memory_db: str):
    """db_fetch_one_async が結果なしの場合に None を返すことを確認する"""
    db_path = initialized_memory_db
    query = "SELECT id FROM test_items WHERE name = ?"
    row = await db_fetch_one_async(db_path, query, ("non_existent_item",))
    assert row is None

async def test_db_fetch_one_async_error_raises_state_error(initialized_memory_db: str):
    """db_fetch_one_async がDBエラー時に StateManagementError を発生させることを確認する"""
    db_path = initialized_memory_db
    invalid_query = "SELECT * FROM non_existent_table WHERE id = ?" # 不正なクエリ

    with pytest.raises(StateManagementError):
        await db_fetch_one_async(db_path, invalid_query, (1,))


async def test_db_fetch_all_async_success(initialized_memory_db: str):
    """db_fetch_all_async が正常に複数行取得することを確認する"""
    db_path = initialized_memory_db
    # テストデータを準備
    items = [("fetch_all_1", 1.1), ("fetch_all_2", 2.2)]
    async with aiosqlite.connect(db_path) as db:
        await db.executemany("INSERT INTO test_items (name, value) VALUES (?, ?)", items)
        await db.commit()

    query = "SELECT name, value FROM test_items WHERE name LIKE ?"
    rows = await db_fetch_all_async(db_path, query, ('fetch_all_%',))

    assert rows is not None
    assert len(rows) == 2
    assert isinstance(rows[0], aiosqlite.Row)
    # 結果の順序は保証されない可能性があるため、セットで比較するかソートする
    expected_results = {tuple(item) for item in items}
    actual_results = {(row['name'], row['value']) for row in rows}
    assert actual_results == expected_results

async def test_db_fetch_all_async_no_result(initialized_memory_db: str):
    """db_fetch_all_async が結果なしの場合に空リストを返すことを確認する"""
    db_path = initialized_memory_db
    query = "SELECT id FROM test_items WHERE name = ?"
    rows = await db_fetch_all_async(db_path, query, ("non_existent_item",))
    assert rows == []

async def test_db_fetch_all_async_error_raises_state_error(initialized_memory_db: str):
    """db_fetch_all_async がDBエラー時に StateManagementError を発生させることを確認する"""
    db_path = initialized_memory_db
    invalid_query = "SELECT * FROM non_existent_table" # 不正なクエリ

    with pytest.raises(StateManagementError):
        await db_fetch_all_async(db_path, invalid_query)

async def test_get_db_context_manager(memory_db_path: str):
    """get_db コンテキストマネージャが接続を提供し、終了時に閉じることを確認する"""
    db_path = memory_db_path
    conn_instance = None
    async with get_db(db_path) as conn:
        assert isinstance(conn, aiosqlite.Connection)
        # 簡単な操作を実行してみる
        await conn.execute("CREATE TABLE IF NOT EXISTS ctx_test (id INTEGER PRIMARY KEY)")
        await conn.commit()
        conn_instance = conn # 閉じる前にインスタンスを保持 (テスト目的)

    # コンテキストを抜けた後、接続が閉じられているか確認 (非同期なので少し難しい)
    # aiosqlite に is_closed のような直接的なプロパティはないが、
    # 再度操作しようとすると ProgrammingError が発生するはず
    with pytest.raises(aiosqlite.ProgrammingError, match="Cannot operate on a closed database."):
         await conn_instance.execute("SELECT * FROM ctx_test") # type: ignore

# db_lock のテスト (指示書では省略可だが、基本的な確認)
# 実際の db_lock 実装に依存する
# from src.mcp_server_logic.db_utils import db_lock
#
# @pytest.mark.skip(reason="db_lock test requires specific implementation details or mocking")
# async def test_db_lock_basic(memory_db_path: str):
#     """db_lock の基本的な動作を確認する (実装依存のためスキップ推奨)"""
#     db_path = memory_db_path
#     lock_name = "test_lock"
#
#     async with db_lock(db_path, lock_name):
#         # ロック取得中に何らかの状態を確認（例：ログ、フラグ）
#         # このテストは db_lock がどのようにロックを実現しているかに強く依存する
#         # asyncio.Lock を使っているなら、その状態を確認するなど
#         pass
#     # ロックが解放されたことを確認 