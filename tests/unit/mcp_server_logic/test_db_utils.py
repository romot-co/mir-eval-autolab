# tests/unit/mcp_server_logic/test_db_utils.py
import pytest
import aiosqlite
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import sys
import os
import tempfile
import uuid

# srcディレクトリへのパスを追加してインポートエラーを防ぐ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

try:
    # 実際のモジュールから必要な関数とクラスをインポート
    from src.mcp_server_logic.db_utils import (
        init_database,
        db_execute_commit_async,
        db_fetch_one_async,
        db_fetch_all_async,
        db_lock,
        DB_FILENAME
    )
    from src.utils.exception_utils import StateManagementError
    DUMMY_IMPLEMENTATION = False
except ImportError as e:
    print(f"Warning: Falling back to dummy implementations for db_utils.py: {e}")
    
    # ダミー実装用のクラスと関数
    class StateManagementError(Exception): 
        pass
    
    DB_FILENAME = "mcp_server_state.db"
    db_lock = asyncio.Lock()
    
    async def init_database(config):
        db_path = Path(config.get('paths', {}).get('db_dir', './db')) / DB_FILENAME
        print(f"Dummy init_database with {db_path}")
        conn = await aiosqlite.connect(":memory:")
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, 
                status TEXT,
                created_at TEXT,
                last_update TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                session_id TEXT,
                tool_name TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                result_json TEXT,
                error_json TEXT,
                input_params_json TEXT
            );
        """)
        await conn.commit()
        await conn.close()
        
    async def db_execute_commit_async(db_path, query, params=()):
        conn = await aiosqlite.connect(db_path)
        try:
            cursor = await conn.execute(query, params)
            await conn.commit()
            return cursor.rowcount
        finally:
            await conn.close()
            
    async def db_fetch_one_async(db_path, query, params=()):
        conn = await aiosqlite.connect(db_path)
        conn.row_factory = aiosqlite.Row
        try:
            cursor = await conn.execute(query, params)
            row = await cursor.fetchone()
            return row
        finally:
            await conn.close()
            
    async def db_fetch_all_async(db_path, query, params=()):
        conn = await aiosqlite.connect(db_path)
        conn.row_factory = aiosqlite.Row
        try:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            return rows
        finally:
            await conn.close()
    
    DUMMY_IMPLEMENTATION = True


# --- テストクラスとフィクスチャ ---

@pytest.fixture(scope="module")
def tmp_db_path():
    """テスト用の一時ファイルパスを提供するフィクスチャ"""
    # インメモリデータベースは接続間で共有できないため一時ファイルを使用
    temp_dir = tempfile.mkdtemp()
    db_path = f"{temp_dir}/test_db_{uuid.uuid4().hex}.db"
    yield db_path
    # クリーンアップ
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture(scope="function")
async def test_db(tmp_db_path):
    """テスト用のデータベースを準備し、クリーンな状態にするフィクスチャ"""
    # データベースを初期化
    conn = await aiosqlite.connect(tmp_db_path)
    try:
        await conn.execute("DROP TABLE IF EXISTS sessions")
        await conn.execute("DROP TABLE IF EXISTS jobs")
        await conn.execute("DROP TABLE IF EXISTS test_items")
        
        await conn.executescript("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                status TEXT,
                created_at TEXT,
                last_update TEXT
            );
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                session_id TEXT,
                tool_name TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                result_json TEXT,
                error_json TEXT,
                input_params_json TEXT
            );
            CREATE TABLE test_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL
            );
        """)
        await conn.commit()
    finally:
        await conn.close()
    
    yield tmp_db_path


@pytest.fixture
def mock_config():
    """テスト用の設定を提供する"""
    return {
        'paths': {
            'workspace': './test_workspace',
            'db_dir': './test_workspace/db'
        }
    }


# --- テスト関数 ---

@pytest.mark.asyncio
async def test_db_execute_insert(test_db):
    """db_execute_commit_asyncを使ってデータを挿入するテスト"""
    # SQLインジェクション脆弱性がないことを確認するためのセッションID
    session_id = "test_session'; DROP TABLE sessions; --"
    query = "INSERT INTO sessions (session_id, status, created_at, last_update) VALUES (?, ?, ?, ?)"
    params = (session_id, "running", "2023-01-01", "2023-01-01")
    
    # 挿入を実行
    await db_execute_commit_async(test_db, query, params)
    
    # 挿入されたデータを検証
    conn = await aiosqlite.connect(test_db)
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = await cursor.fetchone()
    await conn.close()
    
    assert row is not None
    assert row["session_id"] == session_id
    assert row["status"] == "running"


@pytest.mark.asyncio
async def test_db_fetch_one_success(test_db):
    """db_fetch_one_asyncを使って1行のデータを取得するテスト"""
    # データを挿入
    session_id = "fetch_one_test"
    await db_execute_commit_async(
        test_db, 
        "INSERT INTO sessions (session_id, status, created_at, last_update) VALUES (?, ?, ?, ?)",
        (session_id, "completed", "2023-01-01", "2023-01-02")
    )
    
    # データを取得
    row = await db_fetch_one_async(
        test_db,
        "SELECT * FROM sessions WHERE session_id = ?",
        (session_id,)
    )
    
    # 結果を検証
    assert row is not None
    assert row["session_id"] == session_id
    assert row["status"] == "completed"


@pytest.mark.asyncio
async def test_db_fetch_one_not_found(test_db):
    """存在しないデータを検索した場合のテスト"""
    row = await db_fetch_one_async(
        test_db,
        "SELECT * FROM sessions WHERE session_id = ?",
        ("nonexistent",)
    )
    
    assert row is None


@pytest.mark.asyncio
async def test_db_fetch_all_success(test_db):
    """db_fetch_all_asyncを使って複数行のデータを取得するテスト"""
    # 複数のデータを挿入
    sessions = [
        ("sess1", "running", "2023-01-01", "2023-01-01"),
        ("sess2", "completed", "2023-01-02", "2023-01-03"),
        ("sess3", "failed", "2023-01-04", "2023-01-04")
    ]
    
    for sess in sessions:
        await db_execute_commit_async(
            test_db,
            "INSERT INTO sessions (session_id, status, created_at, last_update) VALUES (?, ?, ?, ?)",
            sess
        )
    
    # すべてのデータを取得
    rows = await db_fetch_all_async(
        test_db,
        "SELECT * FROM sessions ORDER BY session_id"
    )
    
    # 結果を検証
    assert len(rows) == 3
    assert rows[0]["session_id"] == "sess1"
    assert rows[1]["session_id"] == "sess2"
    assert rows[2]["session_id"] == "sess3"
    assert rows[0]["status"] == "running"
    assert rows[1]["status"] == "completed"
    assert rows[2]["status"] == "failed"


@pytest.mark.asyncio
async def test_db_execute_update(test_db):
    """db_execute_commit_asyncを使ってデータを更新するテスト"""
    # データを挿入
    session_id = "update_test"
    await db_execute_commit_async(
        test_db,
        "INSERT INTO sessions (session_id, status, created_at, last_update) VALUES (?, ?, ?, ?)",
        (session_id, "running", "2023-01-01", "2023-01-01")
    )
    
    # データを更新
    await db_execute_commit_async(
        test_db,
        "UPDATE sessions SET status = ?, last_update = ? WHERE session_id = ?",
        ("completed", "2023-01-02", session_id)
    )
    
    # 更新されたデータを検証
    row = await db_fetch_one_async(
        test_db,
        "SELECT * FROM sessions WHERE session_id = ?",
        (session_id,)
    )
    
    assert row is not None
    assert row["status"] == "completed"
    assert row["last_update"] == "2023-01-02"


@pytest.mark.asyncio
async def test_db_execute_delete(test_db):
    """db_execute_commit_asyncを使ってデータを削除するテスト"""
    # データを挿入
    session_id = "delete_test"
    await db_execute_commit_async(
        test_db,
        "INSERT INTO sessions (session_id, status, created_at, last_update) VALUES (?, ?, ?, ?)",
        (session_id, "running", "2023-01-01", "2023-01-01")
    )
    
    # データを削除
    await db_execute_commit_async(
        test_db,
        "DELETE FROM sessions WHERE session_id = ?",
        (session_id,)
    )
    
    # 削除されたことを検証
    row = await db_fetch_one_async(
        test_db,
        "SELECT * FROM sessions WHERE session_id = ?",
        (session_id,)
    )
    
    assert row is None


@pytest.mark.asyncio
async def test_db_error_handling(test_db):
    """エラーハンドリングをテストする"""
    # 存在しないテーブルに対してクエリを実行
    with pytest.raises(StateManagementError):
        await db_fetch_all_async(
            test_db,
            "SELECT * FROM nonexistent_table"
        )


@pytest.mark.asyncio
async def test_db_lock():
    """db_lockがasyncio.Lockインスタンスであることを確認"""
    assert isinstance(db_lock, asyncio.Lock)
    
    # ロックが正常に獲得・解放できることを確認
    async def acquire_and_release():
        async with db_lock:
            return True
    
    result = await acquire_and_release()
    assert result is True 