import os
import sqlite3
import logging
import asyncio
from pathlib import Path
import threading # threading をインポート (ロックのため)
from typing import Optional, List # Optional と List をインポート

logger = logging.getLogger('mcp_server.db')

# --- DB Constants and Setup ---
DB_FILENAME = "mcp_server_state.db"
# workspace_dir は config から取得する想定
# db_path = workspace_dir / "db" / DB_FILENAME

db_lock = threading.Lock() # DB書き込み用のロック

# DB接続プールのようなもの (ただし、ここでは単純な接続関数)
def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """データベース接続を取得"""
    conn = sqlite3.connect(str(db_path), timeout=10, check_same_thread=False) # check_same_thread=False はマルチスレッドアクセスに必要
    conn.row_factory = sqlite3.Row # カラム名でアクセスできるようにする
    logger.debug(f"DB connection to {db_path} obtained.")
    return conn

def init_database(config: dict):
    """データベースを初期化 (テーブル作成など)"""
    workspace_dir = Path(config.get('paths', {}).get('workspace', './mcp_workspace'))
    db_dir = workspace_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / DB_FILENAME
    logger.info(f"データベースを初期化中: {db_path}")

    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()

        # Sessions テーブル (改善セッションの状態)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            base_algorithm TEXT,
            start_time REAL,
            last_update REAL,
            status TEXT,  -- e.g., 'running', 'completed', 'failed'
            history TEXT,  -- JSON list of events
            config TEXT,   -- JSON of initial config for the session
            current_metrics TEXT, -- JSON of latest metrics
            best_metrics TEXT, -- JSON of best metrics achieved
            best_code_version TEXT, -- Version tag of the best performing code
            best_code_path TEXT, -- Path to the best performing code file
            baseline_metrics TEXT, -- JSON of baseline metrics
            cycle_state TEXT -- JSON representing the internal state of the improvement loop
        )
        """)

        # Jobs テーブル (非同期ジョブの状態)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            session_id TEXT, -- Optional: link to a session
            tool_name TEXT,
            status TEXT, -- 'pending', 'running', 'completed', 'failed'
            start_time REAL,
            end_time REAL,
            result TEXT, -- JSON serialized result or error message
            task_args TEXT, -- JSON serialized args for the task
            worker_id TEXT, -- ID of the worker that processed the job
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
        """)

        # 他に必要なテーブルがあればここに追加 (e.g., GridSearchResults, Evaluations)

        conn.commit()
        logger.info("データベースの初期化完了")
    except sqlite3.Error as e:
        logger.error(f"データベース初期化中にエラーが発生: {e}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()

# --- DB Helper Functions (Internal, Synchronous) ---

def _db_execute_commit(db_path: Path, sql: str, params: tuple = ()) -> Optional[int]:
    """DBに書き込み操作を実行してコミット (同期)"""
    conn = None
    last_row_id = None
    with db_lock:
        try:
            conn = get_db_connection(db_path)
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            last_row_id = cursor.lastrowid
            logger.debug(f"Executed and committed: {sql[:50]}... with params {params}")
        except sqlite3.Error as e:
            logger.error(f"DB Execute Error: {e} for SQL: {sql}", exc_info=True)
            if conn: conn.rollback()
            raise # エラーを再発生させて呼び出し元で処理
        finally:
            if conn: conn.close()
    return last_row_id

def _db_fetch_one(db_path: Path, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    """DBから一件のレコードを取得 (同期)"""
    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        row = cursor.fetchone()
        logger.debug(f"Fetched one: {sql[:50]}... with params {params}")
        return row
    except sqlite3.Error as e:
        logger.error(f"DB Fetch One Error: {e} for SQL: {sql}", exc_info=True)
        raise
    finally:
        if conn: conn.close()

def _db_fetch_all(db_path: Path, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    """DBから複数のレコードを取得 (同期)"""
    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        logger.debug(f"Fetched all: {sql[:50]}... with params {params}")
        return rows
    except sqlite3.Error as e:
        logger.error(f"DB Fetch All Error: {e} for SQL: {sql}", exc_info=True)
        raise
    finally:
        if conn: conn.close()


# --- Async DB Wrapper Functions ---
# これらの関数は asyncio の run_in_executor を使って同期関数を非同期にラップ

async def db_execute_commit_async(db_path: Path, sql: str, params: tuple = ()) -> Optional[int]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _db_execute_commit, db_path, sql, params)

async def db_fetch_one_async(db_path: Path, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _db_fetch_one, db_path, sql, params)

async def db_fetch_all_async(db_path: Path, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _db_fetch_all, db_path, sql, params) 