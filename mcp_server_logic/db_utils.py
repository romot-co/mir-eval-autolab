import os
import sqlite3
import logging
import asyncio
import threading # threading をインポート
from pathlib import Path
from typing import Optional, List # Optional と List をインポート

# StateManagementError をインポート
from src.utils.exception_utils import StateManagementError

logger = logging.getLogger('mcp_server.db')

# --- DB Constants and Setup ---
DB_FILENAME = "mcp_server_state.db"
# workspace_dir は config から取得する想定
# db_path = workspace_dir / "db" / DB_FILENAME

# db_lock = threading.Lock() # DB書き込み用のロック <- asyncio.Lock に変更
db_lock = asyncio.Lock() # 非同期対応のDBアクセスロック
_sync_db_lock = threading.Lock() # 同期処理用のDBアクセスロック

# DB接続プールのようなもの (ただし、ここでは単純な接続関数)
def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """データベース接続を取得 (同期)

    注意: この関数自体は同期ですが、非同期コンテキスト (run_in_executor)
    から呼び出されることを想定しています。
    check_same_thread=False はそのためです。
    """
    try:
        # タイムアウトを少し長く設定
        conn = sqlite3.connect(str(db_path), timeout=20.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row # カラム名でアクセスできるようにする
        conn.execute("PRAGMA journal_mode=WAL;") # WALモードを有効化 (同時読み書き性能向上)
        conn.execute("PRAGMA busy_timeout = 10000;") # 競合時の待機時間を設定 (10秒)
        logger.debug(f"DB connection to {db_path} obtained.")
        return conn
    except sqlite3.Error as db_err:
        logger.error(f"Failed to get DB connection to {db_path}: {db_err}", exc_info=True)
        raise StateManagementError(f"DB接続エラー: {db_err}") from db_err

def init_database(config: dict):
    """データベースを初期化 (テーブル作成など) (同期)"""
    workspace_dir = Path(config.get('paths', {}).get('workspace', './mcp_workspace'))
    db_dir = workspace_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / DB_FILENAME
    logger.info(f"データベースを初期化中: {db_path}")
    conn = None # finally節のために先に宣言
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
            error_details TEXT, # エラー詳細フィールド追加 (result と分ける)
            task_args TEXT, -- JSON serialized args for the task
            worker_id TEXT, -- ID of the worker that processed the job
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
        """)

        # インデックス追加 (パフォーマンス向上のため)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_last_update ON sessions(last_update);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_start_time ON jobs(start_time);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_end_time ON jobs(end_time);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_session_id ON jobs(session_id);")

        conn.commit()
        logger.info("データベースの初期化完了")
    except (sqlite3.Error, StateManagementError) as db_err: # StateManagementErrorもキャッチ
        logger.error(f"データベース初期化中にエラーが発生: {db_err}", exc_info=True)
        if conn: conn.rollback()
        # StateManagementError はそのまま、sqlite3.Error はラップして raise
        if isinstance(db_err, StateManagementError):
            raise db_err
        else:
            raise StateManagementError(f"DB初期化エラー: {db_err}") from db_err
    finally:
        if conn:
            conn.close()

# --- DB Helper Functions (Internal, Synchronous) ---
# 注意: これらの同期ヘルパーは async db_lock を直接使えない。
# 代わりに _sync_db_lock (threading.Lock) を使用する。

def _db_execute_commit_sync(db_path: Path, sql: str, params: tuple = ()) -> Optional[int]:
    """DBに書き込み操作を実行してコミット (同期ヘルパー)
    呼び出し元で asyncio.Lock() による排他制御がされている前提。
    """
    conn = None
    last_row_id = None
    with _sync_db_lock: # 同期ロックで保護
        try:
            conn = get_db_connection(db_path)
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            last_row_id = cursor.lastrowid
            logger.debug(f"Sync Executed and committed: {sql[:50]}... with params {params}")
        except sqlite3.Error as db_err:
            logger.error(f"Sync DB Execute Error: {db_err} for SQL: {sql}", exc_info=True)
            if conn: conn.rollback()
            raise StateManagementError(f"DB実行エラー (同期): {db_err}") from db_err
        finally:
            if conn: conn.close()
    return last_row_id

def _db_fetch_one_sync(db_path: Path, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    """DBから一件のレコードを取得 (同期ヘルパー)
    読み取り操作なのでロックは必須ではないが、WALモードなら書き込みと並行可能。
    """
    conn = None
    with _sync_db_lock: # 同期ロックで保護 (読み取りでも安全のため)
        try:
            conn = get_db_connection(db_path)
            cursor = conn.cursor()
            cursor.execute(sql, params)
            row = cursor.fetchone()
            logger.debug(f"Sync Fetched one: {sql[:50]}... with params {params}")
            return row
        except sqlite3.Error as db_err:
            logger.error(f"Sync DB Fetch One Error: {db_err} for SQL: {sql}", exc_info=True)
            raise StateManagementError(f"DB読み取りエラー (一件・同期): {db_err}") from db_err
        finally:
            if conn: conn.close()

def _db_fetch_all_sync(db_path: Path, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    """DBから複数のレコードを取得 (同期ヘルパー)
    読み取り操作なのでロックは必須ではないが、WALモードなら書き込みと並行可能。
    """
    conn = None
    with _sync_db_lock: # 同期ロックで保護 (読み取りでも安全のため)
        try:
            conn = get_db_connection(db_path)
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            logger.debug(f"Sync Fetched all: {sql[:50]}... with params {params}")
            return rows
        except sqlite3.Error as db_err:
            logger.error(f"Sync DB Fetch All Error: {db_err} for SQL: {sql}", exc_info=True)
            raise StateManagementError(f"DB読み取りエラー (複数・同期): {db_err}") from db_err
        finally:
            if conn: conn.close()


# --- Async DB Wrapper Functions ---
# これらの関数は asyncio の run_in_executor を使って同期ヘルパーを非同期に呼び出し、
# 書き込み操作には async with db_lock: を適用する。

async def db_execute_commit_async(db_path: Path, sql: str, params: tuple = ()) -> Optional[int]:
    """DB書き込み操作を非同期で実行 (ロック付き)"""
    loop = asyncio.get_running_loop()
    async with db_lock: # ここで非同期ロックを取得
        try:
            # 同期ヘルパーを別スレッドで実行
            return await loop.run_in_executor(None, _db_execute_commit_sync, db_path, sql, params)
        except StateManagementError as sme:
            logger.error(f"Async DB Execute Error (StateManagement): {sme}", exc_info=False) # False avoids double traceback
            raise # そのまま再発生
        except Exception as e:
            # 予期せぬエラーもラップ (run_in_executor 自体のエラーなど)
            logger.error(f"Async DB Execute Unexpected Error: {e}", exc_info=True)
            raise StateManagementError(f"Async DB実行中の予期せぬエラー: {e}") from e

async def db_fetch_one_async(db_path: Path, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    """DB読み取り(一件)操作を非同期で実行 (ロックなし)"""
    loop = asyncio.get_running_loop()
    # 読み取りはロックなし
    try:
        return await loop.run_in_executor(None, _db_fetch_one_sync, db_path, sql, params)
    except StateManagementError as sme:
        logger.error(f"Async DB Fetch One Error (StateManagement): {sme}", exc_info=False)
        raise # そのまま再発生
    except Exception as e:
        logger.error(f"Async DB Fetch One Unexpected Error: {e}", exc_info=True)
        raise StateManagementError(f"Async DB読み取り(一件)中の予期せぬエラー: {e}") from e

async def db_fetch_all_async(db_path: Path, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    """DB読み取り(複数)操作を非同期で実行 (ロックなし)"""
    loop = asyncio.get_running_loop()
    # 読み取りはロックなし
    try:
        return await loop.run_in_executor(None, _db_fetch_all_sync, db_path, sql, params)
    except StateManagementError as sme:
        logger.error(f"Async DB Fetch All Error (StateManagement): {sme}", exc_info=False)
        raise # そのまま再発生
    except Exception as e:
        logger.error(f"Async DB Fetch All Unexpected Error: {e}", exc_info=True)
        raise StateManagementError(f"Async DB読み取り(複数)中の予期せぬエラー: {e}") from e 