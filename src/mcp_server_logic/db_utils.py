import asyncio

# import sqlite3 # sqlite3 を削除
import logging
import os

# import threading # threading を削除
from pathlib import Path
from typing import Any, List, Optional  # Any を追加

import aiosqlite  # aiosqlite をインポート

# StateManagementError をインポート
from src.utils.exception_utils import StateManagementError

logger = logging.getLogger("mcp_server.db")

# --- DB Constants and Setup ---
DB_FILENAME = "mcp_server_state.db"
# workspace_dir は config から取得する想定
# db_path = workspace_dir / "db" / DB_FILENAME

# db_lock = threading.Lock() # DB書き込み用のロック <- asyncio.Lock に変更
db_lock = asyncio.Lock()  # 非同期対応のDBアクセスロック
# _sync_db_lock = threading.Lock() # 同期処理用のDBアクセスロック <- 削除

# --- 同期接続関数削除 ---
# def get_db_connection(db_path: Path) -> sqlite3.Connection:
#     """データベース接続を取得 (同期)
#
#     注意: この関数自体は同期ですが、非同期コンテキスト (run_in_executor)
#     から呼び出されることを想定しています。
#     check_same_thread=False はそのためです。
#     """
#     try:
#         # タイムアウトを少し長く設定
#         conn = sqlite3.connect(str(db_path), timeout=20.0, check_same_thread=False)
#         conn.row_factory = sqlite3.Row # カラム名でアクセスできるようにする
#         conn.execute("PRAGMA journal_mode=WAL;") # WALモードを有効化 (同時読み書き性能向上)
#         conn.execute("PRAGMA busy_timeout = 10000;") # 競合時の待機時間を設定 (10秒)
#         logger.debug(f"DB connection to {db_path} obtained.")
#         return conn
#     except sqlite3.Error as db_err:
#         logger.error(f"Failed to get DB connection to {db_path}: {db_err}", exc_info=True)
#         raise StateManagementError(f"DB接続エラー: {db_err}") from db_err


# init_database を async def に変更
async def init_database(config: dict):
    """データベースを初期化 (テーブル作成など) (非同期)"""
    # 絶対パスでDB_DIRを指定するように修正
    db_dir = Path(
        config.get("paths", {}).get(
            "db", "/Users/user/Documents/mir-eval-autolab/mcp_workspace/db"
        )
    )
    # 必ず絶対パスであることを確認
    if not db_dir.is_absolute():
        db_dir = Path.cwd() / db_dir

    logger.info(f"データベースを初期化中: {db_dir / DB_FILENAME}")
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / DB_FILENAME
    db = None  # finally節のために先に宣言
    try:
        # aiosqlite を使用して非同期接続
        db = await aiosqlite.connect(str(db_path), timeout=20.0)
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout = 10000;")
        db.row_factory = aiosqlite.Row  # aiosqlite.Row を使用

        # Sessions テーブル
        await db.execute(
            """
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
        """
        )

        # Jobs テーブル
        await db.execute(
            """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            session_id TEXT, -- Optional: link to a session
            tool_name TEXT,
            status TEXT, -- 'pending', 'running', 'completed', 'failed'
            start_time REAL,
            end_time REAL,
            result TEXT, -- JSON serialized result or error message
            error_details TEXT, -- JSON blob for detailed error info (e.g., traceback)
            task_args TEXT, -- JSON serialized args for the task
            worker_id TEXT, -- ID of the worker that processed the job
            created_at REAL DEFAULT (strftime('%s', 'now')) -- レコード作成時刻 (デバッグ用)
        )
        """
        )

        # インデックス追加
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_last_update ON sessions(last_update);"
        )
        await db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);")
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_start_time ON jobs(start_time);"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_end_time ON jobs(end_time);"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_session_id ON jobs(session_id);"
        )

        await db.commit()
        logger.info("データベースの初期化完了")
    except aiosqlite.Error as db_err:  # aiosqlite.Error をキャッチ
        logger.error(f"データベース初期化中にエラーが発生: {db_err}", exc_info=True)
        if db:
            await db.rollback()
        raise StateManagementError(f"DB初期化エラー: {db_err}") from db_err
    finally:
        if db:
            await db.close()


# --- 同期 DB Helper Functions 削除 ---
# def _db_execute_commit_sync(db_path: Path, sql: str, params: tuple = ()) -> Optional[int]:
#     """DBに書き込み操作を実行してコミット (同期ヘルパー)
#     呼び出し元で asyncio.Lock() による排他制御がされている前提。
#     """
#     # ... (削除) ...
#
# def _db_fetch_one_sync(db_path: Path, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
#     """DBから一件のレコードを取得 (同期ヘルパー)
#     読み取り操作なのでロックは必須ではないが、WALモードなら書き込みと並行可能。
#     """
#     # ... (削除) ...
#
# def _db_fetch_all_sync(db_path: Path, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
#     """DBから複数のレコードを取得 (同期ヘルパー)
#     読み取り操作なのでロックは必須ではないが、WALモードなら書き込みと並行可能。
#     """
#     # ... (削除) ...


# --- Async DB Wrapper Functions を aiosqlite 直接使用に書き換え ---
async def db_execute_commit_async(
    db_path: Path, sql: str, params: tuple = ()
) -> Optional[int]:
    """DB書き込み操作を非同期で実行 (ロック付き)"""
    rows_affected = None
    db = None
    logger.debug(f"Attempting to acquire DB lock for write: {sql[:50]}...")
    async with db_lock:  # 非同期ロックを取得
        logger.debug(f"Acquired DB lock for write: {sql[:50]}...")
        try:
            logger.debug(f"Connecting to DB (write): {db_path}")
            # aiosqlite を直接使用
            db = await aiosqlite.connect(str(db_path), timeout=20.0)
            await db.execute("PRAGMA journal_mode=WAL;")  # 必要に応じて設定
            await db.execute("PRAGMA busy_timeout = 10000;")
            async with db.execute(sql, params) as cursor:
                rows_affected = cursor.rowcount
            await db.commit()
            logger.debug(
                f"Async Executed and committed: {sql[:50]}... with params {params}"
            )
        except aiosqlite.Error as db_err:
            logger.error(
                f"Async DB Execute Error: {db_err} for SQL: {sql}", exc_info=True
            )
            if db:
                logger.debug("Rolling back transaction due to error.")
                await db.rollback()
            raise StateManagementError(f"DB実行エラー (非同期): {db_err}") from db_err
        finally:
            if db:
                logger.debug(f"Closing DB connection (write): {db_path}")
                await db.close()
            logger.debug(f"Released DB lock for write: {sql[:50]}...")
    return rows_affected


async def db_fetch_one_async(
    db_path: Path, sql: str, params: tuple = ()
) -> Optional[aiosqlite.Row]:
    """DB読み取り(一件)操作を非同期で実行 (ロックなし)"""
    row: Optional[aiosqlite.Row] = None
    db = None
    # 読み取りはロックなし
    logger.debug(f"Attempting DB fetch one (no lock): {sql[:50]}...")
    try:
        logger.debug(f"Connecting to DB (read one): {db_path}")
        db = await aiosqlite.connect(str(db_path), timeout=20.0)
        # 読み取り操作なので WAL/timeout 設定は必須ではないかもしれないが、念のため
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout = 10000;")
        db.row_factory = aiosqlite.Row  # Rowファクトリを設定
        async with db.execute(sql, params) as cursor:
            row = await cursor.fetchone()
        logger.debug(f"Async Fetched one: {sql[:50]}... with params {params}")
        return row
    except aiosqlite.Error as db_err:
        logger.error(
            f"Async DB Fetch One Error: {db_err} for SQL: {sql}", exc_info=True
        )
        raise StateManagementError(
            f"DB読み取りエラー (一件・非同期): {db_err}"
        ) from db_err
    finally:
        if db:
            logger.debug(f"Closing DB connection (read one): {db_path}")
            await db.close()


async def db_fetch_all_async(
    db_path: Path, sql: str, params: tuple = ()
) -> List[aiosqlite.Row]:
    """DB読み取り(複数)操作を非同期で実行 (ロックなし)"""
    rows: List[aiosqlite.Row] = []
    db = None
    # 読み取りはロックなし
    logger.debug(f"Attempting DB fetch all (no lock): {sql[:50]}...")
    try:
        logger.debug(f"Connecting to DB (read all): {db_path}")
        db = await aiosqlite.connect(str(db_path), timeout=20.0)
        # 読み取り操作なので WAL/timeout 設定は必須ではないかもしれないが、念のため
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout = 10000;")
        db.row_factory = aiosqlite.Row  # Rowファクトリを設定
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        logger.debug(f"Async Fetched all: {sql[:50]}... with params {params}")
        return rows
    except aiosqlite.Error as db_err:
        logger.error(
            f"Async DB Fetch All Error: {db_err} for SQL: {sql}", exc_info=True
        )
        raise StateManagementError(
            f"DB読み取りエラー (複数・非同期): {db_err}"
        ) from db_err
    finally:
        if db:
            logger.debug(f"Closing DB connection (read all): {db_path}")
            await db.close()
