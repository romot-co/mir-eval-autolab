import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine, List
import time
import asyncio
import sqlite3

# 関連モジュールインポート (型ヒント用)
from mcp.server.fastmcp import FastMCP
from . import db_utils, utils
from .serialization_utils import JsonNumpyEncoder # JSONエンコーダーをインポート
from src.utils.exception_utils import StateManagementError # StateManagementError をインポート

logger = logging.getLogger('mcp_server.session_manager')

# --- MCP Tools --- #

async def start_session(config: Dict[str, Any], db_path: Path, base_algorithm: str = "Unknown") -> Dict[str, Any]:
    """改善セッションを開始します。"""
    session_id = utils.generate_id()
    current_time = utils.get_timestamp()
    initial_cycle_state = {}

    try:
        await db_utils.db_execute_commit_async(
            db_path,
            "INSERT INTO sessions (session_id, base_algorithm, start_time, last_update, status, history, config, cycle_state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, base_algorithm, current_time, current_time, "active", "[]", json.dumps(config, cls=JsonNumpyEncoder), json.dumps(initial_cycle_state))
        )
        logger.info(f"新しいセッションが作成されました: {session_id}")
        return {
            "session_id": session_id,
            "base_algorithm": base_algorithm,
            "start_time": current_time,
            "last_update": current_time,
            "status": "active",
            "history": [],
            "config": config,
            "cycle_state": initial_cycle_state
        }
    except StateManagementError as sme:
        logger.error(f"セッション開始 DBエラー: {sme}", exc_info=False)
        raise sme # StateManagementError はそのまま再発生
    except Exception as e:
        logger.error(f"セッション開始中に予期せぬエラー: {e}", exc_info=True)
        # 予期せぬエラーは StateManagementError でラップ
        raise StateManagementError(f"Unexpected error starting session: {e}") from e

async def get_session_info(config: Dict[str, Any], db_path: Path, session_id: str) -> Dict[str, Any]:
    """セッション情報を取得します。"""
    current_time = utils.get_timestamp()
    row = None
    try:
        row = await db_utils.db_fetch_one_async(db_path, "SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    except StateManagementError as sme:
         logger.error(f"セッション情報取得 DBエラー ({session_id}): {sme}", exc_info=False)
         raise sme # StateManagementError はそのまま再発生
    except Exception as e:
         logger.error(f"セッション情報取得中に予期せぬエラー ({session_id}): {e}", exc_info=True)
         raise StateManagementError(f"Unexpected error fetching session info: {e}") from e

    if row:
        try:
            # last_update の更新もDB書き込みなのでエラーハンドリングが必要
            await db_utils.db_execute_commit_async(db_path, "UPDATE sessions SET last_update = ? WHERE session_id = ?", (current_time, session_id))

            session_info = dict(row)
            # Deserialize JSON fields
            try:
                session_info['history'] = json.loads(session_info.get('history', '[]') or '[]')
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"セッション {session_id} の履歴JSON解析失敗。空リストを返します。")
                session_info['history'] = []
            try:
                session_info['config'] = json.loads(session_info.get('config', '{}') or '{}')
            except (json.JSONDecodeError, TypeError):
                 logger.warning(f"セッション {session_id} の設定JSON解析失敗。空辞書を返します。")
                 session_info['config'] = {}
            try:
                session_info['cycle_state'] = json.loads(session_info.get('cycle_state', '{}') or '{}')
            except (json.JSONDecodeError, TypeError):
                 logger.warning(f"セッション {session_id} のサイクル状態JSON解析失敗。空辞書を返します。")
                 session_info['cycle_state'] = {}
            for key in ['current_metrics', 'best_metrics', 'baseline_metrics']:
                if session_info.get(key):
                    try:
                         session_info[key] = json.loads(session_info[key])
                    except (json.JSONDecodeError, TypeError):
                         logger.warning(f"セッション {session_id} の {key} JSON解析失敗。Noneを返します。")
                         session_info[key] = None
                else:
                     session_info[key] = None
            return session_info
        except StateManagementError as sme_update: # last_update 更新時のエラー
            logger.error(f"セッション last_update 更新 DBエラー ({session_id}): {sme_update}", exc_info=False)
            # 更新に失敗しても取得した情報自体は返せる場合もあるが、一貫性のためエラーとする
            raise StateManagementError(f"Failed to update session access time: {sme_update}") from sme_update
        except Exception as e:
            logger.error(f"セッション情報処理/更新中に予期せぬエラー ({session_id}): {e}", exc_info=True)
            raise StateManagementError(f"Unexpected error processing session info: {e}") from e
    else:
        raise ValueError(f"セッション {session_id} が見つかりません") # 見つからない場合は ValueError

# _add_session_history_sync 内のエラーハンドリングは db_utils.py で StateManagementError が
# raise されるようになったため、ここで再度ラップする必要性は低いが、
# ValueError (Session not found) は区別して処理する。

# --- 修正: add_session_history を非同期化し、ロック内で fetch & update --- #
async def add_session_history(
    config: Dict[str, Any], # config は get_session_info で使うので残す
    db_path: Path,
    session_id: str,
    event_type: str,
    event_data: Any,
    cycle_state_update: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """セッション履歴にイベントを追加し、cycle_state をアトミックに更新します (非同期、ロック付き)"""
    new_event = {"id": utils.generate_id(), "timestamp": utils.get_timestamp(), "type": event_type, "data": event_data}
    current_timestamp = new_event["timestamp"]

    async with db_utils.db_lock: # --- DBロック開始 --- #
        try:
            # 1. 現在の履歴と cycle_state を取得 (ロック内で実行)
            row = await db_utils.db_fetch_one_async(db_path, "SELECT history, cycle_state FROM sessions WHERE session_id = ?", (session_id,))
            if not row:
                logger.warning(f"履歴追加試行時にセッションが見つかりません: {session_id}")
                raise ValueError(f"セッション {session_id} が見つかりません")

            # 2. 履歴をデシリアライズして更新
            try:
                current_history = json.loads(row["history"] or '[]')
                if not isinstance(current_history, list): current_history = []
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"セッション {session_id} の履歴JSON解析失敗。新しいリストで初期化。", exc_info=True)
                current_history = []
            current_history.append(new_event)
            updated_history_json = json.dumps(current_history, cls=JsonNumpyEncoder)

            # 3. cycle_state をデシリアライズして更新 (必要な場合)
            sql = "" # UPDATE文を初期化
            params = () # パラメータを初期化
            log_msg_suffix = ""
            updated_cycle_state_json = row["cycle_state"] # 更新しない場合のデフォルト値
            if cycle_state_update is not None and isinstance(cycle_state_update, dict):
                try:
                    current_cycle_state = json.loads(row["cycle_state"] or '{}')
                    if not isinstance(current_cycle_state, dict): current_cycle_state = {}
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"セッション {session_id} の cycle_state JSON解析失敗。新しい辞書で初期化。", exc_info=True)
                    current_cycle_state = {}
                current_cycle_state.update(cycle_state_update)
                updated_cycle_state_json = json.dumps(current_cycle_state, cls=JsonNumpyEncoder)
                log_msg_suffix = " および cycle_state 更新"
                sql = "UPDATE sessions SET history = ?, cycle_state = ?, last_update = ? WHERE session_id = ?"
                params = (updated_history_json, updated_cycle_state_json, current_timestamp, session_id)
            else:
                # cycle_state を更新しない場合
                sql = "UPDATE sessions SET history = ?, last_update = ? WHERE session_id = ?"
                params = (updated_history_json, current_timestamp, session_id)
                log_msg_suffix = ""

            # 4. DB更新を実行 (ロック内で実行)
            #   同期ヘルパーではなく、直接SQLとパラメータを指定
            conn = None
            try:
                conn = db_utils.get_db_connection(db_path)
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                logger.info(f"セッション履歴に '{event_type}' を追加しました (Session: {session_id})" + log_msg_suffix)
            except sqlite3.Error as db_err_update:
                 logger.error(f"セッション履歴/状態更新中にDBエラー: {db_err_update}", exc_info=True)
                 if conn: conn.rollback()
                 raise StateManagementError(f"DB error updating session history/state: {db_err_update}") from db_err_update
            finally:
                 if conn: conn.close()

        except ValueError as ve:
            # session not found
            raise ve
        except StateManagementError as sme:
            # fetch or update error
            raise sme
        except Exception as e:
            # JSON decode or other unexpected errors
            logger.error(f"セッション履歴/状態更新プロセスで予期せぬエラー ({session_id}): {e}", exc_info=True)
            raise StateManagementError(f"Unexpected error during add_session_history process: {e}") from e
    # --- DBロック終了 --- #

    # 更新後のセッション情報を返す (ロック外で)
    try:
        return await get_session_info(config, db_path, session_id)
    except (ValueError, StateManagementError) as get_err:
        logger.error(f"履歴追加後のセッション情報取得エラー ({session_id}): {get_err}", exc_info=False)
        # 履歴追加自体は成功している可能性があるので、エラーを返しつつ警告
        raise StateManagementError(f"History added, but failed to retrieve updated session info: {get_err}") from get_err

def cleanup_old_sessions(config: Dict[str, Any]):
    """古いセッションや最大数を超えたセッションをクリーンアップする (同期)"""
    db_path = Path(config['paths']['db']) / db_utils.DB_FILENAME
    session_timeout_seconds = config.get('cleanup', {}).get('session_timeout_seconds', 86400)
    max_sessions_count = config.get('cleanup', {}).get('max_sessions_count', 100)

    if session_timeout_seconds <= 0 and max_sessions_count <= 0:
        logger.info("セッションのタイムアウトと最大数が無効なため、クリーンアップをスキップします。")
        return

    logger.info(f"古いセッションのクリーンアップを開始 (タイムアウト: {session_timeout_seconds}秒, 最大: {max_sessions_count}件)")
    deleted_timeout_count = 0
    deleted_excess_count = 0

    # --- DB操作は同期ヘルパー (_db_*) を使う --- 
    #   cleanup はバックグラウンドスレッドで実行される想定なので、
    #   async def にせず、同期ヘルパーで十分。
    try:
        # タイムアウトしたセッションを削除
        if session_timeout_seconds > 0:
            timeout_threshold = utils.get_timestamp() - session_timeout_seconds
            # 同期ヘルパーを使用
            deleted_timeout_count = db_utils._db_execute_commit_sync(db_path, "DELETE FROM sessions WHERE last_update < ?", (timeout_threshold,))
            if deleted_timeout_count and deleted_timeout_count > 0:
                logger.info(f"{deleted_timeout_count} 件のタイムアウトしたセッションを削除しました。")

        # 最大数を超えたセッションを削除
        if max_sessions_count > 0:
            # 同期ヘルパーを使用
            row = db_utils._db_fetch_one_sync(db_path, "SELECT COUNT(*) FROM sessions")
            current_session_count = row[0] if row else 0
            if current_session_count > max_sessions_count:
                sessions_to_delete_count = current_session_count - max_sessions_count
                logger.info(f"セッション数が最大保持数 {max_sessions_count} を超過 ({current_session_count} 件)。古い {sessions_to_delete_count} 件を削除します。")
                # 同期ヘルパーを使用
                deleted_excess_count = db_utils._db_execute_commit_sync(db_path, "DELETE FROM sessions WHERE session_id IN (SELECT session_id FROM sessions ORDER BY last_update ASC LIMIT ?)", (sessions_to_delete_count,))
                if deleted_excess_count and deleted_excess_count > 0:
                    logger.info(f"{deleted_excess_count} 件の超過セッションを削除しました。")
            else:
                 logger.info(f"現在のセッション数: {current_session_count} (最大: {max_sessions_count}) - 超過削除は不要。")

        logger.info(f"セッションクリーンアップ完了。削除(タイムアウト): {deleted_timeout_count or 0}, 削除(超過): {deleted_excess_count or 0}")

    except StateManagementError as sme:
        logger.error(f"セッションクリーンアップ中にDBエラー: {sme}", exc_info=False)
    except Exception as e:
        logger.error(f"セッションクリーンアップ中に予期せぬエラー: {e}", exc_info=True)

# --- Tool Registration --- #

def register_session_tools(mcp: FastMCP, config: Dict[str, Any], db_path: Path):
    """セッション管理関連のMCPツールを登録"""
    logger.info("Registering session management tools...")

    # MCPツールとして登録するために、必要な引数 (config, db_path) を部分適用
    mcp.tool("start_session")(lambda base_algorithm="Unknown": start_session(config, db_path, base_algorithm))
    mcp.tool("get_session_info")(lambda session_id: get_session_info(config, db_path, session_id))
    mcp.tool("add_session_history")(lambda session_id, event_type, event_data, cycle_state_update=None: add_session_history(config, db_path, session_id, event_type, event_data, cycle_state_update))

    logger.info("Session management tools registered.")

# --- Import Guard --- #
# This module should not be run directly
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 