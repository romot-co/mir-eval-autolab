import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine, List, Tuple
import time
import asyncio
# import sqlite3 # 不要になったため削除

# 関連モジュールインポート (型ヒント用)
from mcp.server.fastmcp import FastMCP
from . import db_utils, utils
# core から JsonNumpyEncoder をインポート
# 注意: core が大きくなりすぎる場合、JsonNumpyEncoder は utils などに移動する方が良い可能性
# from .core import active_jobs # core から移動
from .serialization_utils import JsonNumpyEncoder # core から移動
# StateManagementError をインポート
from src.utils.exception_utils import StateManagementError

logger = logging.getLogger('mcp_server.job_manager')

# --- Job State Management (Moved from core.py) ---
job_queue: asyncio.Queue[Tuple[str, Callable[..., Coroutine], tuple, dict]] = asyncio.Queue()
active_jobs: Dict[str, Dict[str, Any]] = {} # {job_id: {status: ..., future: ..., etc.}}

# --- MCP Tools --- #

async def get_job_status(config: Dict[str, Any], db_path: Path, job_id: str) -> Dict[str, Any]:
    """非同期ジョブのステータスと結果を取得します。"""
    # まずメモリ上の active_jobs を確認 (高速化のため)
    if job_id in active_jobs:
        job_info_mem = active_jobs[job_id].copy() # コピーして返す
        # DBから最新情報を取得してマージすることも検討できるが、ここではメモリ優先
        if job_info_mem['status'] in ['completed', 'failed']:
             # 結果やエラーはメモリ上に保持されている想定
             # (core.job_worker がメモリ上の active_jobs を更新するため)
             return job_info_mem
        # pending or running の場合はDBも確認する（ワーカーによって更新されている可能性）

    # メモリにない、または完了/失敗していない場合はDBを確認
    row = None
    try:
        row = await db_utils.db_fetch_one_async(db_path, "SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    except StateManagementError as sme:
         logger.error(f"ジョブ状態取得 DBエラー ({job_id}): {sme}", exc_info=False)
         raise sme # StateManagementError はそのまま再発生
    except Exception as e:
         logger.error(f"ジョブ状態取得中に予期せぬエラー ({job_id}): {e}", exc_info=True)
         raise StateManagementError(f"Unexpected error fetching job status: {e}") from e

    if row:
        job_info_db = dict(row)
        # 結果やエラー詳細のJSONをデシリアライズ
        job_info_db.setdefault("result", None)
        job_info_db.setdefault("error_details", None)
        job_info_db.setdefault("task_args", None)

        if job_info_db["status"] == 'completed' and job_info_db["result"]:
            try:
                job_info_db["result"] = json.loads(job_info_db["result"])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"ジョブ {job_id} の結果JSONデコード失敗。Rawデータを返します: {job_info_db['result'][:100]}...")
                # 生の文字列を返すか、エラーを示すか？
        elif job_info_db["status"] == 'failed' and job_info_db["error_details"]:
            try:
                # error_details には traceback も含まれる想定
                job_info_db["error_details"] = json.loads(job_info_db["error_details"])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"ジョブ {job_id} のエラー詳細JSONデコード失敗。Rawデータを返します: {job_info_db['error_details'][:100]}...")

        # task_args もデシリアライズ
        if job_info_db["task_args"]:
             try:
                  job_info_db["task_args"] = json.loads(job_info_db["task_args"])
             except (json.JSONDecodeError, TypeError):
                  logger.warning(f"ジョブ {job_id} の引数JSONデコード失敗。Rawデータを返します: {job_info_db['task_args'][:100]}...")
                  job_info_db["task_args"] = {"error": "Failed to decode args"}

        # メモリ上の情報が存在すれば、DB情報で更新する（ただし status が pending/running の場合のみ？）
        if job_id in active_jobs:
            active_jobs[job_id].update(job_info_db) # DB情報でメモリを更新
            return active_jobs[job_id]
        else:
            # DBにしかなかった場合（古いジョブなど）
            return job_info_db
    else:
        # メモリにもDBにもない場合
        raise ValueError(f"ジョブ {job_id} が見つかりません")

# --- Cleanup Function --- #

# --- 修正: cleanup_old_jobs を非同期化 --- #
async def cleanup_old_jobs(config: Dict[str, Any]):
    """古いジョブやスタックしたジョブをクリーンアップする (非同期)"""
    db_path = Path(config['paths']['db']) / db_utils.DB_FILENAME
    job_stuck_timeout_seconds = config.get('cleanup', {}).get('job_stuck_timeout_seconds', 3600)
    job_completed_retention_seconds = config.get('cleanup', {}).get('job_completed_retention_seconds', 604800)
    max_jobs_count = config.get('cleanup', {}).get('max_jobs_count', 500)

    if job_stuck_timeout_seconds <= 0 and job_completed_retention_seconds <= 0 and max_jobs_count <= 0:
        logger.info("ジョブのタイムアウト、保持期間、最大数が無効なため、クリーンアップをスキップします。")
        return

    logger.info(f"古いジョブ/スタックジョブのクリーンアップ開始 (スタック: {job_stuck_timeout_seconds}秒, 保持: {job_completed_retention_seconds}秒, 最大: {max_jobs_count}件)")
    cleaned_job_ids = set()
    now = utils.get_timestamp()

    try:
        # スタックしたジョブを 'failed' に更新
        if job_stuck_timeout_seconds > 0:
            stuck_threshold = now - job_stuck_timeout_seconds
            stuck_error_msg = f"Job timed out (stuck for > {job_stuck_timeout_seconds}s)"
            # start_time ではなく end_time or last_update を見るべき？ -> DBスキーマに last_update がないので start_time で代用
            # --- 修正: 非同期DB関数を使用 --- #
            await db_utils.db_execute_commit_async(
                db_path,
                "UPDATE jobs SET status = 'failed', result = ?, error_details = ?, end_time = ? WHERE (status = 'pending' OR status = 'running') AND start_time < ?",
                (json.dumps({"error": "timeout"}), json.dumps({"error": stuck_error_msg}, cls=JsonNumpyEncoder), now, stuck_threshold)
            )
            # 件数は返さないのでログで示す
            logger.warning(f"スタックしたジョブ (開始 < {stuck_threshold}) のステータスを 'failed' に更新試行完了。")

        # 保持期間を過ぎた完了/失敗ジョブを削除
        if job_completed_retention_seconds > 0:
            retention_threshold = now - job_completed_retention_seconds
            # --- 修正: 非同期DB関数を使用 --- #
            await db_utils.db_execute_commit_async(
                db_path,
                "DELETE FROM jobs WHERE (status = 'completed' OR status = 'failed') AND end_time < ?",
                (retention_threshold,)
            )
            # 件数は返さないのでログで示す
            logger.info(f"保持期間超過 ({job_completed_retention_seconds}秒) の完了/失敗ジョブ (終了 < {retention_threshold}) の削除試行完了。")

        # 最大保持数を超えたジョブを削除 (古いものから)
        if max_jobs_count > 0:
            # --- 修正: 非同期DB関数を使用 --- #
            row = await db_utils.db_fetch_one_async(db_path, "SELECT COUNT(*) FROM jobs")
            current_job_count = row[0] if row else 0
            if current_job_count > max_jobs_count:
                jobs_to_delete_count = current_job_count - max_jobs_count
                logger.info(f"ジョブ数が最大保持数 {max_jobs_count} を超過 ({current_job_count} 件)。古い {jobs_to_delete_count} 件の削除を試行します。")
                # start_time が古い順に削除 (完了/失敗ジョブ優先)
                # --- 修正: 非同期DB関数を使用 --- #
                await db_utils.db_execute_commit_async(
                    db_path,
                    "DELETE FROM jobs WHERE job_id IN (SELECT job_id FROM jobs ORDER BY CASE status WHEN 'completed' THEN 1 WHEN 'failed' THEN 2 ELSE 3 END ASC, start_time ASC LIMIT ?)",
                    (jobs_to_delete_count,)
                )
                # 件数は返さないのでログで示す
                logger.info(f"{jobs_to_delete_count} 件の超過ジョブの削除試行完了。")
            else:
                logger.info(f"現在のジョブ数: {current_job_count} (最大: {max_jobs_count}) - 超過削除は不要。")

        # メモリ上の active_jobs からも古い完了/失敗ジョブを削除
        # (DBから削除されたIDに基づいてメモリをクリーンアップ)
        cleaned_job_ids = set()
        if current_job_count > max_jobs_count:
            # 削除されたIDを特定するのは難しいので、単純に完了/失敗していて古いものをメモリから削除
            retention_threshold_mem = now - job_completed_retention_seconds if job_completed_retention_seconds > 0 else 0
            stuck_threshold_mem = now - job_stuck_timeout_seconds if job_stuck_timeout_seconds > 0 else 0

            ids_to_remove = set()
            for job_id, job_data in list(active_jobs.items()): # イテレーション中に削除するためリスト化
                status = job_data.get('status')
                start_time = job_data.get('start_time', 0)
                end_time = job_data.get('end_time') # job_worker で end_time が active_jobs に追加されるか確認

                should_remove = False
                if status in ['completed', 'failed'] and end_time and retention_threshold_mem > 0 and end_time < retention_threshold_mem:
                    should_remove = True
                elif status in ['pending', 'running'] and stuck_threshold_mem > 0 and start_time < stuck_threshold_mem:
                    # DBで failed に更新されたはずなので、メモリからも削除
                    should_remove = True

                # 最大保持数による削除はDBに任せ、メモリからは完了/失敗かつ古いものだけ削除

                if should_remove:
                    ids_to_remove.add(job_id)

            for job_id in ids_to_remove:
                 if job_id in active_jobs:
                      del active_jobs[job_id]
                      cleaned_job_ids.add(job_id)
                      logger.debug(f"メモリ上のアクティブジョブリストから {job_id} を削除しました。")
            if cleaned_job_ids:
                 logger.info(f"メモリ上のアクティブジョブリストから {len(cleaned_job_ids)} 件の古いジョブを削除しました。")


    except Exception as e:
        logger.error(f"ジョブクリーンアップ中にエラー: {e}", exc_info=True)
    finally:
        # 正確な件数は追跡しないため、簡略化
        logger.info(f"ジョブクリーンアップ完了。メモリクリーンアップ: {len(cleaned_job_ids)} 件。")


# --- Tool Registration --- #

def register_job_tools(mcp: FastMCP, config: Dict[str, Any], db_path: Path):
    """ジョブ管理関連のMCPツールを登録"""
    logger.info("Registering job management tools...")

    # MCPツールとして登録するために、必要な引数 (config, db_path) を部分適用
    mcp.tool("get_job_status")(lambda job_id: get_job_status(config, db_path, job_id))

    logger.info("Job management tools registered.")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 