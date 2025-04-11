import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine, List, Tuple
import time
import asyncio
from asyncio import Queue # Queue を直接インポート
from enum import Enum # Enum をインポート
import traceback # traceback をインポート
from pydantic import BaseModel # BaseModel をインポート

# 関連モジュールインポート (型ヒント用)
from mcp.server.fastmcp import FastMCP
from . import db_utils, utils
# core から JsonNumpyEncoder をインポート
# 注意: core が大きくなりすぎる場合、JsonNumpyEncoder は utils などに移動する方が良い可能性
# from .core import active_jobs # core から移動
from src.utils.json_utils import NumpyEncoder # JSONエンコーダー (旧 JsonNumpyEncoder)
# StateManagementError をインポート
from src.utils.exception_utils import StateManagementError, log_exception
from .schemas import JobInfo, JobStatus, ErrorDetails # schemas をインポート
from . import schemas # schemas モジュール自体を追加でインポート
from src.utils.misc_utils import generate_id, get_timestamp # 変更後
from . import session_manager # ★ 追加: セッションステータス更新用

logger = logging.getLogger('mcp_server.job_manager')

# --- JobStatus Enum (schemas.py からインポートするため削除) --- #
# class JobStatus(Enum):
#     PENDING = "pending"
#     RUNNING = "running"
#     COMPLETED = "completed"
#     FAILED = "failed"

# --- Job Data Class (schemas.py の JobInfo を使用) --- #
Job = JobInfo # Use Pydantic schema

# --- Global Variables --- #
job_queue: Queue[Tuple[str, Callable[..., Coroutine], tuple, dict]] = Queue()
active_jobs: Dict[str, Job] = {} # {job_id: JobInfo}

# --- Helper function to parse DB row into Job dict (schemas.py を使うので不要になる可能性) --- #
def _db_row_to_jobinfo(row: Dict[str, Any]) -> JobInfo:
    """DBの辞書からJobInfo Pydanticモデルに変換 (JSONデコード含む)"""
    job_data = dict(row)
    job_data['status'] = JobStatus(job_data.get('status', 'unknown')) # Enumに変換

    # JSONフィールドのデコード
    for field in ['result', 'error_details', 'task_args']:
        json_str = job_data.get(field)
        if json_str:
            try:
                job_data[field] = json.loads(json_str)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Job {job_data.get('job_id')} field '{field}' JSON decode failed.")
                # エラーの場合、フィールドをNoneにするか、エラー情報を含めるか検討
                job_data[field] = {"error": f"Failed to decode {field} JSON", "raw_data": json_str}
        else:
            job_data[field] = None # DBにNULLが入っている場合など

    # Pydanticモデルへの変換 (エラーハンドリング付き)
    try:
        job_info_model = JobInfo(**job_data)
        return job_info_model
    except Exception as e:
        logger.error(f"Failed to parse job data into JobInfo model for job {job_data.get('job_id')}: {e}", exc_info=True)
        # フォールバックとして辞書を返すか、エラーを発生させるか
        # ここではエラー情報を含めて JobInfo として返す試み (不完全かもしれない)
        return JobInfo(
            job_id=job_data.get('job_id', 'unknown'),
            status=JobStatus.UNKNOWN,
            error_details=ErrorDetails(message=f"Failed to parse job data: {e}")
        )

# --- Job Worker (Moved from core.py) --- #
async def job_worker(worker_id: int, config: Dict[str, Any], db_path: Path):
    """ジョブキューからタスクを取得して実行するワーカ"""
    logger.info(f"Job worker {worker_id} started.")
    while True:
        try:
            job_id, task_coro_func, args, kwargs = await job_queue.get()
            logger.info(f"Worker {worker_id}: Picked up job {job_id}.")

            start_time = time.time()
            # 1. DBステータスを running に更新
            await db_utils.db_execute_commit_async(
                db_path,
                "UPDATE jobs SET status = ?, start_time = ?, worker_id = ? WHERE job_id = ?",
                (JobStatus.RUNNING.value, start_time, str(worker_id), job_id)
            )
            # 2. active_jobs に追加 (JobInfoモデルで)
            # active_jobs[job_id] = {"status": JobStatus.RUNNING.value, "start_time": start_time, "worker_id": str(worker_id)} # 古い形式
            # task_args は DB 登録時のものを使う（再度シリアライズはしない）
            active_jobs[job_id] = JobInfo(
                job_id=job_id,
                status=JobStatus.RUNNING,
                start_time=start_time,
                worker_id=str(worker_id),
                session_id=kwargs.get('session_id'), # kwargs から取得
                tool_name=kwargs.get('tool_name') # kwargs から取得
            )

            result = None
            error_info: Optional[ErrorDetails] = None
            job_status = JobStatus.UNKNOWN

            try:
                # 3. タスク実行 (コルーチンを実行)
                task_result = await task_coro_func(*args, **kwargs)
                job_status = JobStatus.COMPLETED
                result = task_result

            except Exception as e:
                logger.error(f"Worker {worker_id}: Error executing job {job_id}: {e}", exc_info=True)
                job_status = JobStatus.FAILED
                error_info = ErrorDetails(
                    message=str(e),
                    type=type(e).__name__,
                    traceback=traceback.format_exc()
                )
                result = None # エラー時は result は null

                # ★★★ 追加: ジョブ失敗時にセッションステータスも更新 ★★★
                session_id_for_error = kwargs.get('session_id')
                if session_id_for_error:
                    try:
                        logger.warning(f"Job {job_id} failed, updating session {session_id_for_error} status to 'failed'.")
                        await session_manager.update_session_status_internal(
                            db_path=db_path,
                            session_id=session_id_for_error,
                            status="failed", # SessionStatus Enum を使うべきだが、直接文字列で指定
                            error_details=error_info.model_dump() # Pydanticモデルを辞書に
                        )
                        logger.info(f"Session {session_id_for_error} status updated to 'failed' due to job {job_id} failure.")
                    except Exception as session_update_err:
                        # セッション更新自体のエラーはログに残すが、ジョブワーカーは続行
                        logger.error(f"Failed to update session {session_id_for_error} status after job {job_id} failure: {session_update_err}", exc_info=True)
                # ★★★ 追加ここまで ★★★

            finally:
                end_time = time.time()
                logger.info(f"Worker {worker_id}: Job {job_id} finished with status '{job_status.value}' in {end_time - start_time:.2f} seconds.")

                # 4. DBステータスを completed/failed に更新、結果/エラーを保存
                result_json = None
                error_json = None
                try:
                    if result is not None:
                        # 結果が Pydantic モデルの場合、model_dump を使う
                        # if isinstance(result, BaseModel):
                        #     result_json = result.model_dump_json()
                        # else:
                        #     result_json = json.dumps(result, cls=NumpyEncoder)
                        # ★ job_worker は結果を直接シリアライズせず、DB更新時にスキーマから行う想定に変更
                        #    ただし、履歴記録のためにはここで result (オブジェクト) が必要
                        pass # result オブジェクトはこの後使う
                    if error_info is not None:
                        error_json = error_info.model_dump_json()

                    # ★ DB更新: result/error_info オブジェクトは直接渡せないため、json 文字列を渡す
                    result_for_db = None
                    if result is not None:
                        try:
                             # Pydanticモデルかチェックし、適切にダンプ
                             if isinstance(result, BaseModel):
                                 result_for_db = result.model_dump_json()
                             else:
                                 result_for_db = json.dumps(result, cls=NumpyEncoder)
                        except Exception as dump_err:
                             logger.error(f"Failed to serialize job result for DB {job_id}: {dump_err}", exc_info=True)
                             result_for_db = json.dumps({"serialization_error": str(dump_err)}) # エラー情報だけでも保存

                    await db_utils.db_execute_commit_async(
                        db_path,
                        "UPDATE jobs SET status = ?, result = ?, error_details = ?, end_time = ? WHERE job_id = ?",
                        (job_status.value, result_for_db, error_json, end_time, job_id)
                    )
                except Exception as db_err:
                     # DB更新失敗のログは出すが、ワーカーは止めない
                     logger.error(f"Worker {worker_id}: Failed to update job {job_id} status in DB: {db_err}", exc_info=True)

                # ★★★ 追加: ジョブ完了/失敗時に履歴を記録 ★★★
                session_id_for_history = kwargs.get('session_id')
                tool_name_for_history = kwargs.get('tool_name')
                if session_id_for_history and tool_name_for_history:
                    event_type: Optional[str] = None
                    event_data: Optional[Dict[str, Any]] = None
                    try:
                        if job_status == JobStatus.COMPLETED:
                            # ツール名に基づいて完了イベントタイプとデータを決定
                            # (例: tool_name が 'get_code' なら event_type は 'code_load.completed')
                            event_type = f"{tool_name_for_history}.completed"
                            # result オブジェクトを辞書に変換 (Pydanticなら .model_dump())
                            if isinstance(result, BaseModel):
                                event_data = result.model_dump()
                            elif isinstance(result, dict):
                                event_data = result
                            else:
                                # 単純な値やリストの場合の処理 (必要に応じて調整)
                                event_data = {"result": result}
                            # job_id を event_data に追加 (履歴スキーマで定義されていれば)
                            if event_data is not None: event_data['job_id'] = job_id

                        elif job_status == JobStatus.FAILED and error_info:
                            # ツール名に基づいて失敗イベントタイプを決定
                            event_type = f"{tool_name_for_history}.failed"
                            # error_info (ErrorDetails) を辞書に変換
                            event_data = error_info.model_dump()
                            # job_id を event_data に追加 (履歴スキーマで定義されていれば)
                            if event_data is not None: event_data['job_id'] = job_id

                        if event_type and event_data:
                            logger.debug(f"Adding history event '{event_type}' for session {session_id_for_history} (Job: {job_id})")
                            await session_manager.add_session_history(
                                config=config, # ★ config が必要
                                db_path=db_path,
                                session_id=session_id_for_history,
                                event_type=event_type,
                                event_data=event_data
                                # cycle_state_update はここでは設定しない (必要なら別途行う)
                            )
                        else:
                             logger.debug(f"Skipping history add for job {job_id}: No event type or data determined (Status: {job_status})")

                    except Exception as history_err:
                         # 履歴記録自体のエラーはログに残す
                         logger.error(f"Failed to add history for completed/failed job {job_id} (Session: {session_id_for_history}): {history_err}", exc_info=True)
                # ★★★ 追加ここまで ★★★

                # 5. active_jobs から削除
                if job_id in active_jobs:
                    del active_jobs[job_id]

                # 6. キューにタスク完了を通知
                job_queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"Job worker {worker_id} cancelled.")
            break
        except Exception as e:
            # ワーカー自体の予期せぬエラー (キュー取得など)
            logger.critical(f"Job worker {worker_id} encountered critical error: {e}", exc_info=True)
            # 必要ならここでワーカーを再起動するロジックなど
            await asyncio.sleep(5) # 少し待ってからループ再開

# --- Job Worker Start (Moved from core.py) --- #
async def start_job_workers(num_workers: int, config: Dict[str, Any], db_path: Path):
    """指定された数のジョブワーカータスクを開始する"""
    logger.info(f"Starting {num_workers} job workers...")
    worker_tasks = []
    for i in range(num_workers):
        task = asyncio.create_task(job_worker(i + 1, config, db_path))
        worker_tasks.append(task)
    logger.info(f"{num_workers} job workers scheduled.")
    # ワーカータスクのリストを返すか、グローバルに保持するか？ (現状は起動するだけ)

# --- Start Async Job (Moved from core.py) --- #
async def start_async_job(
    config: Dict[str, Any],
    db_path: Path,
    task_coroutine_func: Callable[..., Coroutine], # 実行するコルーチン関数
    tool_name: str,
    session_id: Optional[str] = None,
    *args, **kwargs
) -> Dict[str, str]: # 戻り値を JobStartResponse 形式の Dict に変更
    """非同期ジョブを開始し、DBに登録してキューに入れる"""
    job_id = generate_id("job")
    current_time = get_timestamp()

    # 引数をJSON文字列として保存 (デバッグ用)
    task_args_json = None
    try:
        # kwargs にコルーチンや非シリアライズ可能なオブジェクトが含まれる可能性
        # 安全な方法でシリアライズ可能な引数のみを保存する
        serializable_args = {
            k: v for k, v in kwargs.items()
            if isinstance(v, (str, int, float, bool, list, dict, tuple)) or v is None
        }
        # args (タプル) も同様に処理
        serializable_pos_args = tuple(
            arg for arg in args
            if isinstance(arg, (str, int, float, bool, list, dict, tuple)) or arg is None
        )
        task_args_for_db = {"args": serializable_pos_args, "kwargs": serializable_args}
        task_args_json = json.dumps(task_args_for_db, cls=NumpyEncoder)
    except Exception as e:
        logger.warning(f"Failed to serialize task args for job {job_id}: {e}")
        task_args_json = json.dumps({"error": "Failed to serialize arguments"})

    # 1. DBにジョブを登録 (pending)
    try:
        await db_utils.db_execute_commit_async(
            db_path,
            "INSERT INTO jobs (job_id, session_id, tool_name, status, created_at, task_args) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, session_id, tool_name, JobStatus.PENDING.value, current_time, task_args_json)
        )
    except Exception as e:
        logger.error(f"Failed to register job {job_id} in DB: {e}", exc_info=True)
        # DB登録失敗時はエラーを返すか、例外を発生させるか？
        # MCPツール側で処理できるように、辞書でエラー情報を返す
        return {"job_id": job_id, "status": "error", "message": f"Failed to register job in DB: {e}"}

    # 2. ジョブキューに追加
    # ここで渡す *args, **kwargs はシリアライズ前のオリジナルのもの
    await job_queue.put((job_id, task_coroutine_func, args, kwargs))
    logger.info(f"Job {job_id} (Tool: {tool_name}, Session: {session_id}) queued.")

    # 3. JobStartResponse 形式の辞書を返す
    return {"job_id": job_id, "status": JobStatus.PENDING.value, "message": "Job successfully queued."}


# --- Functions for Job Management (Replaces JobManager class methods) --- #
async def get_job_status(db_path: Path, job_id: str) -> JobInfo:
    """指定されたJob IDのジョブ情報をDBから取得します。"""
    # active_jobs (メモリ) を参照して running 状態を取得
    if job_id in active_jobs:
        logger.debug(f"Getting status for active job {job_id} from memory.")
        # メモリ上の JobInfo を返す (コピーを返すのが安全)
        return active_jobs[job_id].model_copy()

    # DBから取得
    logger.debug(f"Getting status for job {job_id} from DB.")
    sql = "SELECT * FROM jobs WHERE job_id = ?"
    try:
        row = await db_utils.db_fetch_one_async(db_path, sql, (job_id,))
        if row:
            return _db_row_to_jobinfo(dict(row))
        else:
            # 見つからない場合はエラー情報を含む JobInfo を返す
            logger.warning(f"Job ID {job_id} not found in database or active jobs.")
            return JobInfo(
                job_id=job_id,
                status=JobStatus.UNKNOWN, # または専用の NOT_FOUND ステータス？
                error_details=ErrorDetails(message="Job ID not found")
            )
    except Exception as e:
         logger.error(f"Failed to get job {job_id} status from DB: {e}", exc_info=True)
         # DBアクセスエラーの場合もエラー情報を含む JobInfo を返す
         return JobInfo(
             job_id=job_id,
             status=JobStatus.UNKNOWN,
             error_details=ErrorDetails(message=f"Error retrieving job status: {e}", type=type(e).__name__)
         )

# cleanup_old_jobs は変更なし

# --- Tool Registration --- #
def register_job_tools(mcp: FastMCP, config: Dict[str, Any], db_path: Path):
    """ジョブ管理関連のMCPツールを登録"""
    logger.info("Registering job management tools...")
    # --- 修正: job_manager_instance を削除し、get_job_status 関数を登録 --- #
    # get_job_status 関数をラップして db_path を渡す
    # ★ lambda に変更し、input_schema を追加
    mcp.tool("get_job_status", input_schema=schemas.GetJobStatusInput)(
        lambda job_id: get_job_status(db_path, job_id)
    )
    logger.info("Job management tools registered.")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 