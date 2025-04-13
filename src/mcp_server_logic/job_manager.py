import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine, List, Tuple, Union
import time
import asyncio
from asyncio import Queue # Queue を直接インポート
from enum import Enum # Enum をインポート
import traceback # traceback をインポート
from pydantic import BaseModel # BaseModel をインポート
import sqlite3
import pickle

# 関連モジュールインポート (型ヒント用)
from mcp.server.fastmcp import FastMCP
from . import db_utils
# core から JsonNumpyEncoder をインポート
# 注意: core が大きくなりすぎる場合、JsonNumpyEncoder は utils などに移動する方が良い可能性
# from .core import active_jobs # core から移動
from src.utils.json_utils import NumpyEncoder # JSONエンコーダー (旧 JsonNumpyEncoder)
# StateManagementError をインポート
from src.utils.exception_utils import StateManagementError, log_exception, FileError, ConfigError
from .schemas import JobInfo, JobStatus, ErrorDetails # schemas をインポート
from . import schemas # schemas モジュール自体を追加でインポート
from src.utils.misc_utils import generate_id, get_timestamp # 変更後
from . import session_manager # ★ 追加: セッションステータス更新用
from src.utils.path_utils import validate_path_within_allowed_dirs, get_workspace_dir # パス検証用に追加

logger = logging.getLogger('mcp_server.job_manager')

# --- JobStatus Enum (schemas.py からインポートするため削除) --- #
# class JobStatus(Enum):
#     PENDING = "pending"
#     RUNNING = "running"
#     COMPLETED = "completed"
#     FAILED = "failed"

# --- Job Data Class (schemas.py の JobInfo を使用) --- #
# Job = JobInfo # Use Pydantic schema (別名をやめる)

# --- Global Variables --- #
job_queue: Queue[Tuple[str, Callable[..., Coroutine], tuple, dict]] = Queue()
active_jobs: Dict[str, JobInfo] = {} # {job_id: JobInfo} (JobInfo を直接使う)

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
    
    # ワーカー起動時にDBパスを検証
    try:
        workspace_dir = get_workspace_dir(config)
        validated_db_path = validate_path_within_allowed_dirs(
            db_path,
            allowed_base_dirs=[workspace_dir],
            check_existence=True,
            check_is_file=True,
            allow_absolute=True
        )
        db_path = validated_db_path
        logger.debug(f"Worker {worker_id}: Validated DB path: {db_path}")
    except (ConfigError, FileError, ValueError) as e:
        logger.critical(f"Worker {worker_id}: Failed to validate DB path: {e}. Worker cannot start.")
        return  # ワーカーを終了
    
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
    task_coroutine: Coroutine[Any, Any, Any], # 実際のコルーチンを受け取る
    tool_name: str,
    session_id: Optional[str] = None,
    **kwargs # e.g., detector_name, version
) -> Dict[str, str]: # JobStartResponse 形式
    """非同期ジョブを開始し、DBに登録する。"""
    start_time = time.monotonic()
    
    # --- Generate Job ID --- #
    # job_id = f"job_{tool_name}_{get_timestamp()}" # 古い形式
    job_id = generate_id() # ★★★ 引数なしに変更 ★★★
    logger.info(f"Starting async job: {job_id} (Tool: {tool_name}, Session: {session_id}) with args: {kwargs}")
    
    # --- Job Record Preparation --- #
    job_record = schemas.JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        start_time=start_time,
        worker_id=None,
        session_id=session_id,
        tool_name=tool_name,
        **kwargs
    )

    # DBパスの検証
    try:
        # ワークスペースディレクトリを取得
        workspace_dir = get_workspace_dir(config)
        # DBパスが許可されたディレクトリ内にあることを検証
        validated_db_path = validate_path_within_allowed_dirs(
            db_path,
            allowed_base_dirs=[workspace_dir],
            check_existence=True,  # DBファイルが存在することを確認
            check_is_file=True,    # ファイルであることを確認
            allow_absolute=True    # 絶対パスを許可
        )
        # 検証済みパスを使用
        db_path = validated_db_path
    except (ConfigError, FileError, ValueError) as e:
        logger.error(f"Failed to validate DB path for job {job_id}: {e}")
        return {
            "job_id": job_id, 
            "status": "error", 
            "message": f"Failed to validate DB path: {e}"
        }

    # 引数をJSON文字列として保存 (デバッグ用)
    task_args_json = None
    try:
        # kwargs にコルーチンや非シリアライズ可能なオブジェクトが含まれる可能性
        # 安全な方法でシリアライズ可能な引数のみを保存する
        serializable_args = {
            k: v for k, v in kwargs.items()
            if isinstance(v, (str, int, float, bool, list, dict, tuple)) or v is None
        }
        # args変数が定義されていないため、空のタプルを使用
        serializable_pos_args = tuple()
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
            (job_id, session_id, tool_name, JobStatus.PENDING.value, start_time, task_args_json)
        )
    except Exception as e:
        logger.error(f"Failed to register job {job_id} in DB: {e}", exc_info=True)
        # DB登録失敗時はエラーを返すか、例外を発生させるか？
        # MCPツール側で処理できるように、辞書でエラー情報を返す
        return {"job_id": job_id, "status": "error", "message": f"Failed to register job in DB: {e}"}

    # 2. ジョブキューに追加
    # 未定義の args 変数を使わず、空のタプルを渡す
    await job_queue.put((job_id, task_coroutine, tuple(), kwargs))
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

    # DBパスの検証
    try:
        # configを引数で受け取っていないため、関数内でworkspace_dirを直接取得することはできない
        # しかし、db_path自体が通常validate済みのパスが渡されることを前提に、簡易的な検証を行う
        
        # DBファイルが存在するか確認
        if not db_path.exists():
            logger.error(f"DB file does not exist: {db_path}")
            return JobInfo(
                job_id=job_id,
                status=JobStatus.UNKNOWN,
                error_details=ErrorDetails(message=f"DB file does not exist: {db_path}")
            )
            
        # DBファイルがファイルであるか確認
        if not db_path.is_file():
            logger.error(f"DB path is not a file: {db_path}")
            return JobInfo(
                job_id=job_id,
                status=JobStatus.UNKNOWN,
                error_details=ErrorDetails(message=f"DB path is not a file: {db_path}")
            )
    except Exception as e:
        logger.error(f"Failed to validate DB path for job status query: {e}", exc_info=True)
        return JobInfo(
            job_id=job_id,
            status=JobStatus.UNKNOWN,
            error_details=ErrorDetails(message=f"Failed to validate DB path: {e}", type=type(e).__name__)
        )

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

# --- New: Function to list all jobs --- #
async def list_jobs(
    db_path: Path,
    session_id: Optional[str] = None,
    status: Optional[Union[str, List[str]]] = None,
    limit: int = 50
) -> List[schemas.JobInfo]:
    """指定された条件でジョブを取得します
    
    Args:
        db_path: データベースファイルのパス
        session_id: セッションIDでフィルタリング（オプション）
        status: JobStatusまたはそのリストでフィルタリング（オプション）
        limit: 最大取得件数
        
    Returns:
        JobInfoオブジェクトのリスト
    """
    _validate_db_path(db_path)
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM jobs"
        params = []
        conditions = []
        
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        
        if status:
            if isinstance(status, str):
                # 単一のステータス文字列の場合
                conditions.append("status = ?")
                params.append(status)
            else:
                # ステータスのリストの場合
                placeholders = ", ".join(["?" for _ in status])
                conditions.append(f"status IN ({placeholders})")
                params.extend(status)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY create_time DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        jobs = []
        for row in cursor.fetchall():
            job_data = dict(row)
            
            # バイナリデータを復元
            if job_data["args"]:
                job_data["args"] = pickle.loads(job_data["args"])
            
            # JobInfoオブジェクトを作成
            job_info = schemas.JobInfo(
                job_id=job_data["job_id"],
                tool_name=job_data["tool_name"],
                status=job_data["status"],
                queue_time=job_data.get("queue_time"),
                start_time=job_data.get("start_time"),
                end_time=job_data.get("end_time"),
                worker_id=job_data.get("worker_id"),
                session_id=job_data.get("session_id"),
                args=job_data.get("args")
            )
            jobs.append(job_info)
        
        return jobs
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()

# --- 新規: アクティブなワーカー情報を取得する関数 --- #
async def get_worker_status(stalled_threshold_seconds: int = 3600) -> Dict[str, Any]:
    """ジョブワーカーの状態情報を取得します
    
    Args:
        stalled_threshold_seconds: この秒数以上実行されているジョブを停滞中と見なす
        
    Returns:
        ワーカー状態の情報を含む辞書
    """
    try:
        current_time = time.time()
        status_info = {
            "active_workers": len(JOB_WORKERS),
            "queue_size": JOB_QUEUE.qsize(),
            "running_jobs": [],
            "potentially_stalled_jobs": []
        }
        
        # 実行中のジョブとその情報を収集
        for worker_id, worker_info in JOB_WORKERS.items():
            if worker_info["current_job"]:
                job_id = worker_info["current_job"]
                job_start_time = worker_info.get("job_start_time", 0)
                running_time = current_time - job_start_time if job_start_time else 0
                
                job_info = {
                    "job_id": job_id,
                    "worker_id": worker_id,
                    "running_time_seconds": int(running_time),
                    "tool_name": worker_info.get("tool_name", "unknown")
                }
                
                # 実行中のジョブのリストに追加
                status_info["running_jobs"].append(job_info)
                
                # 停滞している可能性のあるジョブをチェック
                if running_time > stalled_threshold_seconds:
                    status_info["potentially_stalled_jobs"].append(job_info)
        
        return status_info
    except Exception as e:
        logger.error(f"Failed to get worker status: {e}", exc_info=True)
        raise

# --- Job cleanup function --- #
async def cleanup_old_jobs(config: Dict[str, Any]) -> None:
    """古いジョブレコードをクリーンアップします"""
    cleanup_config = config.get('cleanup', {})
    jobs_cleanup_config = cleanup_config.get('jobs', {})
    
    if not jobs_cleanup_config.get('enabled', False):
        logger.info("ジョブのクリーンアップは無効です。")
        return
    
    # DB パスを取得
    db_path = Path(config.get('paths', {}).get('db', 'mcp.db'))
    if not db_path.is_absolute():
        # 相対パスの場合、workspace_dir からの相対パスとみなす
        try:
            workspace_dir = get_workspace_dir(config)
            db_path = workspace_dir / db_path
        except (ConfigError, FileError) as e:
            logger.error(f"ジョブクリーンアップ: workspace_dir の取得に失敗しました: {e}")
            return
    
    if not db_path.exists():
        logger.error(f"ジョブクリーンアップ: DB ファイルが存在しません: {db_path}")
        return
    
    retention_days = jobs_cleanup_config.get('retention_days', 7)
    if retention_days <= 0:
        logger.warning("ジョブクリーンアップの retention_days が正の値ではありません。クリーンアップをスキップします。")
        return
    
    cutoff_time = time.time() - (retention_days * 24 * 60 * 60)  # 秒単位での保持期間
    
    logger.info(f"古いジョブのクリーンアップを実行します。{retention_days}日以上前のジョブを削除します。")
    
    try:
        # 古いジョブの数を取得
        count_sql = """
            SELECT COUNT(*) FROM jobs 
            WHERE created_at < ? AND status != ?
        """
        row = await db_utils.db_fetch_one_async(db_path, count_sql, (cutoff_time, JobStatus.RUNNING.value))
        if row:
            count = row[0]
            logger.info(f"削除対象の古いジョブ: {count}件")
            
            if count > 0:
                # 古いジョブを削除
                delete_sql = """
                    DELETE FROM jobs 
                    WHERE created_at < ? AND status != ?
                """
                await db_utils.db_execute_commit_async(db_path, delete_sql, (cutoff_time, JobStatus.RUNNING.value))
                logger.info(f"古いジョブを削除しました: {count}件")
        else:
            logger.warning("ジョブ数の取得に失敗しました。")
    
    except Exception as e:
        logger.error(f"ジョブクリーンアップ中にエラーが発生しました: {e}", exc_info=True)

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
    
    # --- 新規: list_jobs 関数を登録 --- #
    @mcp.tool("list_jobs")
    async def list_jobs_tool(session_id: Optional[str] = None, limit: int = 50, status: Optional[Union[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """ジョブの一覧を取得します。特定のセッションIDまたはステータスが指定された場合はフィルタリングします。
        
        Args:
            session_id: フィルタリングするセッションID（オプション）
            limit: 最大取得件数
            status: フィルタリングするステータス（オプション）
                   文字列の場合: "PENDING", "QUEUED", "RUNNING", "COMPLETED", "FAILED", "UNKNOWN" のいずれか
                   リストの場合: 上記ステータスの組み合わせ
        """
        jobs = await list_jobs(db_path, session_id, limit, status)
        # JobInfo オブジェクトをdict()ではなくmodel_dump()で辞書に変換して返す
        return [job.model_dump() for job in jobs]
    
    # --- 新規: 実行中のジョブのみ取得するツールを追加 --- #
    @mcp.tool("get_running_jobs")
    async def get_running_jobs_tool(session_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """実行中のジョブの一覧を取得します。
        
        Args:
            session_id: フィルタリングするセッションID（オプション）
            limit: 最大取得件数
        """
        jobs = await list_jobs(db_path, session_id, limit, status="RUNNING")
        return [job.model_dump() for job in jobs]
        
    # --- 新規: ワーカー状態取得ツールを登録 --- #
    @mcp.tool("get_worker_status")
    async def get_worker_status_tool(stalled_threshold_seconds: int = 3600) -> Dict[str, Any]:
        """ジョブワーカーの状態と情報を取得します。"""
        return await get_worker_status(stalled_threshold_seconds)
    
    logger.info("Job management tools registered.")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 

def register_tools():
    """ジョブ管理関連のツールを登録します。"""
    
    # 既存のコード...
    
    @register_tool
    async def list_jobs_tool(session_id: Optional[str] = None, status: Optional[Union[str, List[str]]] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """登録されているジョブの一覧を取得します。
        
        Args:
            session_id: 特定のセッションIDに限定する場合に指定
            status: ジョブステータスでフィルタリング (PENDING, QUEUED, RUNNING, COMPLETED, FAILED, UNKNOWN)
            limit: 取得する最大件数
            
        Returns:
            ジョブ情報のリスト
        """
        try:
            jobs = await list_jobs(db_path, session_id=session_id, status=status, limit=limit)
            return [job.model_dump() for job in jobs]
        except Exception as e:
            logger.error(f"Error in list_jobs_tool: {e}")
            return []

    @register_tool
    async def get_running_jobs(session_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """実行中のジョブを取得します
        
        Args:
            session_id: セッションIDでフィルタリング（オプション）
            limit: 最大取得件数
            
        Returns:
            実行中のJobInfoオブジェクトのリスト
        """
        try:
            jobs = await list_jobs(db_path, session_id=session_id, status="RUNNING", limit=limit)
            return [job.model_dump() for job in jobs]
        except Exception as e:
            logger.error(f"Error in get_running_jobs: {e}")
            return []

    @register_tool
    async def get_worker_status_tool(stalled_threshold_seconds: int = 3600) -> Dict[str, Any]:
        """ジョブワーカーの状態情報を取得します。
        
        Args:
            stalled_threshold_seconds: この秒数以上実行されているジョブを停滞中と見なす
            
        Returns:
            ワーカーの状態情報（アクティブなワーカー数、キューサイズ、実行中ジョブ、停滞中ジョブなど）
        """
        try:
            return await get_worker_status(stalled_threshold_seconds)
        except Exception as e:
            logger.error(f"Error in get_worker_status_tool: {e}")
            return {"error": str(e)}
    
    # 既存のツール登録...
    
    return {
        "add_job": add_job_tool,
        "cancel_job": cancel_job_tool,
        "get_job": get_job_tool,
        "list_jobs": list_jobs_tool,
        "get_running_jobs": get_running_jobs,
        "get_worker_status": get_worker_status_tool,
    } 