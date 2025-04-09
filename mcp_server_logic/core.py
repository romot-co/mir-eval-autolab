import os
import json
import yaml
import logging
import asyncio
import time
import threading
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine, Tuple
from concurrent.futures import ThreadPoolExecutor

# 関連モジュールのインポート
from . import db_utils, session_manager # job_manager の直接インポートは削除
from .utils import generate_id, get_timestamp # utilsからインポート
from src.utils.path_utils import get_project_root, load_environment_variables, get_workspace_dir, ensure_dir # path_utils からインポート
# job_manager から必要なものをインポート
from .job_manager import job_queue, active_jobs # job_queueとactive_jobsをインポート
from .serialization_utils import JsonNumpyEncoder # JsonNumpyEncoder をインポート
import traceback # トレースバック用に追加
# StateManagementError, FileError をインポート
from src.utils.exception_utils import StateManagementError, FileError
import sqlite3 # DBエラー用

logger = logging.getLogger('mcp_server.core')

# --- Logging Configuration (Moved from mcp_server.py) ---
# BasicConfig はサーバーエントリポイント (mcp_server.py) で行う想定
# ここでは logger を取得するだけ

# --- Configuration Loading and Setup (Moved & Modified from mcp_server.py) ---
load_environment_variables() # .env をロード
PROJECT_ROOT = get_project_root()

def load_config(config_path: Optional[Path] = None) -> dict:
    """設定ファイルを読み込む (デフォルトパス or 指定パス)"""
    if config_path is None:
        config_path = PROJECT_ROOT / 'config.yaml'

    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            logger.info(f"設定ファイル {config_path} を読み込みました。")
        except Exception as e:
            logger.warning(f"{config_path} の読み込みに失敗しました: {e}")
            config_data = {}
    else:
        logger.warning(f"{config_path} が見つかりません。デフォルト設定を使用します。")
        config_data = {}

    # --- デフォルト値、パス、リソース制限、タイムアウト、クリーンアップ設定の適用 ---
    # workspace_dir の確定
    workspace_dir_str = os.environ.get("MIREX_WORKSPACE") or config_data.get('paths', {}).get('workspace')
    if not workspace_dir_str:
        workspace_dir_str = str(PROJECT_ROOT / 'mcp_workspace')
        logger.warning(f"ワークスペースパスが未指定のため、デフォルト ({workspace_dir_str}) を使用します。")
    workspace_dir = Path(workspace_dir_str).resolve()

    # デフォルトパス
    default_paths = {
        "workspace": str(workspace_dir),
        "detectors": str(PROJECT_ROOT / "src" / "detectors"),
        "improved_versions": str(workspace_dir / "improved_versions"),
        "evaluation_results": str(workspace_dir / "evaluation_results"),
        "grid_search_results": str(workspace_dir / "grid_search_results"),
        "data": str(workspace_dir / "data"),
        "audio": str(workspace_dir / "data" / "audio"), # デフォルトを修正
        "reference": str(workspace_dir / "data" / "reference"), # デフォルトを修正
        "visualizations": str(workspace_dir / "visualizations"),
        "scientific_output": str(workspace_dir / "scientific_output"),
        "db": str(workspace_dir / "db") # DBディレクトリ
    }

    # リソース制限
    resource_limits = configure_resource_limits()

    # タイムアウト
    default_timeouts = {
        'llm': resource_limits['llm_timeout'],
        'evaluation': resource_limits['evaluation_timeout'],
        'grid_search': resource_limits['grid_search_timeout'],
        'job': resource_limits['job_timeout']
    }

    # クリーンアップ設定
    default_cleanup = {
        'session_timeout_seconds': 86400, 'job_stuck_timeout_seconds': 3600,
        'job_completed_retention_seconds': 604800, 'max_sessions_count': 100,
        'max_jobs_count': 500, 'interval_seconds': 3600,
        'workspace': {
            'enabled': True, 'retention_days': 14,
            'target_dirs': ['evaluation_results', 'grid_search_results', 'improved_versions', 'visualizations', 'scientific_output', 'db'] # dbも追加
        }
    }

    # --- 設定のマージ ---
    final_config = {}
    final_config['paths'] = {**default_paths, **config_data.get('paths', {})}
    final_config['resource_limits'] = {**resource_limits, **config_data.get('resource_limits', {})}
    final_config['timeouts'] = {**default_timeouts, **config_data.get('timeouts', {})}
    final_config['cleanup'] = {**default_cleanup, **config_data.get('cleanup', {})} # Deep merge is better
    final_config['server'] = config_data.get('server', {})
    final_config['llm'] = config_data.get('llm', {})
    final_config['evaluation'] = config_data.get('evaluation', {})
    final_config['grid_search'] = config_data.get('grid_search', {})
    final_config['datasets'] = config_data.get('datasets', {})
    # ... 他のトップレベル設定 ...

    # 環境変数による上書き (例: APIキー、タイムアウト)
    final_config['llm']['api_key'] = os.environ.get("ANTHROPIC_API_KEY", final_config['llm'].get('api_key'))
    final_config['timeouts']['llm'] = int(os.environ.get('MCP_LLM_TIMEOUT', final_config['timeouts']['llm']))
    # ... 他の環境変数上書き ...

    return final_config

def configure_resource_limits() -> Dict[str, Any]:
    """実行環境に基づいてリソース制限を設定 (Moved from mcp_server.py)"""
    is_desktop = os.environ.get("CLAUDE_DESKTOP_MODE", "false").lower() in ["true", "1", "yes"]
    cpu_count = os.cpu_count() or 4
    # Default timeouts based on environment type
    if is_desktop:
        timeouts = {"llm_timeout": 120, "evaluation_timeout": 600, "grid_search_timeout": 900, "job_timeout": 300}
        max_concurrent = max(min(cpu_count // 2, 2), 1)
    else:
        timeouts = {"llm_timeout": 180, "evaluation_timeout": 1200, "grid_search_timeout": 1800, "job_timeout": 600}
        max_concurrent = max(min(cpu_count - 1, 4), 2)

    # Allow environment variables to override specific limits
    limits = {
        "max_concurrent_jobs": int(os.environ.get("MCP_MAX_CONCURRENT_JOBS", max_concurrent)),
        "max_jobs_history": int(os.environ.get("MCP_MAX_JOBS_HISTORY", 100)),
        **timeouts # Merge default timeouts
    }
    logger.info(f"リソース制限: 同時実行={limits['max_concurrent_jobs']}, 履歴={limits['max_jobs_history']}")
    return limits

def log_config(config_dict: Dict[str, Any]):
    """設定内容をログに出力 (機密情報マスク) (Moved from mcp_server.py)"""
    loggable_config = {}
    sensitive_keys = ['api_key', 'password', 'secret']
    def _log_recursive(cfg, sensitive):
        loggable = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                loggable[k] = _log_recursive(v, sensitive)
            elif any(s_key in k.lower() for s_key in sensitive):
                loggable[k] = "******"
            else:
                loggable[k] = v
        return loggable

    loggable_config = _log_recursive(config_dict, sensitive_keys)
    try:
        config_str = json.dumps(loggable_config, indent=2, default=str)
        logger.info(f"読み込まれた設定:\n{config_str}")
    except Exception as e:
        logger.error(f"設定のログ出力に失敗: {e}")
        logger.info(f"Raw Config (may contain sensitive data): {config_dict}")


def ensure_workspace_directories(config: Dict[str, Any]) -> bool:
    """設定されたワークスペースディレクトリが存在することを確認・作成 (Moved from mcp_server.py)"""
    logger.info("ワークスペースディレクトリを確認・作成中...")
    paths_to_ensure = [
        config['paths']['workspace'],
        config['paths']['improved_versions'],
        config['paths']['evaluation_results'],
        config['paths']['grid_search_results'],
        config['paths']['data'],
        config['paths']['audio'],
        config['paths']['reference'],
        config['paths']['visualizations'],
        config['paths']['scientific_output'],
        config['paths']['db'] # DBディレクトリも確認
    ]
    try:
        for path_str in paths_to_ensure:
            ensure_dir(Path(path_str))
        logger.info("ワークスペースディレクトリの準備完了。")
        return True
    except Exception as e:
        logger.error(f"ワークスペースディレクトリの作成に失敗しました: {e}", exc_info=True)
        # FileError でラップ
        raise FileError(f"Failed to ensure workspace directories: {e}") from e

# --- Job Execution Setup (Moved from mcp_server.py) ---
# job_queue と active_jobs は job_manager.py に移動
# job_queue: asyncio.Queue[Tuple[str, Callable[..., Coroutine], tuple, dict]] = asyncio.Queue()
# active_jobs: Dict[str, Dict[str, Any]] = {} # {job_id: {status: ..., future: ...}}
# ThreadPoolExecutor はブロッキング I/O や CPU バウンドな同期タスクを実行するために使用
executor = ThreadPoolExecutor(max_workers=os.cpu_count()) # ワーカー数は調整可能


async def job_worker(worker_id: int, config: Dict[str, Any]):
    """非同期ジョブをキューから取得して実行するワーカー (Moved from mcp_server.py, DB操作エラーハンドリング修正)"""
    db_path = Path(config['paths']['db']) / db_utils.DB_FILENAME
    logger.info(f"ジョブワーカー {worker_id} 開始")
    while True:
        job_id, task_coro_func, args, kwargs = await job_queue.get()
        logger.info(f"[Worker {worker_id}] [Job {job_id}] ジョブ '{kwargs.get('tool_name', 'unknown')}' を取得しました。")
        if job_id not in active_jobs:
            logger.warning(f"[Worker {worker_id}] [Job {job_id}] Job info not found in active_jobs dict.")
            active_jobs[job_id] = {'status': 'pending', 'tool_name': kwargs.get('tool_name', 'unknown')}

        active_jobs[job_id]['status'] = 'running'
        active_jobs[job_id]['worker_id'] = worker_id
        start_time_mono = time.monotonic()
        start_time_ts = get_timestamp()

        try:
            # DBにジョブ開始を記録 (エラーハンドリング追加)
            try:
                await db_utils.db_execute_commit_async(
                    db_path,
                    "UPDATE jobs SET status = 'running', start_time = ?, worker_id = ? WHERE job_id = ?",
                    (start_time_ts, str(worker_id), job_id)
                )
            except (sqlite3.Error, Exception) as db_err:
                 # ログ記録の失敗はジョブ自体の成否に影響させない
                 logger.error(f"[Worker {worker_id}] [Job {job_id}] ジョブ開始DB記録エラー: {db_err}", exc_info=True)
                 # StateManagementError を raise しない

            # コルーチン関数を実行
            result = await task_coro_func(job_id, *args, **kwargs)

            end_time_mono = time.monotonic()
            duration = end_time_mono - start_time_mono
            end_time_ts = get_timestamp()
            logger.info(f"[Worker {worker_id}] [Job {job_id}] ジョブ完了。実行時間: {duration:.2f}秒")
            active_jobs[job_id]['status'] = 'completed'
            active_jobs[job_id]['result'] = result
            active_jobs[job_id]['end_time'] = end_time_ts # メモリにも終了時間を記録

            # DBにジョブ完了を記録 (エラーハンドリング追加)
            try:
                await db_utils.db_execute_commit_async(
                    db_path,
                    "UPDATE jobs SET status = 'completed', end_time = ?, result = ? WHERE job_id = ?",
                    (end_time_ts, json.dumps(result, cls=JsonNumpyEncoder), job_id)
                )
            except (sqlite3.Error, Exception) as db_err:
                 logger.error(f"[Worker {worker_id}] [Job {job_id}] ジョブ完了DB記録エラー: {db_err}", exc_info=True)

        except Exception as e:
            end_time_mono = time.monotonic()
            duration = end_time_mono - start_time_mono
            end_time_ts = get_timestamp()
            error_message = f"Error in job {job_id} ({kwargs.get('tool_name', 'unknown')}): {type(e).__name__}: {e}"
            error_traceback = traceback.format_exc()
            logger.error(f"[Worker {worker_id}] {error_message}", exc_info=True)
            active_jobs[job_id]['status'] = 'failed'
            active_jobs[job_id]['error'] = error_message
            active_jobs[job_id]['traceback'] = error_traceback
            active_jobs[job_id]['end_time'] = end_time_ts # メモリにも終了時間とエラー情報を記録

            # DBにジョブ失敗を記録 (エラーハンドリング追加)
            try:
                await db_utils.db_execute_commit_async(
                    db_path,
                    "UPDATE jobs SET status = 'failed', end_time = ?, result = ? WHERE job_id = ?",
                    (end_time_ts, json.dumps({"error": error_message, "traceback": error_traceback}), job_id)
                )
            except (sqlite3.Error, Exception) as db_err:
                 logger.error(f"[Worker {worker_id}] [Job {job_id}] ジョブ失敗DB記録エラー: {db_err}", exc_info=True)

        finally:
            job_queue.task_done()

async def start_job_workers(num_workers: int, config: Dict[str, Any]):
    """指定された数のジョブワーカーを起動 (Moved from mcp_server.py)"""
    logger.info(f"{num_workers} 個のジョブワーカーを起動します...")
    tasks = []
    for i in range(num_workers):
        task = asyncio.create_task(job_worker(i + 1, config))
        tasks.append(task)
    logger.info("ジョブワーカーが起動しました。")
    # ワーカータスクの完了を待つ必要はない (サーバーが動いている限り動き続ける)

def start_async_job(
    config: Dict[str, Any],
    task_coroutine_func: Callable[..., Coroutine], # 同期関数ではなくコルーチン関数を受け取る
    tool_name: str,
    session_id: Optional[str] = None,
    session_id_for_job: Optional[str] = None, # 引数名が重複するので変更
    *args, **kwargs
) -> str:
    """
    非同期ジョブをキューに追加し、ジョブIDを返す (Moved & Modified from mcp_server.py)。
    task_coroutine_func は job_id を第一引数として受け取るコルーチンである必要がある。
    """
    # Use the specific session_id_for_job argument
    actual_session_id = session_id_for_job or session_id
    db_path = Path(config['paths']['db']) / db_utils.DB_FILENAME
    job_id = generate_id()
    logger.info(f"[Job {job_id}] ジョブ '{tool_name}' をキューに追加します。Session: {actual_session_id}")

    # アクティブジョブリストに追加 (job_manager からインポートされたものを使用)
    active_jobs[job_id] = {
        'status': 'pending',
        'tool_name': tool_name,
        'session_id': actual_session_id,
        'start_time': get_timestamp(),
        'args': args,
        'kwargs': kwargs
    }

    # DBにジョブ情報を記録 (同期的に実行、ワーカー開始前に必要)
    try:
        db_utils._db_execute_commit( # 同期関数を使用
            db_path,
            """INSERT INTO jobs (job_id, session_id, tool_name, status, start_time, task_args)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (job_id, actual_session_id, tool_name, 'pending', get_timestamp(), json.dumps({'args': args, 'kwargs': kwargs}, cls=JsonNumpyEncoder))
        )
    except Exception as e:
         logger.error(f"[Job {job_id}] Failed to record job in DB: {e}", exc_info=True)
         # ジョブ開始前にDB記録失敗したらエラーを返すか、処理を中止すべきか？ -> ここではログのみ
         # active_jobs から削除する必要があるかもしれない
         if job_id in active_jobs: del active_jobs[job_id]
         raise # エラーを再発生

    # ジョブキューに追加
    # kwargs に tool_name を含めてワーカーに渡す
    kwargs_for_worker = kwargs.copy()
    kwargs_for_worker['tool_name'] = tool_name
    job_queue.put_nowait((job_id, task_coroutine_func, args, kwargs_for_worker))

    return job_id

# --- Cleanup Setup (Moved & Modified from mcp_server.py) ---
def cleanup_workspace_files(config: Dict[str, Any]):
    """古いワークスペースファイルを削除 (Moved from mcp_server.py)"""
    cleanup_config = config.get('cleanup', {}).get('workspace', {})
    if not cleanup_config.get('enabled', False):
        logger.info("ワークスペースのクリーンアップは無効です。")
        return

    retention_days = cleanup_config.get('retention_days', 14)
    target_dirs_short = cleanup_config.get('target_dirs', [])
    workspace_base = Path(config['paths']['workspace'])
    target_dirs_full = [workspace_base / d for d in target_dirs_short if (workspace_base / d).exists()]

    if not target_dirs_full:
        logger.info("クリーンアップ対象のワークスペースディレクトリが見つかりません。")
        return

    cutoff_time = datetime.now() - timedelta(days=retention_days)
    cutoff_timestamp = cutoff_time.timestamp()
    logger.info(f"ワークスペースのクリーンアップ開始 ({retention_days}日以上前のファイルを削除)... Cutoff: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")
    deleted_count = 0
    error_count = 0

    for target_dir in target_dirs_full:
        logger.debug(f"ディレクトリをスキャン中: {target_dir}")
        try:
            for item in target_dir.iterdir():
                try:
                    item_stat = item.stat()
                    # ファイル/ディレクトリの最終変更時刻で判断
                    if item_stat.st_mtime < cutoff_timestamp:
                        if item.is_file():
                            item.unlink()
                            logger.debug(f"  削除(ファイル): {item}")
                            deleted_count += 1
                        elif item.is_dir():
                            shutil.rmtree(item)
                            logger.debug(f"  削除(ディレクトリ): {item}")
                            deleted_count += 1
                except Exception as item_e:
                    logger.warning(f"  アイテム {item} の処理中にエラー: {item_e}")
                    error_count += 1
        except Exception as dir_e:
             logger.error(f"ディレクトリ {target_dir} のスキャン中にエラー: {dir_e}")
             error_count += 1

    logger.info(f"ワークスペースのクリーンアップ完了。{deleted_count} 個のアイテムを削除しました。{error_count} 個のエラーが発生しました。")


# start_cleanup_thread は循環参照を避けるため、session_manager と job_manager をインポートせず、
# 必要なクリーンアップ関数を引数として受け取るように変更する
def start_cleanup_thread(
    config: Dict[str, Any],
    cleanup_sessions_func: Callable[[Dict[str, Any]], None], # session_manager.cleanup_old_sessions の想定
    cleanup_jobs_func: Callable[[Dict[str, Any]], None] # job_manager.cleanup_old_jobs の想定
):
    """
    定期的にクリーンアップ処理を実行するバックグラウンドスレッドを開始
    (Moved & Modified from mcp_server.py)
    """
    interval = config.get('cleanup', {}).get('interval_seconds', 3600)
    if interval <= 0:
        logger.info("クリーンアップ間隔が無効なため、クリーンアップスレッドは開始されません。")
        return

    def cleanup_run():
        logger.info(f"クリーンアップスレッド開始 (実行間隔: {interval}秒)")
        while True:
            try:
                logger.info("定期クリーンアップ処理を実行中...")
                # セッションのクリーンアップ
                cleanup_sessions_func(config)
                # ジョブのクリーンアップ
                cleanup_jobs_func(config)
                # ワークスペースファイルのクリーンアップ
                cleanup_workspace_files(config)
                logger.info("定期クリーンアップ処理完了。")
            except Exception as e:
                logger.error(f"クリーンアップ処理中にエラーが発生: {e}", exc_info=True)
            # 次の実行まで待機
            time.sleep(interval)

    cleanup_thread = threading.Thread(target=cleanup_run, daemon=True, name="CleanupThread")
    cleanup_thread.start()
    logger.info("クリーンアップスレッドがバックグラウンドで開始されました。")


# JSON Encoder for Numpy types (moved to a central place if needed, or keep here if only core needs it)
# class JsonNumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
#                             np.int16, np.int32, np.int64, np.uint8,
#                             np.uint16, np.uint32, np.uint64)):
#             return int(obj)
#         elif isinstance(obj, (np.float_, np.float16, np.float32,
#                               np.float64)):
#             return float(obj)
#         elif isinstance(obj, (np.ndarray,)): # リストに変換
#             return obj.tolist()
#         elif isinstance(obj, np.bool_):
#             return bool(obj)
#         elif isinstance(obj, (np.void)): # void 型は None に変換するなど、適切な処理を検討
#             return None
#         elif isinstance(obj, Path): # Path オブジェクトを文字列に変換
#             return str(obj)
#         return super(JsonNumpyEncoder, self).default(obj) 