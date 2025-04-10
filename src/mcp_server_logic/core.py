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
from typing import Dict, Any, Optional, Callable, Coroutine, Tuple, AsyncGenerator, Awaitable
# from concurrent.futures import ThreadPoolExecutor # 削除

# 関連モジュールのインポート
from . import db_utils, session_manager # job_manager の直接インポートは削除
from .utils import generate_id, get_timestamp # utilsからインポート
from src.utils.path_utils import get_project_root, load_environment_variables, get_workspace_dir, ensure_dir, get_db_dir, get_output_base_dir # path_utils からインポート
# job_manager から必要なものをインポート
from .job_manager import job_queue, active_jobs # job_queueとactive_jobsをインポート
from .serialization_utils import JsonNumpyEncoder # JsonNumpyEncoder をインポート
import traceback # トレースバック用に追加
# StateManagementError, FileError をインポート
from src.utils.exception_utils import StateManagementError, FileError, ConfigError # ConfigError を追加
# import sqlite3 # DBエラー用 <- 削除

logger = logging.getLogger('mcp_server.core')

# --- Executor (for DB access in async context) ---
# DBアクセスが非同期化されたため、ThreadPoolExecutor は不要になった
# executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1) # 削除

# --- Logging Configuration (Moved from mcp_server.py) ---
# BasicConfig はサーバーエントリポイント (mcp_server.py) で行う想定
# ここでは logger を取得するだけ

# --- Configuration Loading and Setup (Moved & Modified from mcp_server.py) ---
load_environment_variables() # .env をロード
PROJECT_ROOT = get_project_root() # ここで ConfigError の可能性あり

# --- デフォルト設定 --- #
# config.yaml と同じ構造でデフォルト値を定義
DEFAULT_CONFIG = {
    'server': {
        'poll_interval_seconds': 5,
        'job_timeout_seconds': 600,
        'session_timeout_seconds': 3600,
        'request_timeout_seconds': 60,
        'log_level': 'INFO',
        'port': 5002 # uvicorn 起動時にも参照される
    },
    'paths': {
        'data': 'data',
        'audio': 'data/audio',
        'reference': 'data/reference',
        'improved_versions': 'improved_versions',
        'evaluation_results': 'evaluation_results',
        'grid_search_results': 'grid_search_results',
        'visualizations': 'visualizations',
        'scientific_output': 'scientific_output',
        'db': 'db',
        'detectors_src': 'src/detectors'
    },
    'resource_limits': { # デフォルトは configure_resource_limits で動的に決定
        # 'max_concurrent_jobs': 'auto',
        # 'max_jobs_history': 100,
        # 'llm_timeout': 180,
        # 'evaluation_timeout': 1200,
        # 'grid_search_timeout': 1800,
        # 'job_timeout': 600
    },
    'cleanup': {
        'interval_seconds': 3600,
        'session_timeout_seconds': 86400,
        'max_sessions_count': 100,
        'job_stuck_timeout_seconds': 3600,
        'job_completed_retention_seconds': 604800,
        'max_jobs_count': 500,
        'workspace': {
            'enabled': True,
            'retention_days': 14,
            'target_dirs': [
                'evaluation_results', 'grid_search_results', 'improved_versions',
                'visualizations', 'scientific_output'
            ]
        }
    },
    'evaluation': {
        'default_dataset': 'synthesized_v1',
        'default_metrics': ['note.*', 'onset.*'],
        'save_plots': True,
        'save_results_json': True,
        'mir_eval_options': {
            'onset_tolerance': 0.05,
            'pitch_tolerance': 50.0
        }
    },
    'grid_search': {
        'default_best_metric': 'note.f_measure',
        'default_n_jobs': -1
    },
    'llm': {
        'client_type': 'ClaudeClient',
        'api_key': None, # APIキーは環境変数での設定を強く推奨
        'api_key_openai': None,
        'model': 'claude-3-opus-20240229',
        'api_base': 'https://api.anthropic.com',
        'api_version': '2023-06-01',
        'api_base_openai': 'https://api.openai.com/v1',
        'max_tokens': 4096,
        'timeout': 180,
        'desktop_mode': False,
        'desktop_url': 'http://localhost:5000'
    },
    'datasets': { # デフォルトのデータセット定義は最小限に
        'synthesized_v1': {
            'description': 'Basic synthesized dataset',
            'audio_dir': 'datasets/synthesized/audio', # paths.audio を参照する方が良いかも
            'ref_dir': 'datasets/synthesized/labels',   # paths.reference を参照する方が良いかも
            'ref_pattern': '*.csv'
        }
    },
    'detectors': {}
}

def _deep_update(source: Dict, overrides: Dict) -> Dict:
    """辞書を再帰的に更新する"""
    for key, value in overrides.items():
        if isinstance(value, dict):
            # get node or create one
            node = source.setdefault(key, {})
            _deep_update(node, value)
        else:
            source[key] = value
    return source

def _get_env_var_override(config: Dict, prefix: str = 'MCP') -> Dict:
    """環境変数から設定を上書きする辞書を生成する"""
    overrides = {}
    separator = '__' # ネストを示すセパレータ

    def find_and_set(cfg_dict: Dict, current_prefix: str):
        for key, default_value in cfg_dict.items():
            env_var_name = f"{current_prefix}{separator}{key.upper()}"
            env_value = os.environ.get(env_var_name)

            if env_value is not None:
                # 型変換を試みる
                original_type = type(default_value)
                try:
                    if original_type == bool:
                        converted_value = env_value.lower() in ['true', '1', 'yes']
                    elif original_type == int:
                        converted_value = int(env_value)
                    elif original_type == float:
                        converted_value = float(env_value)
                    elif original_type == list:
                        # カンマ区切りの文字列をリストに変換 (要素の型は維持しない)
                        converted_value = [item.strip() for item in env_value.split(',') if item.strip()]
                    else: # str や NoneType など
                        converted_value = env_value

                    # ネストされた辞書構造を再現
                    keys = current_prefix.replace(prefix + separator, '').lower().split(separator)
                    if keys == ['']: keys = []
                    keys.append(key)

                    temp_dict = overrides
                    for i, k_part in enumerate(keys):
                        if i == len(keys) - 1:
                            temp_dict[k_part] = converted_value
                        else:
                            temp_dict = temp_dict.setdefault(k_part, {})

                except (ValueError, TypeError) as e:
                    logger.warning(f"環境変数 '{env_var_name}' の値 '{env_value}' を型 '{original_type.__name__}' に変換できませんでした: {e}")

            # 再帰的に処理
            if isinstance(default_value, dict):
                find_and_set(default_value, env_var_name)

    find_and_set(config, prefix)
    return overrides

def load_config(config_path_str: Optional[str] = None) -> Dict[str, Any]:
    """設定ファイルを読み込み、デフォルト値、YAMLファイル、環境変数でマージする。

    優先順位:
    1. 環境変数 (MCP__GROUP__KEY 形式)
       - 例: `server.log_level` は環境変数 `MCP__SERVER__LOG_LEVEL` で上書き可能。
       - ネストされたキーは `__` で連結します。
       - 値は元の設定値の型 (bool, int, float, list[str], str) に変換されます。
       - bool: 'true', '1', 'yes' (小文字) が True。
       - list[str]: カンマ区切りの文字列。
    2. YAMLファイル (config_path_str で指定)
    3. デフォルト値 (DEFAULT_CONFIG)

    Parameters
    ----------
    config_path_str : Optional[str], optional
        設定ファイル (config.yaml) のパス文字列, by default None

    Returns
    -------
    Dict[str, Any]
        最終的な設定辞書

    Raises
    ------
    ConfigError
        設定ファイルの読み込みや解析に失敗した場合
    """
    # 1. デフォルト設定をロード
    final_config = DEFAULT_CONFIG.copy()

    # --- resource_limits の動的設定 --- #
    dynamic_limits = configure_resource_limits()
    _deep_update(final_config.setdefault('resource_limits', {}), dynamic_limits)

    # 2. YAML ファイルを読み込んで上書き
    yaml_loaded = False
    config_path_to_load: Optional[Path] = None

    if config_path_str:
        config_path = Path(config_path_str).resolve()
        if config_path.exists():
            config_path_to_load = config_path
        else:
             logger.warning(f"指定された設定ファイルが見つかりません: {config_path}。デフォルトを試します。")

    if not config_path_to_load:
        default_yaml_path = PROJECT_ROOT / 'config.yaml'
        if default_yaml_path.exists():
             config_path_to_load = default_yaml_path
        else:
             logger.info("デフォルト設定ファイル config.yaml が見つかりません。デフォルト値と環境変数のみ使用します。")

    if config_path_to_load:
        try:
            with open(config_path_to_load, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
            _deep_update(final_config, yaml_config)
            yaml_loaded = True
            logger.info(f"設定ファイル {config_path_to_load} を読み込み、デフォルト値を上書きしました。")
        except yaml.YAMLError as e:
            msg = f"設定ファイル {config_path_to_load} の解析に失敗しました: {e}"
            logger.error(msg)
            raise ConfigError(msg) from e
        except Exception as e:
            msg = f"設定ファイル {config_path_to_load} の読み込み中に予期せぬエラー: {e}"
            logger.error(msg, exc_info=True)
            raise ConfigError(msg) from e

    # 3. 環境変数で上書き
    env_overrides = _get_env_var_override(final_config)
    if env_overrides:
        _deep_update(final_config, env_overrides)
        logger.info("環境変数から設定を上書きしました。")

    # 4. パスを絶対パスに解決し、存在確認・作成を行うヘルパー関数
    def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """設定内の主要なパスを絶対パスに解決し、一部は存在確認/作成する"""
        try:
            # Project root is already resolved
            cfg['paths']['project_root'] = str(PROJECT_ROOT) # Add project root for reference

            # Workspace path (handled by get_workspace_dir)
            workspace_path = get_workspace_dir(cfg) # Ensure directory exists and is writable
            cfg['paths']['workspace'] = str(workspace_path)

            # Output base path (handled by get_output_base_dir)
            output_base_path = get_output_base_dir(cfg)
            cfg['paths']['output_base'] = str(output_base_path)

            # DB path (handled by get_db_dir)
            db_dir_path = get_db_dir(cfg)
            cfg['paths']['db_dir'] = str(db_dir_path) # Store the directory
            cfg['paths']['db_path'] = str(db_dir_path / db_utils.DB_FILENAME) # Full path to DB file

            # Improved versions path (handled by get_improved_versions_dir)
            improved_versions_path = get_improved_versions_dir(cfg)
            cfg['paths']['improved_versions'] = str(improved_versions_path)

            # Source directories (resolve relative to project root, check existence)
            for key in ['detectors_src', 'templates']: # Add others if needed
                path_str = cfg['paths'].get(key)
                if path_str:
                    path_obj = Path(path_str)
                    if not path_obj.is_absolute():
                        path_obj = PROJECT_ROOT / path_obj
                    resolved_path = path_obj.resolve()
                    if not resolved_path.is_dir(): # Check if source dir exists
                         logger.warning(f"設定されたソースディレクトリ '{key}' が見つかりません: {resolved_path}")
                         # raise ConfigError(f"Source directory '{key}' not found: {resolved_path}") # Or just warn
                    cfg['paths'][key] = str(resolved_path)

            # Datasets base (resolve, check existence)
            datasets_base_str = cfg['paths'].get('datasets_base')
            if datasets_base_str:
                 path_obj = Path(datasets_base_str)
                 if not path_obj.is_absolute():
                      path_obj = PROJECT_ROOT / path_obj
                 resolved_path = path_obj.resolve()
                 if not resolved_path.is_dir():
                     logger.warning(f"データセットベースディレクトリが見つかりません: {resolved_path}")
                     # raise ConfigError(f"Datasets base directory not found: {resolved_path}")
                 cfg['paths']['datasets_base'] = str(resolved_path)

        except (ConfigError, FileError, TypeError) as e:
            # Path utils functions already log errors
            raise ConfigError(f"設定内のパス解決またはディレクトリ検証中にエラーが発生しました: {e}") from e
        except Exception as e:
            logger.error(f"パス解決中に予期せぬエラー: {e}", exc_info=True)
            raise ConfigError(f"パス解決中に予期せぬエラー: {e}") from e
        return cfg

    # 5. パス解決を実行
    try:
        final_config = resolve_paths(final_config)
    except ConfigError as e:
         # load_config は致命的なエラーとして扱う
         logger.critical(f"設定の読み込みとパス解決に失敗しました: {e}", exc_info=True)
         raise

    # 6. ensure_workspace_directories の呼び出しを削除
    # ensure_workspace_directories(final_config)

    # 設定完了のログ
    logger.info("設定の読み込みとパス解決が完了しました。")
    # logger.debug(f"Final Config: {json.dumps(final_config, cls=JsonNumpyEncoder, indent=2)}") # デバッグ用に詳細ログ
    return final_config

# --- Workspace Directory Setup (Removed - Handled in load_config via path_utils) ---
# def ensure_workspace_directories(config: Dict[str, Any]):
#     """ワークスペース関連のディレクトリが存在することを確認し、必要であれば作成する"""
#     # This logic is now integrated into get_workspace_dir, get_db_dir, etc. in path_utils.py
#     # and called during load_config -> resolve_paths.
#     pass

# --- Resource Limit Configuration --- #
def configure_resource_limits() -> Dict[str, Any]:
    """リソース制限関連の設定を決定する（CPU数などに基づいて）。環境変数も考慮。"""
    limits = {}
    # CPU数に基づいてデフォルトの並列数を決定
    try:
        max_workers = os.cpu_count() or 1
        default_jobs = max(1, max_workers - 1) # 1コアはシステム用に残す
    except NotImplementedError:
        default_jobs = 2 # 取得できない場合は適当な値

    # 環境変数 > デフォルト
    env_max_jobs = os.environ.get('MCP_MAX_CONCURRENT_JOBS')
    if env_max_jobs:
        if env_max_jobs.lower() == 'auto':
            limits['max_concurrent_jobs'] = default_jobs
        else:
            try:
                limits['max_concurrent_jobs'] = int(env_max_jobs)
            except ValueError:
                logger.warning(f"環境変数 MCP_MAX_CONCURRENT_JOBS の値 '{env_max_jobs}' は無効です。自動設定 ({default_jobs}) を使用します。")
                limits['max_concurrent_jobs'] = default_jobs
    else:
        limits['max_concurrent_jobs'] = default_jobs

    # その他の制限値 (環境変数から取得、なければデフォルトを使用)
    limits['max_jobs_history'] = int(os.environ.get('MCP_MAX_JOBS_HISTORY', 100))
    limits['llm_timeout'] = int(os.environ.get('MCP_LLM_TIMEOUT', 180))
    limits['evaluation_timeout'] = int(os.environ.get('MCP_EVALUATION_TIMEOUT', 1200))
    limits['grid_search_timeout'] = int(os.environ.get('MCP_GRID_SEARCH_TIMEOUT', 1800))
    # job_timeout は server.job_timeout_seconds と統合すべきかもしれない
    # 注意: ここで final_config を参照できないため、デフォルト値を直接使うか、load_config 内で設定する
    # limits['job_timeout'] = int(os.environ.get('MCP_JOB_TIMEOUT', DEFAULT_CONFIG['server']['job_timeout_seconds']))
    # load_config で server.job_timeout_seconds が設定されるので、ここでは不要と判断

    logger.info(f"リソース制限を設定しました: {limits}")
    return limits

def log_config(config_dict: Dict[str, Any]):
    """設定内容をログに出力する (APIキーなどの機密情報はマスク)"""
    sensitive_keys = ['api_key', 'api_key_openai']

    def _log_recursive(cfg, sensitive):
        log_str = "{\n"
        indent = "  " * (cfg.get('_depth', 0) + 1)
        for key, value in cfg.items():
            if key == '_depth': continue
            if key in sensitive:
                log_str += f"{indent}'{key}': '********',\n"
            elif isinstance(value, dict):
                 value['_depth'] = cfg.get('_depth', 0) + 1
                 log_str += f"{indent}'{key}': {_log_recursive(value, sensitive)},\n"
                 del value['_depth'] # クリーンアップ
            else:
                 log_str += f"{indent}'{key}': {repr(value)},\n"
        log_str += indent[:-2] + "}"
        return log_str

    config_copy = json.loads(json.dumps(config_dict)) # Deep copy to avoid modifying original
    logger.info(f"--- サーバー設定 ---")
    logger.info(_log_recursive(config_copy, sensitive_keys))
    logger.info(f"--------------------")

# --- Job Execution Setup (Moved from mcp_server.py) ---
# job_queue と active_jobs は job_manager.py に移動
# job_queue: asyncio.Queue[Tuple[str, Callable[..., Coroutine], tuple, dict]] = asyncio.Queue()
# active_jobs: Dict[str, Dict[str, Any]] = {} # {job_id: {status: ..., future: ...}}
# ThreadPoolExecutor はブロッキング I/O や CPU バウンドな同期タスクを実行するために使用
# executor = ThreadPoolExecutor(max_workers=os.cpu_count()) # ワーカー数は調整可能


async def job_worker(worker_id: int, config: Dict[str, Any]):
    """非同期ジョブをキューから取得して実行するワーカー (Moved from mcp_server.py, DB操作エラーハンドリング修正)"""
    db_path = Path(config['paths']['db']) / db_utils.DB_FILENAME
    logger.info(f"ジョブワーカー {worker_id} 開始")
    while True:
        job_id, task_coro_func, args, kwargs = await job_queue.get()
        tool_name = kwargs.get('tool_name', 'unknown') # tool_name を取得
        logger.info(f"[Worker {worker_id}] [Job {job_id}] ジョブ '{tool_name}' を取得しました。")
        if job_id not in active_jobs:
            logger.warning(f"[Worker {worker_id}] [Job {job_id}] Job info not found in active_jobs dict.")
            # 見つからない場合でも、初期情報を active_jobs に作成
            active_jobs[job_id] = {
                'status': 'pending',
                'tool_name': tool_name,
                'start_time': None,
                'end_time': None,
                'result': None,
                'error_details': None,
                'worker_id': None,
                'task_args': kwargs # kwargs全体を保存するか検討
            }

        active_jobs[job_id]['status'] = 'running'
        active_jobs[job_id]['worker_id'] = worker_id
        start_time_mono = time.monotonic()
        start_time_ts = get_timestamp()
        active_jobs[job_id]['start_time'] = start_time_ts

        job_result = None
        job_error = None
        error_details_dict = None # エラー詳細を格納する辞書

        try:
            # DBにジョブ開始を記録 (エラーハンドリング修正)
            try:
                # --- 修正: 開始時もロックを追加 (念のため) --- #
                async with db_utils.db_lock:
                    await db_utils.db_execute_commit_async(
                        db_path,
                        "UPDATE jobs SET status = 'running', start_time = ?, worker_id = ? WHERE job_id = ?",
                        (start_time_ts, str(worker_id), job_id)
                    )
            except StateManagementError as db_err:
                 # DB記録エラーは警告ログを出すが、ジョブ実行は試みる
                 logger.warning(f"[Worker {worker_id}] [Job {job_id}] ジョブ開始DB記録エラー: {db_err}", exc_info=False)
                 # 致命的ではないので、ここでは raise しない

            # --- コルーチン関数を実行 --- #
            job_result = await task_coro_func(job_id, *args, **kwargs)
            # --- 実行ここまで --- #

            end_time_mono = time.monotonic()
            duration = end_time_mono - start_time_mono
            end_time_ts = get_timestamp()
            logger.info(f"[Worker {worker_id}] [Job {job_id}] ジョブ完了。実行時間: {duration:.2f}秒")
            active_jobs[job_id]['status'] = 'completed'
            active_jobs[job_id]['result'] = job_result
            active_jobs[job_id]['end_time'] = end_time_ts # メモリにも終了時間を記録

            # DBにジョブ完了を記録 (エラーハンドリング修正)
            try:
                result_json = json.dumps(job_result, cls=JsonNumpyEncoder)
                # --- 修正: カラム名修正 (completed_at -> end_time) --- #
                update_sql = "UPDATE jobs SET status = ?, result = ?, end_time = ? WHERE job_id = ?"
                async with db_utils.db_lock:
                    await db_utils.db_execute_commit_async(
                        db_path,
                        update_sql,
                        ('completed', result_json, end_time_ts, job_id)
                    )
                logger.debug(f"ジョブ {job_id} の状態を completed に更新")
            except StateManagementError as db_err:
                 # DB記録エラーは警告ログを出すが、ジョブ自体は完了扱い
                 logger.warning(f"[Worker {worker_id}] [Job {job_id}] ジョブ完了DB記録エラー: {db_err}", exc_info=False)
                 # ここでも raise しない

        except Exception as e:
            # --- ジョブ実行中のエラー処理 --- #
            job_error = e # エラーオブジェクトを保持
            end_time_mono = time.monotonic()
            duration = end_time_mono - start_time_mono
            end_time_ts = get_timestamp()
            error_message = f"Error in job {job_id} ({kwargs.get('tool_name', 'unknown')})" # 基本メッセージ
            tb_str = traceback.format_exc() # トレースバックを取得

            # StateManagementError かどうかでログレベルとメッセージを調整
            if isinstance(e, StateManagementError):
                logger.error(f"[Worker {worker_id}] [Job {job_id}] 状態管理エラー発生: {e}。実行時間: {duration:.2f}秒", exc_info=False) # トレースバックは別途記録
                error_message = f"State Management Error: {e}"
            else:
                logger.error(f"[Worker {worker_id}] [Job {job_id}] 予期せぬエラー発生: {type(e).__name__}: {e}。実行時間: {duration:.2f}秒", exc_info=False)
                error_message = f"Unexpected Error: {type(e).__name__}: {e}"

            # メモリ上のジョブ情報を更新
            active_jobs[job_id]['status'] = 'failed'
            active_jobs[job_id]['end_time'] = end_time_ts
            active_jobs[job_id]['result'] = None # 失敗時は result は None
            # エラー詳細を辞書に格納
            error_details_dict = {
                 "error_type": type(e).__name__,
                 "error_message": str(e),
                 "traceback": tb_str
            }
            active_jobs[job_id]['error_details'] = error_details_dict

            # DBにジョブ失敗を記録 (エラーハンドリング修正)
            try:
                error_details_json = json.dumps(error_details_dict)
                # --- 修正: カラム名修正 (error -> error_details, completed_at -> end_time) --- #
                update_sql = "UPDATE jobs SET status = ?, error_details = ?, end_time = ? WHERE job_id = ?"
                async with db_utils.db_lock:
                    await db_utils.db_execute_commit_async(
                        db_path,
                        update_sql,
                        ('failed', error_details_json, end_time_ts, job_id)
                    )
                logger.debug(f"ジョブ {job_id} の状態を failed に更新 (エラー)")
            except StateManagementError as db_err:
                 # DB記録エラーは警告ログを出す (ジョブ失敗は既に確定)
                 logger.warning(f"[Worker {worker_id}] [Job {job_id}] ジョブ失敗DB記録エラー: {db_err}", exc_info=False)

        finally:
            job_queue.task_done()
            logger.debug(f"[Worker {worker_id}] [Job {job_id}] Task marked as done.")

async def start_job_workers(num_workers: int, config: Dict[str, Any]):
    """指定された数のジョブワーカーを起動 (Moved from mcp_server.py)"""
    logger.info(f"{num_workers} 個のジョブワーカーを起動します...")
    tasks = []
    for i in range(num_workers):
        task = asyncio.create_task(job_worker(i + 1, config))
        tasks.append(task)
    logger.info("ジョブワーカーが起動しました。")
    # ワーカータスクの完了を待つ必要はない (サーバーが動いている限り動き続ける)

async def start_async_job(
    config: Dict[str, Any],
    task_coroutine_func: Callable[..., Coroutine], # 同期関数ではなくコルーチン関数を受け取る
    tool_name: str,
    session_id: Optional[str] = None,
    # session_id_for_job: Optional[str] = None, # 引数名が重複するので削除し、session_id を使う
    *args, **kwargs
) -> str:
    """非同期ジョブをキューに追加し、DBに登録する"""
    job_id = generate_id(prefix=f"{tool_name[:5]}_j")
    logger.info(f"Queueing job {job_id} for tool '{tool_name}' (Session: {session_id})")
    current_time = get_timestamp()

    # kwargs に tool_name を追加（job_worker で利用するため）
    kwargs_with_tool = kwargs.copy()
    kwargs_with_tool['tool_name'] = tool_name

    # ジョブをDBに登録
    try:
        db_path = Path(config['paths']['db_path'])
        task_args_json = json.dumps({"args": args, "kwargs": kwargs}, cls=JsonNumpyEncoder) # 引数をJSON化
        async with db_utils.db_lock: # INSERT もロック
            await db_utils.db_execute_commit_async(
                db_path,
                "INSERT INTO jobs (job_id, tool_name, session_id, status, start_time, task_args) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, tool_name, session_id, 'pending', current_time, task_args_json)
            )
    except (ConfigError, FileError, StateManagementError) as e:
         logger.error(f"Failed to add job {job_id} to DB: {e}", exc_info=False)
         raise StateManagementError(f"Failed to register job {job_id} in database.") from e
    except Exception as e:
         logger.error(f"Unexpected error registering job {job_id} in DB: {e}", exc_info=True)
         raise StateManagementError(f"Unexpected error registering job {job_id}.") from e

    # ジョブキューに追加
    try:
         # --- 修正: job_worker が期待する形式に合わせる --- #
         await job_queue.put( (job_id, task_coroutine_func, args, kwargs_with_tool) ) # put_nowait を put に変更 (キューが一杯なら待機)
         active_jobs[job_id] = {
             'status': 'pending',
             'tool_name': tool_name,
             'session_id': session_id,
             'start_time': None, # DBには登録済みだが、メモリ上はまだ開始していない
             'task_args': {"args": args, "kwargs": kwargs}, # kwargs は tool_name を含まない元のもの
             'queued_time': current_time
         } # In-memory status
         logger.debug(f"Job {job_id} successfully added to queue.")
    except asyncio.QueueFull:
         # put は待機するので、QueueFull は通常発生しないはずだが、念のため
         logger.error(f"Job queue is full. Failed to queue job {job_id}.")
         try:
             # --- 修正: 非同期DB関数を使用 --- #
             async with db_utils.db_lock:
                 await db_utils.db_execute_commit_async(
                     db_path,
                     "UPDATE jobs SET status = 'failed', error_details = ? WHERE job_id = ?",
                     (json.dumps({"error": "Queue full"}, cls=JsonNumpyEncoder), job_id)
                 )
         except Exception as db_e:
              logger.error(f"Failed to update status for job {job_id} after queue full error: {db_e}")
         raise StateManagementError(f"Job queue is full, cannot add job {job_id}.")
    except Exception as e:
         logger.error(f"Error adding job {job_id} to queue: {e}", exc_info=True)
         # キュー追加失敗時もDBのステータスを更新
         try:
             async with db_utils.db_lock:
                 await db_utils.db_execute_commit_async(
                     db_path,
                     "UPDATE jobs SET status = 'failed', error_details = ? WHERE job_id = ?",
                     (json.dumps({"error": f"Queueing error: {e}"}, cls=JsonNumpyEncoder), job_id)
                 )
         except Exception as db_e:
              logger.error(f"Failed to update status for job {job_id} after queueing error: {db_e}")
         raise StateManagementError(f"Error adding job {job_id} to queue: {e}") from e

    return job_id

# --- Cleanup Setup (Moved & Modified from mcp_server.py) ---
async def cleanup_workspace_files(config: Dict[str, Any]):
    """古いワークスペースファイルやディレクトリを削除する (非同期)"""
    cleanup_config = config.get('cleanup', {})
    workspace_cleanup_config = cleanup_config.get('workspace', {})

    if not workspace_cleanup_config.get('enabled', False):
        logger.info("Workspace file cleanup is disabled.")
        return

    retention_days = workspace_cleanup_config.get('retention_days', 14)
    if retention_days <= 0:
        logger.warning("Workspace cleanup retention_days is not positive, skipping cleanup.")
        return

    cutoff_time = datetime.now() - timedelta(days=retention_days)
    logger.info(f"Running workspace file cleanup. Deleting items older than {cutoff_time} ({retention_days} days)...")

    try:
        # ターゲットディレクトリのベースパスを取得 (output_base)
        output_base_dir = get_output_base_dir(config) # path_utils を使用
        target_base_dirs = [output_base_dir]

        # (オプション) workspace 内の特定のサブディレクトリも対象にする場合
        # workspace_dir = get_workspace_dir(config)
        # target_base_dirs.append(workspace_dir / 'some_subdir_to_clean')

    except (ConfigError, FileError) as e:
         logger.error(f"クリーンアップ対象のベースディレクトリ取得に失敗しました: {e}. クリーンアップを中止します。")
         return

    # target_dirs の設定は現在無視し、output_base 内を走査する方式に変更
    # target_subdirs = workspace_cleanup_config.get('target_dirs', [])
    # logger.info(f"Targeting subdirectories: {target_subdirs} within workspace/output paths.")

    deleted_count = 0
    error_count = 0

    for base_dir in target_base_dirs:
         if not base_dir.is_dir():
             logger.warning(f"クリーンアップ対象のベースディレクトリが存在しません: {base_dir}")
             continue

         logger.info(f"Scanning directory for cleanup: {base_dir}")
         # asyncio.to_thread を使用して同期的なファイル操作を非同期実行
         loop = asyncio.get_running_loop()

         # iterdir は同期的だが、多数のファイルがある場合は非同期化も検討
         items_to_check = list(base_dir.iterdir())

         for item in items_to_check:
             try:
                 # サブディレクトリのみを対象とする (ファイルは無視)
                 # is_dir() もブロッキングの可能性があるので to_thread 化
                 is_directory = await loop.run_in_executor(None, item.is_dir)
                 if not is_directory:
                     continue

                 # stat() もブロッキングなので to_thread 化
                 try:
                     stat_result = await loop.run_in_executor(None, item.stat)
                     item_mtime = datetime.fromtimestamp(stat_result.st_mtime)
                 except OSError as stat_err:
                     logger.warning(f"Could not get modification time for {item}: {stat_err}. Skipping.")
                     continue

                 if item_mtime < cutoff_time:
                     logger.info(f"Deleting old workspace directory: {item} (Last modified: {item_mtime}) ")
                     # shutil.rmtree もブロッキングなので to_thread 化
                     await loop.run_in_executor(None, shutil.rmtree, item)
                     deleted_count += 1
                 # else: logger.debug(f"Keeping recent directory: {item} (Last modified: {item_mtime})")

             except Exception as e:
                 logger.error(f"Error cleaning up item {item}: {e}", exc_info=True)
                 error_count += 1

    logger.info(f"Workspace cleanup finished. Deleted {deleted_count} old directories. Encountered {error_count} errors.")


# --- 修正: start_cleanup_thread を非同期タスクに変更 --- #
async def start_cleanup_task(
    config: Dict[str, Any],
    # 非同期化されたクリーンアップ関数を受け取る
    cleanup_sessions_func: Callable[[Dict[str, Any]], Awaitable[None]],
    cleanup_jobs_func: Callable[[Dict[str, Any]], Awaitable[None]],
    cleanup_workspace_func: Callable[[Dict[str, Any]], Awaitable[None]]
):
    """定期的にクリーンアップ処理を実行する非同期タスクを開始"""
    interval = config.get('cleanup', {}).get('interval_seconds', 3600)
    if interval <= 0:
        logger.info("クリーンアップ間隔が無効なため、クリーンアップタスクは開始されません。")
        return

    async def cleanup_run():
        logger.info(f"クリーンアップタスク開始 (実行間隔: {interval}秒)")
        while True:
            try:
                logger.info("定期クリーンアップ処理を実行中...")
                # 非同期クリーンアップ関数を await で実行
                await cleanup_sessions_func(config)
                await cleanup_jobs_func(config)
                await cleanup_workspace_func(config)
                logger.info("定期クリーンアップ処理完了。")
            except asyncio.CancelledError:
                 logger.info("クリーンアップタスクがキャンセルされました。")
                 break # キャンセルされたらループを抜ける
            except Exception as e:
                logger.error(f"クリーンアップ処理中にエラーが発生: {e}", exc_info=True)
            # 次の実行まで非同期で待機
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                 logger.info("クリーンアップタスク待機中にキャンセルされました。")
                 break # 待機中にキャンセルされた場合もループを抜ける

    # asyncio.create_task でバックグラウンドタスクとして起動
    task = asyncio.create_task(cleanup_run(), name="CleanupTask")
    logger.info("クリーンアップタスクがバックグラウンドでスケジュールされました。")
    return task # タスクオブジェクトを返す (オプション)

# --- 古い同期スレッド開始関数は削除 --- #
# def start_cleanup_thread(
#     config: Dict[str, Any],
#     cleanup_sessions_func: Callable[[Dict[str, Any]], None], # session_manager.cleanup_old_sessions の想定
#     cleanup_jobs_func: Callable[[Dict[str, Any]], None] # job_manager.cleanup_old_jobs の想定
# ):
# ... (削除) ...

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