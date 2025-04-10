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
from src.utils.exception_utils import StateManagementError, FileError, ConfigError # ConfigError を追加
import sqlite3 # DBエラー用

logger = logging.getLogger('mcp_server.core')

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
    # configure_resource_limits は環境変数も考慮するため、
    # 環境変数オーバーライドの前に実行する。
    dynamic_limits = configure_resource_limits()
    _deep_update(final_config.setdefault('resource_limits', {}), dynamic_limits)

    # 2. YAML ファイルを読み込んで上書き
    if config_path_str:
        config_path = Path(config_path_str)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f) or {}
                _deep_update(final_config, yaml_config)
                logger.info(f"設定ファイル {config_path} を読み込み、デフォルト値を上書きしました。")
            except yaml.YAMLError as e:
                msg = f"設定ファイル {config_path} の解析に失敗しました: {e}"
                logger.error(msg)
                raise ConfigError(msg) from e
            except Exception as e:
                msg = f"設定ファイル {config_path} の読み込み中に予期せぬエラー: {e}"
                logger.error(msg, exc_info=True)
                raise ConfigError(msg) from e
        else:
            logger.warning(f"指定された設定ファイルが見つかりません: {config_path}。デフォルト値と環境変数のみ使用します。")
    else:
        # デフォルトの config.yaml を試す
        default_yaml_path = PROJECT_ROOT / 'config.yaml'
        if default_yaml_path.exists():
            try:
                with open(default_yaml_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f) or {}
                _deep_update(final_config, yaml_config)
                logger.info(f"デフォルト設定ファイル {default_yaml_path} を読み込み、デフォルト値を上書きしました。")
            except yaml.YAMLError as e:
                logger.warning(f"デフォルト設定ファイル {default_yaml_path} の解析に失敗: {e}。無視します。")
            except Exception as e:
                 logger.warning(f"デフォルト設定ファイル {default_yaml_path} の読み込みエラー: {e}。無視します。")
        else:
             logger.info("デフォルト設定ファイル config.yaml が見つかりません。デフォルト値を使用します。")

    # 3. 環境変数で上書き
    env_overrides = _get_env_var_override(DEFAULT_CONFIG) # デフォルト構造を元に探索
    if env_overrides:
        _deep_update(final_config, env_overrides)
        logger.info(f"環境変数 (MCP__* 形式) で設定を上書きしました。")
        # logger.debug(f"環境変数による上書き内容: {json.dumps(env_overrides, indent=2)}")

    # 4. パスの絶対パス化 (ワークスペース基準 or プロジェクトルート基準)
    #    get_workspace_dir は config を必要とする場合があるので、ここで実行
    try:
        workspace_dir = get_workspace_dir(final_config)
    except (FileError, PermissionError, ConfigError) as path_err:
         # ワークスペースが準備できない場合は致命的エラー
         logger.critical(f"ワークスペースディレクトリの準備に失敗: {path_err}")
         raise ConfigError(f"ワークスペースディレクトリの準備に失敗: {path_err}") from path_err

    final_config['paths']['workspace'] = str(workspace_dir) # 確定したパスを設定

    for key, value in final_config.get('paths', {}).items():
        if key == 'workspace': continue # 既に絶対パス
        if value: # 空でないパスのみ処理
             p = Path(value)
             if not p.is_absolute():
                 # デフォルトではワークスペースからの相対パスとみなす
                 # (detectors_src など一部はプロジェクトルート基準が良いかもしれない)
                 base_dir = workspace_dir
                 if key == 'detectors_src': # 例外的にプロジェクトルート基準
                      base_dir = PROJECT_ROOT
                 final_config['paths'][key] = str((base_dir / p).resolve())
             else:
                  # 絶対パス指定の場合はそのまま (resolve で正規化)
                  final_config['paths'][key] = str(p.resolve())

    # datasets 内のパスも同様に絶対パス化 (環境変数展開後)
    for name, dataset_cfg in final_config.get('datasets', {}).items():
        for key in ['audio_dir', 'ref_dir']:
            if key in dataset_cfg and dataset_cfg[key]:
                 p = Path(dataset_cfg[key])
                 if not p.is_absolute():
                      # デフォルトではプロジェクトルート基準とみなす (データセットはプロジェクト内に置かれる想定)
                      dataset_cfg[key] = str((PROJECT_ROOT / p).resolve())
                 else:
                      dataset_cfg[key] = str(p.resolve())

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
            # 見つからない場合でも、初期情報を active_jobs に作成
            active_jobs[job_id] = {
                'status': 'pending',
                'tool_name': kwargs.get('tool_name', 'unknown'),
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
                update_sql = "UPDATE jobs SET status = ?, result = ?, completed_at = ? WHERE job_id = ?"
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
                update_sql = "UPDATE jobs SET status = ?, error = ?, completed_at = ? WHERE job_id = ?"
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