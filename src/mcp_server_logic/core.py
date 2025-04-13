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
from . import db_utils # from . import db_utils, utils # utilsからインポート
from src.utils.misc_utils import generate_id, get_timestamp # 変更後
from src.utils.path_utils import get_project_root, load_environment_variables, get_workspace_dir, ensure_dir, get_db_dir, get_output_base_dir, get_improved_versions_dir, validate_path_within_allowed_dirs # validate_path_within_allowed_dirs を追加
# job_manager から必要なものをインポート
# from .job_manager import job_queue, active_jobs # job_queueとactive_jobsをインポート # 削除
# job_manager モジュール自体をインポートする
from . import job_manager
from src.utils.json_utils import NumpyEncoder # JsonNumpyEncoder をインポート (旧 JsonNumpyEncoder)
import traceback # トレースバック用に追加
# StateManagementError, FileError をインポート
from src.utils.exception_utils import StateManagementError, FileError, ConfigError # ConfigError を追加
# import sqlite3 # DBエラー用 <- 削除
from . import schemas # 追加
from .db_utils import db_execute_commit_async, db_lock # 追加

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
        # 'api_key': None, # APIキーは環境変数での設定を強く推奨 - ホスト側で管理するため削除
        # 'api_key_openai': None, # ホスト側で管理するため削除
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
    _deep_update(final_config, env_overrides)

    # 4. パス解決 (resolve_paths に移行せず、ここで実行)
    # ★ resolve_paths 関数を削除し、以下のロジックをここに展開する
    # final_config = resolve_paths(final_config)

    # --- パス解決ロジックここから (元 resolve_paths の内容) --- #
    try:
        workspace_root = get_workspace_dir() # ここで PROJECT_ROOT が使われる
        logger.debug(f"Workspace root identified as: {workspace_root}")

        path_cfg = final_config.get('paths', {})

        # 各パス設定を処理
        for key, relative_path_str in path_cfg.items():
            if not isinstance(relative_path_str, str):
                logger.warning(f"パス設定 '.paths.{key}' の値が文字列ではありません: {relative_path_str}。スキップします。")
                continue

            # 絶対パスに変換 (ワークスペースルート基準)
            absolute_path = (workspace_root / relative_path_str).resolve()

            # パスの安全性を検証 (ワークスペース内に収まっているか)
            try:
                validate_path_within_allowed_dirs(
                    absolute_path,
                    [workspace_root], # 許可ディレクトリはワークスペースルートのみ
                    check_existence=False, # 存在チェックは ensure_dir で行う
                    allow_absolute=True # 検証対象は絶対パス
                )
            except (ValueError, FileError) as path_err:
                 # 安全でないパス設定はエラーとして処理 (設定ファイルの誤り)
                 raise ConfigError(f"パス設定 '.paths.{key}' ('{relative_path_str}') がワークスペース外または不正です: {path_err}") from path_err

            # 存在確認とディレクトリ作成 (必要に応じて)
            # DBパス、detectors_src はファイルかもしれないので is_dir=False とする
            is_dir_expected = key not in ['db', 'detectors_src']
            try:
                # ensure_dir はパスがファイルでもディレクトリでも作成しようとする
                # ディレクトリを期待する場合は is_dir=True でチェック
                # ここではパスの解決と安全確認が主目的なので、ディレクトリ作成はオプションとする
                # 実際にパスを使う際に ensure_dir を呼び出す方が適切かもしれない
                # ensure_dir(absolute_path, is_dir=is_dir_expected, check_writable=is_dir_expected)
                pass # 存在確認・作成はここでは行わない

            except FileError as fe:
                 # ディレクトリ作成失敗などのエラー
                 raise ConfigError(f"パス設定 '.paths.{key}' ('{absolute_path}') の準備に失敗しました: {fe}") from fe

            # 設定辞書のパスを絶対パス文字列で更新
            path_cfg[key] = str(absolute_path)

        # --- パス解決ロジックここまで --- #

    except ConfigError: # ConfigError はそのまま上に投げる
        raise
    except Exception as e:
        logger.critical(f"設定ファイルの読み込み/パス解決中に予期せぬエラー: {e}", exc_info=True)
        raise ConfigError(f"設定処理中に予期せぬエラーが発生しました: {e}") from e

    # 5. 設定内容をログ出力 (APIキーなどはマスク)
    log_config(final_config)

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

# --- Job Execution Setup (Moved to job_manager.py) ---
# job_queue: asyncio.Queue[Tuple[str, Callable[..., Coroutine], tuple, dict]] = asyncio.Queue()
# active_jobs: Dict[str, Dict[str, Any]] = {}


# --- Job Worker Functions (Moved to job_manager.py) --- #

# async def job_worker(worker_id: int, config: Dict[str, Any]):
#     """ジョブキューからタスクを取得して実行するワーカ"""
#     pass # 削除
#
# async def start_job_workers(num_workers: int, config: Dict[str, Any]):
#     """指定された数のジョブワーカータスクを開始する"""
#     pass # 削除
#
# async def start_async_job(
#     config: Dict[str, Any],
#     task_coroutine_func: Callable[..., Coroutine],
#     tool_name: str,
#     session_id: Optional[str] = None,
#     *args, **kwargs
# ) -> str:
#     """非同期ジョブを開始し、DBに登録してキューに入れる"""
#     pass # 削除


# --- Workspace Cleanup --- #
async def cleanup_workspace_files(config: Dict[str, Any]):
    """古いワークスペースファイルを削除する"""
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
                 # 既存のファイルかディレクトリかチェック
                 is_directory = item.is_dir()
                 if not is_directory:
                     continue

                 # ファイルの場合、最終アクセス時刻をチェック
                 try:
                     stat_result = item.stat()
                     last_access_time = stat_result.st_atime
                 except OSError as e:
                     logger.warning(f"stat取得エラー: {item} - {e}")
                     continue

                 if last_access_time < cutoff_time:
                     logger.info(f"期限切れディレクトリを削除: {item}")
                     try:
                         await loop.run_in_executor(None, shutil.rmtree, item)
                         deleted_count += 1
                     except Exception as e:
                         logger.error(f"Error cleaning up item {item}: {e}", exc_info=True)
                         error_count += 1

             except Exception as e:
                 logger.error(f"Error cleaning up item {item}: {e}", exc_info=True)
                 error_count += 1

    logger.info(f"Workspace cleanup finished. Deleted {deleted_count} old directories. Encountered {error_count} errors.")


# --- 修正: start_cleanup_thread を非同期タスクに変更 --- #
async def start_cleanup_task(
    config: Dict[str, Any],
    # 非同期化されたクリーンアップ関数を受け取る
    # 引数の型ヒントを修正
    cleanup_sessions_func: Callable[[Dict[str, Any]], Awaitable[None]],
    cleanup_jobs_func: Callable[[Dict[str, Any]], Awaitable[None]],
    cleanup_workspace_func: Callable[[Dict[str, Any]], Awaitable[None]]
):
    """定期的なクリーンアップタスクを開始する"""
    interval = config.get('cleanup', {}).get('interval_seconds', 3600)
    if interval <= 0:
        logger.info("定期クリーンアップは無効です (interval <= 0)。")
        return None # タスクを返さない

    logger.info(f"定期クリーンアップタスクを開始します (間隔: {interval}秒)。")

    async def cleanup_run():
        while True:
            try:
                await asyncio.sleep(interval)
                logger.info("定期クリーンアップを開始します...")

                # 並列実行ではなく、順番に実行する方が安全かもしれない
                start_time = time.monotonic()

                # 1. セッションのクリーンアップ
                logger.debug("古いセッションのクリーンアップを開始...")
                await cleanup_sessions_func(config)
                logger.debug("セッションのクリーンアップ完了。")

                # 2. ジョブのクリーンアップ
                logger.debug("古いジョブのクリーンアップを開始...")
                # cleanup_jobs_func は job_manager.cleanup_old_jobs を想定
                await cleanup_jobs_func(config)
                logger.debug("ジョブのクリーンアップ完了。")

                # 3. ワークスペースファイルのクリーンアップ
                logger.debug("ワークスペースファイルのクリーンアップを開始...")
                await cleanup_workspace_func(config)
                logger.debug("ワークスペースファイルのクリーンアップ完了。")

                duration = time.monotonic() - start_time
                logger.info(f"定期クリーンアップ完了 (所要時間: {duration:.2f}秒)。次の実行は {interval}秒後。")

            except asyncio.CancelledError:
                logger.info("クリーンアップタスクがキャンセルされました。")
                break
            except Exception as e:
                 logger.error(f"定期クリーンアップ実行中にエラーが発生しました: {e}", exc_info=True)
                 # エラーが発生しても次のインターバルで再試行するためにループは継続
                 # ただし、頻繁にエラーが発生する場合は警告を出すなどの処理を追加検討
                 await asyncio.sleep(interval) # エラー後もインターバル待機

    # クリーンアップタスクをバックグラウンドで開始
    task = asyncio.create_task(cleanup_run(), name="cleanup_task")
    logger.info("定期クリーンアップタスクがバックグラウンドでスケジュールされました。")
    return task # 開始したタスクを返す

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