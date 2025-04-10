import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine, List, Awaitable
import asyncio # asyncio をインポート
from functools import partial # partial をインポート

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP
from src.evaluation.evaluation_runner import run_evaluation as run_evaluation_core # 同期関数
from src.evaluation.grid_search.core import run_grid_search as run_grid_search_core # 同期関数
from src.utils import path_utils
from src.utils.path_utils import get_output_dir, ensure_dir, validate_path_within_allowed_dirs, get_output_base_dir, get_workspace_dir, get_dataset_paths, get_project_root # get_project_root を追加
from src.utils.json_utils import NumpyEncoder
from src.utils.exception_utils import EvaluationError, GridSearchError, FileError, ConfigError # FileError, ConfigError をインポート

logger = logging.getLogger('mcp_server.evaluation_tools')

# --- Synchronous Task Functions --- #

def _run_evaluate(
    job_id: str,
    evaluation_results_dir: Path,
    add_history_sync_func: Callable,
    **kwargs
) -> Dict[str, Any]:
    """評価ジョブを実行します (同期)"""
    detector_name = kwargs.get('detector_name')
    session_id = kwargs.get('session_id')
    logger.info(f"[Job {job_id}] Running evaluation for '{detector_name}'... Session: {session_id}")

    # 出力ディレクトリの設定 (session_id や job_id を含める)
    if 'output_dir' not in kwargs or not kwargs['output_dir']:
        eval_results_base_dir = evaluation_results_dir
        unique_suffix = f"session_{session_id or 'nosession'}_job_{job_id}"
        kwargs['output_dir'] = str(get_output_dir(eval_results_base_dir, unique_suffix))
    try:
        ensure_dir(Path(kwargs['output_dir']))
    except Exception as e:
        logger.error(f"[Job {job_id}] Failed to create evaluation output directory: {e}", exc_info=True)
        raise EvaluationError(f"Failed to create output directory: {e}") from e

    # 履歴追加: 開始
    if session_id:
        try:
             add_history_sync_func(session_id, "evaluation_started", {"detector": detector_name, "params": kwargs})
        except Exception as hist_e:
             logger.warning(f"[Job {job_id}] Failed to add evaluation started history: {hist_e}")

    try:
        # run_evaluation_core は同期関数と仮定
        # kwargs には detector_name, dataset_name, audio_path などが含まれる
        # run_evaluation_core が結果辞書を返すことを期待
        results = run_evaluation_core(**kwargs)
        logger.info(f"[Job {job_id}] Evaluation finished for '{detector_name}'.")

        # 履歴追加: 完了
        if session_id:
             try:
                 # 結果のサマリーなどを履歴に追加すると良い
                 summary = results.get(detector_name, {}).get("overall_metrics", {})
                 f_measure = summary.get('note', {}).get('f_measure', 'N/A')
                 add_history_sync_func(session_id, "evaluation_complete", {"detector": detector_name, "f_measure": f_measure, "summary": summary})
             except Exception as hist_e:
                 logger.warning(f"[Job {job_id}] Failed to add evaluation complete history: {hist_e}")

        return results

    except Exception as e:
        logger.error(f"[Job {job_id}] Error during evaluation for '{detector_name}': {e}", exc_info=True)
        # 履歴追加: 失敗
        if session_id:
            try:
                 add_history_sync_func(session_id, "evaluation_failed", {"detector": detector_name, "error": str(e)})
            except Exception as hist_e:
                 logger.warning(f"[Job {job_id}] Failed to add evaluation failed history: {hist_e}")
        # EvaluationError を raise するか、元の例外を再 raise するか
        raise EvaluationError(f"Evaluation failed for {detector_name}: {e}") from e

def _execute_grid_search(
    job_id: str,
    grid_search_results_dir: Path,
    add_history_sync_func: Callable,
    **kwargs
) -> Dict[str, Any]:
    """グリッドサーチジョブを実行します (同期)"""
    config_path = kwargs.get('config_path')
    session_id = kwargs.get('session_id')
    logger.info(f"[Job {job_id}] Running grid search using config '{config_path}'... Session: {session_id}")

    # 設定ファイルから detector_name を読み込む試み (履歴用)
    detector_name_from_config = "unknown"
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            grid_config_content = yaml.safe_load(f)
            detector_name_from_config = grid_config_content.get('detector_name', 'unknown')
    except Exception as e:
        logger.warning(f"[Job {job_id}] Failed to read detector name from grid config '{config_path}': {e}")

    # 出力ディレクトリの設定
    if 'output_dir' not in kwargs or not kwargs['output_dir']:
        grid_results_base_dir = grid_search_results_dir
        unique_suffix = f"session_{session_id or 'nosession'}_job_{job_id}"
        kwargs['output_dir'] = str(get_output_dir(grid_results_base_dir, unique_suffix))
    try:
        ensure_dir(Path(kwargs['output_dir']))
    except Exception as e:
        logger.error(f"[Job {job_id}] Failed to create grid search output directory: {e}", exc_info=True)
        raise GridSearchError(f"Failed to create output directory: {e}") from e

    # 履歴追加: 開始
    if session_id:
        try:
            add_history_sync_func(session_id, "grid_search_started", {"config_path": config_path, "detector": detector_name_from_config})
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add grid search started history: {hist_e}")

    try:
        # run_grid_search_core は同期関数と仮定
        # kwargs には config_path, output_dir, num_procs などが含まれる
        results = run_grid_search_core(**kwargs)
        logger.info(f"[Job {job_id}] Grid search finished. Output: {kwargs['output_dir']}")

        # 履歴追加: 完了
        if session_id:
            try:
                best_params = results.get('best_params')
                best_score = results.get('best_score')
                add_history_sync_func(session_id, "grid_search_complete", {"detector": detector_name_from_config, "best_params": best_params, "best_score": best_score})
            except Exception as hist_e:
                logger.warning(f"[Job {job_id}] Failed to add grid search complete history: {hist_e}")

        # 結果には output_directory を含めるようにする (run_grid_search_core が返すか確認)
        if isinstance(results, dict) and 'output_directory' not in results:
             results['output_directory'] = kwargs['output_dir']
        return results

    except Exception as e:
        logger.error(f"[Job {job_id}] Error during grid search: {e}", exc_info=True)
        # 履歴追加: 失敗
        if session_id:
            try:
                add_history_sync_func(session_id, "grid_search_failed", {"detector": detector_name_from_config, "error": str(e)})
            except Exception as hist_e:
                logger.warning(f"[Job {job_id}] Failed to add grid search failed history: {hist_e}")
        raise GridSearchError(f"Grid search failed: {e}") from e

# --- Asynchronous Task Wrapper Functions --- #
# 同期コア関数を非同期で実行するためのラッパー

async def _run_evaluate_async(
    job_id: str,
    output_base_dir: Path, # 出力ベースディレクトリを受け取る
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]], # 非同期履歴追加関数
    config: Dict[str, Any], # config も受け取る (データセットパス取得のため)
    user_output_dir: Optional[str], # ユーザー指定の出力ディレクトリ (オプション)
    **kwargs
) -> Dict[str, Any]:
    """評価ジョブを非同期で実行します (コアは同期)

    Args:
        job_id (str): ジョブID
        output_base_dir (Path): 出力先のベースディレクトリ
        add_history_async_func (Callable[..., Awaitable[Dict[str, Any]]]): 非同期履歴追加関数
        config (Dict[str, Any]): アプリケーション設定
        user_output_dir (Optional[str]): ユーザーが指定した出力ディレクトリ (Noneの場合、自動生成)
        **kwargs: run_evaluation_core に渡す引数 (detector_name, dataset_name など)

    Returns:
        Dict[str, Any]: 評価結果

    Raises:
        EvaluationError: 評価中にエラーが発生した場合
        ConfigError: パス設定に問題がある場合
        FileError: ファイルアクセスに問題がある場合
    """
    detector_name = kwargs.get('detector_name')
    session_id = kwargs.get('session_id')
    logger.info(f"[Job {job_id}] Async Wrapper: Running evaluation for '{detector_name}'... Session: {session_id}")

    # --- 出力ディレクトリ決定 ---
    final_output_dir: Path
    try:
        if user_output_dir:
            # ユーザー指定の場合、output_base_dir 配下にあるか検証
            logger.info(f"[Job {job_id}] User specified output directory: {user_output_dir}")
            # allowed_base_dirs はリストである必要あり
            # allow_absolute=True は user_output_dir が絶対パスの場合に必要
            final_output_dir = validate_path_within_allowed_dirs(
                user_output_dir,
                allowed_base_dirs=[output_base_dir],
                check_existence=False, # 存在しなくてもOK (ensure_dirで作成)
                allow_absolute=True # ユーザーは絶対パスで指定する可能性あり
            )
            ensure_dir(final_output_dir) # 検証後にディレクトリ作成/確認
            logger.info(f"[Job {job_id}] Using validated user-specified output directory: {final_output_dir}")
        else:
            # ユーザー指定がない場合、自動生成
            logger.info(f"[Job {job_id}] No output directory specified by user, generating automatically.")
            unique_suffix = f"evaluation_{session_id or 'nosession'}_{job_id}"
            # get_output_dir はパスを構築するだけ
            final_output_dir = get_output_dir(output_base_dir, unique_suffix=unique_suffix)
            ensure_dir(final_output_dir) # ここでディレクトリ作成
            logger.info(f"[Job {job_id}] Generated output directory: {final_output_dir}")

        # run_evaluation_core に渡すパスを更新
        kwargs['output_dir'] = str(final_output_dir)

    except (ValueError, FileError, TypeError, ConfigError) as e: # ConfigErrorも捕捉
        logger.error(f"[Job {job_id}] Failed to prepare evaluation output directory: {e}", exc_info=True)
        # エラー発生時はジョブ失敗として履歴に残したい
        if session_id:
            try:
                await add_history_async_func(session_id, "evaluation_failed", {"detector": detector_name, "error": f"Output directory handling failed: {e}"})
            except Exception as hist_e:
                logger.warning(f"[Job {job_id}] Failed to add history for output dir error: {hist_e}")
        raise EvaluationError(f"Failed to prepare output directory: {e}") from e

    # --- 入力パス検証 ---
    # allowed_dirs を設定 (プロジェクトルート、ワークスペース、データセットベース)
    try:
        project_root = get_project_root()
        workspace_dir = get_workspace_dir(config)
        datasets_base_dir = Path(config.get('paths', {}).get('datasets_base', 'datasets'))
        if not datasets_base_dir.is_absolute():
            datasets_base_dir = project_root / datasets_base_dir

        allowed_dirs = [project_root, workspace_dir, datasets_base_dir]
        # データセット名が指定されている場合は、そのデータセットのディレクトリも許可
        if kwargs.get('dataset_name'):
            try:
                audio_d, label_d, _, _ = get_dataset_paths(config, kwargs['dataset_name'])
                allowed_dirs.extend([audio_d, label_d])
            except (ConfigError, FileError) as e:
                 logger.warning(f"[Job {job_id}] Could not get dataset paths for validation: {e}")
                 # エラーでも続行するが警告を出す

        # audio_path, ref_path があれば検証
        if kwargs.get('audio_path'):
             kwargs['audio_path'] = str(validate_path_within_allowed_dirs(kwargs['audio_path'], allowed_dirs, check_existence=True, check_is_file=True))
        if kwargs.get('ref_path'):
             kwargs['ref_path'] = str(validate_path_within_allowed_dirs(kwargs['ref_path'], allowed_dirs, check_existence=True, check_is_file=True))

    except (ValueError, FileError, ConfigError) as e:
         logger.error(f"[Job {job_id}] Input path validation failed: {e}", exc_info=True)
         if session_id:
             try:
                 await add_history_async_func(session_id, "evaluation_failed", {"detector": detector_name, "error": f"Input path validation failed: {e}"})
             except Exception as hist_e:
                 logger.warning(f"[Job {job_id}] Failed to add history for path validation error: {hist_e}")
         raise EvaluationError(f"Input path validation failed: {e}") from e

    # 履歴追加: 開始
    if session_id:
        try:
             await add_history_async_func(session_id, "evaluation_started", {"detector": detector_name, "params": kwargs})
        except Exception as hist_e:
             logger.warning(f"[Job {job_id}] Failed to add evaluation started history: {hist_e}")

    try:
        # 同期コア関数を別スレッドで実行
        loop = asyncio.get_running_loop()
        sync_task = partial(run_evaluation_core, **kwargs)
        results = await loop.run_in_executor(None, sync_task)
        logger.info(f"[Job {job_id}] Evaluation finished for '{detector_name}'.")

        # 履歴追加: 完了
        if session_id:
             try:
                 # summary = results.get(detector_name, {}).get("overall_metrics", {}) # run_evaluation_core の戻り値構造に依存
                 summary = results.get("overall_summary", {})
                 # f_measure = summary.get('note', {}).get('f_measure', 'N/A') # 内部構造が変わった可能性
                 f_measure_dict = summary.get('detectors', {}).get(detector_name, {}).get('metrics', {}).get('note', {})
                 f_measure = f_measure_dict.get('f_measure_mean', 'N/A') # mean を取得

                 await add_history_async_func(session_id, "evaluation_complete", {
                     "detector": detector_name,
                     "f_measure": f_measure,
                     "output_directory": str(final_output_dir), # 出力ディレクトリも履歴に
                     "summary": summary # サマリー全体を入れる (大きすぎないか注意)
                 })
             except Exception as hist_e:
                 logger.warning(f"[Job {job_id}] Failed to add evaluation complete history: {hist_e}")

        # 結果に実際に出力されたディレクトリパスを含める (run_evaluation_core が既に含めているはずだが念のため)
        if isinstance(results, dict) and 'output_directory' not in results:
            results['output_directory'] = str(final_output_dir)

        return results

    except Exception as e:
        # asyncio.CancelledError などもここでキャッチされる
        if isinstance(e, asyncio.CancelledError):
            logger.warning(f"[Job {job_id}] Evaluation task for '{detector_name}' was cancelled.")
            # 履歴追加はしない (サーバー側で cancelled ステータスになるはず)
            raise # CancelledError は再発生させる

        logger.error(f"[Job {job_id}] Error during evaluation for '{detector_name}': {e}", exc_info=True)
        # 履歴追加: 失敗
        if session_id:
            try:
                 await add_history_async_func(session_id, "evaluation_failed", {"detector": detector_name, "error": str(e)})
            except Exception as hist_e:
                 logger.warning(f"[Job {job_id}] Failed to add evaluation failed history: {hist_e}")
        raise EvaluationError(f"Evaluation failed for {detector_name}: {e}") from e

async def _execute_grid_search_async(
    job_id: str,
    output_base_dir: Path,
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    config: Dict[str, Any],
    user_output_dir: Optional[str], # ユーザー指定の出力ディレクトリ (オプション)
    **kwargs
) -> Dict[str, Any]:
    """グリッドサーチジョブを非同期で実行します (コアは同期)

    Args:
        job_id (str): ジョブID
        output_base_dir (Path): 出力先のベースディレクトリ
        add_history_async_func (Callable[..., Awaitable[Dict[str, Any]]]): 非同期履歴追加関数
        config (Dict[str, Any]): アプリケーション設定
        user_output_dir (Optional[str]): ユーザーが指定した出力ディレクトリ (Noneの場合、自動生成)
        **kwargs: run_grid_search_core に渡す引数 (config_path など)

    Returns:
        Dict[str, Any]: グリッドサーチ結果

    Raises:
        GridSearchError: グリッドサーチ中にエラーが発生した場合
        ConfigError: パス設定に問題がある場合
        FileError: ファイルアクセスに問題がある場合
    """
    config_path = kwargs.get('config_path')
    session_id = kwargs.get('session_id')
    logger.info(f"[Job {job_id}] Async Wrapper: Running grid search using config '{config_path}'... Session: {session_id}")

    # --- 出力ディレクトリ決定 ---
    final_output_dir: Path
    try:
        if user_output_dir:
            # ユーザー指定の場合、output_base_dir 配下にあるか検証
            logger.info(f"[Job {job_id}] User specified output directory: {user_output_dir}")
            final_output_dir = validate_path_within_allowed_dirs(
                user_output_dir,
                allowed_base_dirs=[output_base_dir],
                check_existence=False,
                allow_absolute=True
            )
            ensure_dir(final_output_dir)
            logger.info(f"[Job {job_id}] Using validated user-specified output directory: {final_output_dir}")
        else:
            # ユーザー指定がない場合、自動生成
            logger.info(f"[Job {job_id}] No output directory specified by user, generating automatically.")
            unique_suffix = f"grid_search_{session_id or 'nosession'}_{job_id}"
            final_output_dir = get_output_dir(output_base_dir, unique_suffix=unique_suffix)
            ensure_dir(final_output_dir)
            logger.info(f"[Job {job_id}] Generated output directory: {final_output_dir}")

        # run_grid_search_core に渡すパスを更新
        kwargs['output_dir'] = str(final_output_dir)

    except (ValueError, FileError, TypeError, ConfigError) as e: # ConfigErrorも捕捉
        logger.error(f"[Job {job_id}] Failed to prepare grid search output directory: {e}", exc_info=True)
        if session_id:
            try:
                await add_history_async_func(session_id, "grid_search_failed", {"config_path": config_path, "error": f"Output directory handling failed: {e}"})
            except Exception as hist_e:
                logger.warning(f"[Job {job_id}] Failed to add history for output dir error: {hist_e}")
        raise GridSearchError(f"Failed to prepare output directory: {e}") from e

    # --- 入力パス検証 (Grid Search Config) ---
    # allowed_dirs を設定 (プロジェクトルート、ワークスペース)
    try:
        project_root = get_project_root()
        workspace_dir = get_workspace_dir(config)
        allowed_dirs = [project_root, workspace_dir]

        # config_path を検証
        kwargs['config_path'] = str(validate_path_within_allowed_dirs(kwargs['config_path'], allowed_dirs, check_existence=True, check_is_file=True))

    except (ValueError, FileError, ConfigError) as e:
         logger.error(f"[Job {job_id}] Grid search config path validation failed: {e}", exc_info=True)
         if session_id:
             try:
                 await add_history_async_func(session_id, "grid_search_failed", {"config_path": config_path, "error": f"Grid search config path validation failed: {e}"})
             except Exception as hist_e:
                 logger.warning(f"[Job {job_id}] Failed to add history for config path error: {hist_e}")
         raise GridSearchError(f"Grid search config path validation failed: {e}") from e

    # --- 履歴追加: 開始 ---
    detector_name_from_config = "unknown" # Keep this logic
    try:
        import yaml
        with open(kwargs['config_path'], 'r', encoding='utf-8') as f:
            grid_config_content = yaml.safe_load(f)
            detector_name_from_config = grid_config_content.get('detector_name', 'unknown')
    except Exception as e:
        logger.warning(f"[Job {job_id}] Failed to read detector name from grid config '{kwargs['config_path']}': {e}")
        # detector_name は unknown のまま続行
    except yaml.YAMLError as e:
        logger.error(f"[Job {job_id}] Failed to read detector name from grid config '{kwargs['config_path']}': Invalid YAML - {e}")
        # detector_name は unknown のまま続行

    if session_id:
        try:
            await add_history_async_func(session_id, "grid_search_started", {"config_path": kwargs['config_path'], "detector": detector_name_from_config})
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add grid search started history: {hist_e}")

    # --- 同期コア関数実行 ---
    try:
        # 同期コア関数を別スレッドで実行
        loop = asyncio.get_running_loop()
        sync_task = partial(run_grid_search_core, **kwargs)
        results = await loop.run_in_executor(None, sync_task)
        logger.info(f"[Job {job_id}] Grid search finished. Output: {kwargs['output_dir']}")

        # --- 履歴追加: 完了 ---
        if session_id:
            try:
                best_params = results.get('best_params')
                best_score = results.get('best_score')
                await add_history_async_func(session_id, "grid_search_complete", {
                    "detector": detector_name_from_config,
                    "best_params": best_params,
                    "best_score": best_score,
                    "output_directory": str(final_output_dir)
                })
            except Exception as hist_e:
                logger.warning(f"[Job {job_id}] Failed to add grid search complete history: {hist_e}")

        # Ensure output_directory is in results
        if isinstance(results, dict) and 'output_directory' not in results:
             results['output_directory'] = kwargs['output_dir']
        return results

    except Exception as e:
        if isinstance(e, asyncio.CancelledError):
             logger.warning(f"[Job {job_id}] Grid search task was cancelled.")
             raise

        logger.error(f"[Job {job_id}] Error during grid search: {e}", exc_info=True)
        # 履歴追加: 失敗
        if session_id:
            try:
                await add_history_async_func(session_id, "grid_search_failed", {"detector": detector_name_from_config, "error": str(e)})
            except Exception as hist_e:
                logger.warning(f"[Job {job_id}] Failed to add grid search failed history: {hist_e}")
        raise GridSearchError(f"Grid search failed: {e}") from e

# --- Tool Registration --- #

def register_evaluation_tools(
    mcp: FastMCP,
    config: Dict[str, Any],
    start_async_job_func: Callable[..., Coroutine[Any, Any, str]], # 戻り値は job_id (str) の想定
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]] # 非同期履歴追加関数
):
    """MCPサーバーに評価とグリッドサーチのツールを登録"""
    logger.info("Registering evaluation and grid search tools...")
    try:
        output_base_dir = get_output_base_dir(config)
        logger.info(f"Evaluation/Grid Search output base directory: {output_base_dir}")
    except (ConfigError, FileError) as e:
        logger.error(f"Failed to get or validate output base directory: {e}. Evaluation tools will not be registered.", exc_info=True)
        return # ツールを登録せずに終了

    @mcp.tool("run_evaluation")
    async def run_evaluation_tool(
        detector_name: str,
        dataset_name: Optional[str] = None,
        audio_path: Optional[str] = None,
        ref_path: Optional[str] = None,
        output_dir: Optional[str] = None, # オプション引数として追加
        num_procs: Optional[int] = None,
        include_note_metrics: bool = True,
        include_pitch_metrics: bool = True,
        include_frame_metrics: bool = True,
        skip_existing: bool = False,
        session_id: str = ""
    ) -> Dict[str, Any]:
        """指定された検出器の評価を実行します。

        結果は自動生成されたディレクトリ、または指定された `output_dir` (output_base_dir 配下である必要あり) に保存されます。

        Args:
            detector_name (str): 評価する検出器の名前。
            dataset_name (Optional[str], optional): 評価に使用するデータセット名。 Defaults to None.
            audio_path (Optional[str], optional): 単一の音声ファイルパス。 Defaults to None.
            ref_path (Optional[str], optional): 単一の参照ファイルパス。 Defaults to None.
            output_dir (Optional[str], optional): 結果の出力先ディレクトリ。省略時は自動生成。 Defaults to None.
            num_procs (Optional[int], optional): 並列処理数。 Defaults to None.
            include_note_metrics (bool, optional): Note metricsを含めるか。 Defaults to True.
            include_pitch_metrics (bool, optional): Pitch metricsを含めるか。 Defaults to True.
            include_frame_metrics (bool, optional): Frame metricsを含めるか。 Defaults to True.
            skip_existing (bool, optional): 既存の結果をスキップするか。 Defaults to False.
            session_id (str, optional): 関連付けるセッションID。 Defaults to "".

        Returns:
            Dict[str, Any]: 開始されたジョブの情報 (job_id)。
        """
        # 引数チェック (dataset_name or audio/ref_path)
        if not dataset_name and not (audio_path and ref_path):
            return {"error": "Either 'dataset_name' or both 'audio_path' and 'ref_path' must be provided."}
        if dataset_name and (audio_path or ref_path):
            return {"error": "'dataset_name' cannot be used with 'audio_path' or 'ref_path'."}

        # その他の引数準備 (省略) ...

        # 非同期タスクを開始
        # output_base_dir, add_history_sync_func, config は partial で束縛済み？ -> いや、ここで渡す
        # ★ user_output_dir としてツール引数の output_dir を渡す
        job_id = await start_async_job_func(
            _run_evaluate_async, # Target coroutine
            "run_evaluation",    # Tool name for logging/tracking
            session_id=session_id,
            # Args for _run_evaluate_async (besides job_id)
            output_base_dir=output_base_dir,
            add_history_async_func=add_history_async_func,
            config=config,
            user_output_dir=output_dir, # ★ ここで渡す
            # Args for run_evaluation_core (passed via **kwargs in wrapper)
            detector_name=detector_name,
            dataset_name=dataset_name,
            audio_path=audio_path,
            ref_path=ref_path,
            num_procs=num_procs,
            include_note_metrics=include_note_metrics,
            include_pitch_metrics=include_pitch_metrics,
            include_frame_metrics=include_frame_metrics,
            skip_existing=skip_existing,
            # session_id は wrapper にも core にも渡る (必要なら)
            session_id_core=session_id # core 関数用に別名で渡すか、core側で引数名を変える
        )
        return {"status": "pending", "job_id": job_id}

    @mcp.tool("run_grid_search")
    async def run_grid_search_tool(
        config_path: str,
        output_dir: Optional[str] = None, # オプション引数として追加
        num_procs: Optional[int] = None,
        skip_existing: bool = False,
        session_id: str = ""
    ) -> Dict[str, Any]:
        """グリッドサーチを実行して最適なパラメータを見つけます。

        結果は自動生成されたディレクトリ、または指定された `output_dir` (output_base_dir 配下である必要あり) に保存されます。

        Args:
            config_path (str): グリッドサーチ設定ファイル (YAML) のパス。
            output_dir (Optional[str], optional): 結果の出力先ディレクトリ。省略時は自動生成。 Defaults to None.
            num_procs (Optional[int], optional): 並列処理数。 Defaults to None.
            skip_existing (bool, optional): 既存の結果をスキップするか。 Defaults to False.
            session_id (str, optional): 関連付けるセッションID。 Defaults to "".

        Returns:
            Dict[str, Any]: 開始されたジョブの情報 (job_id)。
        """
        # 引数準備 (省略) ...

        # 非同期タスクを開始
        # ★ user_output_dir としてツール引数の output_dir を渡す
        job_id = await start_async_job_func(
            _execute_grid_search_async, # Target coroutine
            "run_grid_search",          # Tool name
            session_id=session_id,
            # Args for _execute_grid_search_async (besides job_id)
            output_base_dir=output_base_dir,
            add_history_async_func=add_history_async_func,
            config=config,
            user_output_dir=output_dir, # ★ ここで渡す
            # Args for run_grid_search_core (passed via **kwargs in wrapper)
            config_path=config_path,
            num_procs=num_procs,
            skip_existing=skip_existing,
            # session_id は wrapper にも core にも渡る (必要なら)
            session_id_core=session_id
        )
        return {"status": "pending", "job_id": job_id}

    logger.info("Evaluation tools registered.")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 