import asyncio  # asyncio をインポート
import json
import logging
import os
import traceback
from functools import partial  # partial をインポート
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP

from src.evaluation.evaluation_runner import (
    run_evaluation as run_evaluation_core,  # 同期関数
)
from src.evaluation.grid_search.core import (
    run_grid_search as run_grid_search_core,  # 同期関数
)
from src.utils import path_utils
from src.utils.exception_utils import (  # FileError, ConfigError をインポート
    ConfigError,
    EvaluationError,
    FileError,
    GridSearchError,
)
from src.utils.json_utils import NumpyEncoder
from src.utils.path_utils import (  # get_project_root を追加
    ensure_dir,
    get_available_datasets,
    get_dataset_paths,
    get_output_base_dir,
    get_output_dir,
    get_project_root,
    get_workspace_dir,
    validate_path_within_allowed_dirs,
)

from . import schemas  # schemas モジュール自体をインポート

# --- 追加: エラー時や開始時のスキーマ --- #
# --- 追加: 履歴イベントスキーマ --- #
from .schemas import JobStatus  # JobStatus をインポートに追加
from .schemas import RunGridSearchInput  # ツール入力スキーマ
from .schemas import (  # 全てのイベントスキーマをHistoryEventBaseDataに統一; 追加; 必要に応じて失敗イベントスキーマもインポート
    EvaluationResultData,
    GridSearchResultData,
    HistoryEventBaseData,
    RunEvaluationInput,
)

logger = logging.getLogger("mcp_server.evaluation_tools")

# --- Synchronous Task Functions (現状維持 or 削除して Async のみに?) --- #
# Note: 履歴追加ロジックは非同期ラッパーに移動するため、ここでの add_history_sync_func は不要になる


def _run_evaluate(
    job_id: str,
    evaluation_results_dir: Path,
    # add_history_sync_func: Callable, # 削除
    **kwargs,
) -> Dict[str, Any]:
    """評価ジョブを実行します (同期、履歴追加なし)"""
    detector_name = kwargs.get("detector_name")
    session_id = kwargs.get("session_id")
    logger.info(
        f"[Job {job_id}] CORE: Running evaluation for '{detector_name}'... Session: {session_id}"
    )

    # 出力ディレクトリの設定 (この関数内でのみ使用)
    output_dir_path: Optional[Path] = None
    if "output_dir" in kwargs and kwargs["output_dir"]:
        output_dir_path = Path(kwargs["output_dir"])
    else:
        # ラッパーで生成されるはずだが、フォールバックとして生成ロジックを残すことも検討
        logger.warning(
            f"[Job {job_id}] CORE: Output directory not provided, using default within evaluation_results_dir."
        )
        eval_results_base_dir = evaluation_results_dir
        unique_suffix = f"session_{session_id or 'nosession'}_job_{job_id}"
        output_dir_path = get_output_dir(eval_results_base_dir, unique_suffix)
        kwargs["output_dir"] = str(output_dir_path)

    if output_dir_path:
        try:
            ensure_dir(output_dir_path)
        except Exception as e:
            logger.error(
                f"[Job {job_id}] CORE: Failed to ensure output directory {output_dir_path}: {e}",
                exc_info=True,
            )
            raise EvaluationError(f"Failed to ensure output directory: {e}") from e
    else:
        logger.error(
            f"[Job {job_id}] CORE: Output directory is invalid or could not be determined."
        )
        raise EvaluationError("Output directory is invalid or could not be determined.")

    # 履歴追加は削除

    try:
        results = run_evaluation_core(**kwargs)
        logger.info(f"[Job {job_id}] CORE: Evaluation finished for '{detector_name}'.")
        return results
    except Exception as e:
        logger.error(
            f"[Job {job_id}] CORE: Error during evaluation for '{detector_name}': {e}",
            exc_info=True,
        )
        raise EvaluationError(f"Evaluation failed for {detector_name}: {e}") from e


def _execute_grid_search(
    job_id: str,
    grid_search_results_dir: Path,
    # add_history_sync_func: Callable, # 削除
    **kwargs,
) -> Dict[str, Any]:
    """グリッドサーチジョブを実行します (同期、履歴追加なし)"""
    # grid_config 辞書が渡される想定に変わったため、config_path は使わない
    # config_path = kwargs.get('config_path')
    session_id = kwargs.get("session_id")
    detector_name = kwargs.get("grid_config", {}).get(
        "detector_name", "unknown"
    )  # grid_configから取得
    logger.info(
        f"[Job {job_id}] CORE: Running grid search for '{detector_name}'... Session: {session_id}"
    )

    # 出力ディレクトリの設定 (この関数内でのみ使用)
    output_dir_path: Optional[Path] = None
    if "output_dir" in kwargs and kwargs["output_dir"]:
        output_dir_path = Path(kwargs["output_dir"])
    else:
        logger.warning(
            f"[Job {job_id}] CORE: Grid search output directory not provided, using default within grid_search_results_dir."
        )
        grid_results_base_dir = grid_search_results_dir
        unique_suffix = f"session_{session_id or 'nosession'}_job_{job_id}"
        output_dir_path = get_output_dir(grid_results_base_dir, unique_suffix)
        kwargs["output_dir"] = str(output_dir_path)

    if output_dir_path:
        try:
            ensure_dir(output_dir_path)
        except Exception as e:
            logger.error(
                f"[Job {job_id}] CORE: Failed to ensure grid search output directory {output_dir_path}: {e}",
                exc_info=True,
            )
            raise GridSearchError(f"Failed to ensure output directory: {e}") from e
    else:
        logger.error(
            f"[Job {job_id}] CORE: Grid search output directory is invalid or could not be determined."
        )
        raise GridSearchError(
            "Grid search output directory is invalid or could not be determined."
        )

    # 履歴追加は削除

    try:
        results = run_grid_search_core(**kwargs)
        logger.info(
            f"[Job {job_id}] CORE: Grid search finished. Output: {kwargs['output_dir']}"
        )
        # run_grid_search_core が output_directory を返すか確認し、なければ追加
        if isinstance(results, dict) and "output_directory" not in results:
            results["output_directory"] = kwargs["output_dir"]
        return results
    except Exception as e:
        logger.error(
            f"[Job {job_id}] CORE: Error during grid search: {e}", exc_info=True
        )
        raise GridSearchError(f"Grid search failed: {e}") from e


# --- Asynchronous Task Wrapper Functions --- #
# ★ 履歴記録でスキーマを使用するように修正


async def _run_evaluate_async(
    job_id: str,
    output_base_dir: Path,
    add_history_async_func: Callable[..., Awaitable[None]],  # 戻り値を None に
    config: Dict[str, Any],
    user_output_dir: Optional[str],
    # --- 履歴記録とコア関数呼び出しに必要な引数 --- #
    detector_name: str,
    dataset_name: Optional[str] = None,
    audio_path: Optional[str] = None,  # 検証済みパス (文字列)
    ref_path: Optional[str] = None,  # 検証済みパス (文字列)
    num_procs: Optional[int] = None,
    code_version: Optional[str] = None,  # ツール引数から直接渡す
    detector_params: Optional[Dict[str, Any]] = None,  # ツール引数から直接渡す
    session_id: Optional[str] = None,
    save_plots: bool = False,
    save_results_json: bool = True,
) -> EvaluationResultData:  # ★ 戻り値の型をスキーマに
    """評価ジョブを非同期で実行します (コアは同期、履歴スキーマ使用)"""
    logger.info(
        f"[Job {job_id}] ASYNC WRAPPER: Running evaluation for '{detector_name}'... Session: {session_id}"
    )

    final_output_dir_str: Optional[str] = None  # 結果スキーマと履歴用
    final_output_dir: Optional[Path] = None  # ensure_dir用
    error_occurred = False
    error_message = "Unknown error"

    # --- 出力ディレクトリ決定と検証 --- #
    try:
        if user_output_dir:
            logger.info(
                f"[Job {job_id}] User specified output directory: {user_output_dir}"
            )
            final_output_dir = validate_path_within_allowed_dirs(
                user_output_dir,
                allowed_base_dirs=[output_base_dir],
                check_existence=False,
                allow_absolute=True,
            )
            logger.info(
                f"[Job {job_id}] Using validated user-specified output directory: {final_output_dir}"
            )
        else:
            logger.info(
                f"[Job {job_id}] No output directory specified by user, generating automatically."
            )
            unique_suffix = f"evaluation_{session_id or 'nosession'}_{job_id}"
            final_output_dir = get_output_dir(
                output_base_dir, unique_suffix=unique_suffix
            )
            logger.info(
                f"[Job {job_id}] Generated output directory: {final_output_dir}"
            )
        # ディレクトリ作成
        ensure_dir(final_output_dir)
        final_output_dir_str = str(final_output_dir)
    except (ValueError, FileError, TypeError, ConfigError) as e:
        logger.error(
            f"[Job {job_id}] Failed to prepare evaluation output directory: {e}",
            exc_info=True,
        )
        error_message = f"Output directory handling failed: {e}"
        error_occurred = True  # エラーフラグを設定
        # エラー時の履歴記録は finally ブロックでまとめて行う
        raise EvaluationError(error_message) from e  # ★ raise は維持

    # --- 入力パス検証 --- #
    verified_audio_path = None
    verified_ref_path = None
    try:
        project_root = get_project_root()
        workspace_dir = get_workspace_dir(config)
        # datasets_base_dir の取得と検証はここで行う
        datasets_base_dir = Path(
            config.get("paths", {}).get("datasets_base", "datasets")
        )
        if not datasets_base_dir.is_absolute():
            datasets_base_dir = project_root / datasets_base_dir
        allowed_dirs = [project_root, workspace_dir, datasets_base_dir]
        if dataset_name:
            try:
                audio_d, label_d, _, _ = get_dataset_paths(config, dataset_name)
                allowed_dirs.extend([audio_d, label_d])
            except (ConfigError, FileError) as e:
                logger.warning(
                    f"[Job {job_id}] Could not get dataset paths ('{dataset_name}') for validation: {e}"
                )
        # audio_path/ref_path が指定されている場合は検証
        if audio_path:
            verified_audio_path = str(
                validate_path_within_allowed_dirs(
                    audio_path, allowed_dirs, check_existence=True, check_is_file=True
                )
            )
        if ref_path:
            verified_ref_path = str(
                validate_path_within_allowed_dirs(
                    ref_path, allowed_dirs, check_existence=True, check_is_file=True
                )
            )

    except (ValueError, FileError, ConfigError) as e:
        logger.error(f"[Job {job_id}] Input path validation failed: {e}", exc_info=True)
        error_message = f"Input path validation failed: {e}"
        error_occurred = True  # エラーフラグを設定
        # エラー時の履歴記録は finally ブロックでまとめて行う
        raise EvaluationError(error_message) from e  # ★ raise は維持

    # --- 履歴記録: 開始 --- #
    if session_id:
        try:
            # ★ スキーマ: HistoryEventBaseData を使用
            start_event_data = schemas.HistoryEventBaseData(
                job_id=job_id,
                detector_name=detector_name,
                dataset_name=dataset_name,
                code_version=code_version,
                parameters_used=detector_params,
            )
            await add_history_async_func(
                session_id, "evaluation_started", start_event_data.model_dump()
            )
        except Exception as hist_e:
            logger.warning(
                f"[Job {job_id}] Failed to add evaluation started history: {hist_e}"
            )

    results_dict: Dict[str, Any] = {}
    try:
        # コア関数呼び出し用の引数を準備
        core_kwargs = {
            "detector_name": detector_name,
            "output_dir": final_output_dir_str,  # 検証済みの出力ディレクトリ
            "dataset_name": dataset_name,
            "audio_paths": [verified_audio_path] if verified_audio_path else None,
            "ref_paths": [verified_ref_path] if verified_ref_path else None,
            "num_procs": num_procs,
            # "code_version": code_version, # run_evaluation_core が解釈する場合
            "detector_params": (
                {detector_name: detector_params}
                if detector_params and detector_name
                else None
            ),
            "save_plots": save_plots,
            "save_results_json": save_results_json,
            "job_id": job_id,
            "session_id": session_id,
        }

        loop = asyncio.get_running_loop()
        sync_func = partial(run_evaluation_core, **core_kwargs)
        results_dict = await loop.run_in_executor(None, sync_func)

        logger.info(f"[Job {job_id}] Evaluation core function completed successfully.")

        # --- 履歴記録: 正常完了 --- #
        if session_id:
            try:
                # ★ スキーマ: HistoryEventBaseData
                complete_event_data = schemas.HistoryEventBaseData(
                    job_id=job_id,
                    evaluation_results=results_dict.get("metrics", {}),  # metricsを渡す
                    summary=results_dict.get("summary"),
                    detector_name=detector_name,
                    code_version_hash=code_version,  # code_version を渡す
                    parameters_used=detector_params,
                    dataset_name=dataset_name,
                    # 他に必要なフィールドがあれば追加
                )
                await add_history_async_func(
                    session_id, "evaluation_complete", complete_event_data.model_dump()
                )
            except Exception as hist_e:
                logger.warning(
                    f"[Job {job_id}] Failed to add evaluation completed history: {hist_e}"
                )

        # --- ジョブ結果の準備 --- #
        # run_evaluation_core の戻り値 (results_dict) を EvaluationResultData スキーマに変換
        job_result = EvaluationResultData(
            summary=results_dict.get("summary"),
            metrics=results_dict.get("metrics"),
            output_dir=final_output_dir_str,
            results_json_path=results_dict.get(
                "results_json_path"
            ),  # コア関数が返す想定
            plot_path=results_dict.get("plot_path"),  # コア関数が返す想定
        )
        return job_result

    except Exception as e_core:
        logger.error(
            f"[Job {job_id}] Error executing evaluation core function: {e_core}",
            exc_info=True,
        )
        error_message = f"Evaluation core execution failed: {e_core}"
        error_occurred = True
        # ★ raise は finally の後に移動させるか、ここでエラー情報を記録して再 raise する
        # ここで raise すると finally での履歴記録が実行される
        raise EvaluationError(error_message) from e_core

    finally:
        # --- 履歴記録: 失敗時 (エラー発生時のみ) --- #
        if error_occurred and session_id:
            try:
                # ★ スキーマ: HistoryEventBaseData を使用
                fail_event_data = schemas.HistoryEventBaseData(
                    job_id=job_id,
                    error=error_message,
                    error_type=(
                        type(e_core).__name__
                        if "e_core" in locals()
                        else "EvaluationError"
                    ),
                    traceback=traceback.format_exc() if "e_core" in locals() else None,
                    detector_name=detector_name,
                    dataset_name=dataset_name,
                    parameters_used=detector_params,
                )
                await add_history_async_func(
                    session_id, "evaluation_failed", fail_event_data.model_dump()
                )
            except Exception as hist_e:
                logger.warning(
                    f"[Job {job_id}] Failed to add evaluation failed history: {hist_e}"
                )


async def _execute_grid_search_async(
    job_id: str,
    output_base_dir: Path,
    add_history_async_func: Callable[..., Awaitable[None]],  # 戻り値を None に
    config: Dict[str, Any],
    user_output_dir: Optional[str],
    # --- run_grid_search_core に渡す引数 --- #
    grid_config: Dict[str, Any],  # YAMLの内容 (辞書)
    num_procs: Optional[int] = None,
    skip_existing: bool = False,
    best_metric: Optional[str] = "note.f_measure",
    code_version: Optional[str] = None,
    session_id: Optional[str] = None,
) -> GridSearchResultData:  # ★ 戻り値の型をスキーマに
    """グリッドサーチジョブを非同期で実行します (コアは同期、履歴スキーマ使用)"""
    detector_name = grid_config.get("detector_name", "unknown")  # configから取得
    logger.info(
        f"[Job {job_id}] ASYNC WRAPPER: Running grid search for '{detector_name}'... Session: {session_id}"
    )

    final_output_dir_str: Optional[str] = None
    final_output_dir: Optional[Path] = None
    error_occurred = False
    error_message = "Unknown error"

    # --- 出力ディレクトリ決定と検証 --- #
    try:
        if user_output_dir:
            logger.info(
                f"[Job {job_id}] User specified grid search output directory: {user_output_dir}"
            )
            final_output_dir = validate_path_within_allowed_dirs(
                user_output_dir,
                allowed_base_dirs=[output_base_dir],
                check_existence=False,
                allow_absolute=True,
            )
            logger.info(
                f"[Job {job_id}] Using validated user-specified grid search output directory: {final_output_dir}"
            )
        else:
            logger.info(
                f"[Job {job_id}] No grid search output directory specified, generating automatically."
            )
            unique_suffix = f"gridsearch_{session_id or 'nosession'}_{job_id}"
            final_output_dir = get_output_dir(
                output_base_dir, unique_suffix=unique_suffix
            )
            logger.info(
                f"[Job {job_id}] Generated grid search output directory: {final_output_dir}"
            )
        # ディレクトリ作成
        ensure_dir(final_output_dir)
        final_output_dir_str = str(final_output_dir)
    except (ValueError, FileError, TypeError, ConfigError) as e:
        logger.error(
            f"[Job {job_id}] Failed to prepare grid search output directory: {e}",
            exc_info=True,
        )
        error_message = f"Grid search output directory handling failed: {e}"
        error_occurred = True
        # エラー時の履歴記録は finally で行う
        raise GridSearchError(error_message) from e

    # --- 履歴記録: 開始 --- #
    if session_id:
        try:
            # ★ スキーマ: HistoryEventBaseData を使用
            start_event_data = schemas.HistoryEventBaseData(
                job_id=job_id,
                detector_name=detector_name,
                code_version=code_version,
                config_used=grid_config,
            )
            await add_history_async_func(
                session_id, "grid_search_started", start_event_data.model_dump()
            )
        except Exception as hist_e:
            logger.warning(
                f"[Job {job_id}] Failed to add grid search started history: {hist_e}"
            )

    results_dict: Dict[str, Any] = {}
    try:
        # コア関数呼び出し用の引数を準備
        core_kwargs = {
            "grid_config": grid_config,
            "output_dir": final_output_dir_str,  # 検証済みパス文字列
            "num_procs": num_procs,
            "skip_existing": skip_existing,
            "best_metric": best_metric,
            # "code_version": code_version, # run_grid_search_core が解釈する場合
            "job_id": job_id,
            "session_id": session_id,
        }

        loop = asyncio.get_running_loop()
        sync_func = partial(run_grid_search_core, **core_kwargs)
        results_dict = await loop.run_in_executor(None, sync_func)

        logger.info(f"[Job {job_id}] Grid search core function completed successfully.")

        # --- 履歴記録: 正常完了 --- #
        if session_id:
            try:
                # ★ スキーマ: HistoryEventBaseData を使用
                complete_event_data = schemas.HistoryEventBaseData(
                    job_id=job_id,
                    summary=results_dict.get("summary"),
                    best_params=results_dict.get("best_params"),
                    best_score=results_dict.get("best_score"),
                    best_params_path=results_dict.get("best_params_path"),
                    results_csv_path=results_dict.get("results_csv_path"),
                    output_dir=final_output_dir_str,
                )
                await add_history_async_func(
                    session_id, "grid_search_complete", complete_event_data.model_dump()
                )
            except Exception as hist_e:
                logger.warning(
                    f"[Job {job_id}] Failed to add grid search completed history: {hist_e}"
                )

        # --- ジョブ結果の準備 --- #
        # run_grid_search_core の戻り値 (results_dict) を GridSearchResultData スキーマに変換
        job_result = GridSearchResultData(
            summary=results_dict.get("summary"),
            best_params=results_dict.get("best_params"),
            best_score=results_dict.get("best_score"),
            best_params_path=results_dict.get("best_params_path"),  # コア関数が返す想定
            results_csv_path=results_dict.get("results_csv_path"),  # コア関数が返す想定
            output_dir=final_output_dir_str,  # 非同期ラッパーで決定したパス
        )
        return job_result

    except Exception as e_core:
        logger.error(
            f"[Job {job_id}] Error executing grid search core function: {e_core}",
            exc_info=True,
        )
        error_message = f"Grid search core execution failed: {e_core}"
        error_occurred = True
        raise GridSearchError(error_message) from e_core

    finally:
        # --- 履歴記録: 失敗時 (エラー発生時のみ) --- #
        if error_occurred and session_id:
            try:
                # ★ スキーマ: HistoryEventBaseData を使用
                fail_event_data = schemas.HistoryEventBaseData(
                    job_id=job_id,
                    error=error_message,
                    error_type=(
                        type(e_core).__name__
                        if "e_core" in locals()
                        else "GridSearchError"
                    ),
                    traceback=traceback.format_exc() if "e_core" in locals() else None,
                    detector_name=detector_name,
                    config_used=grid_config,
                )
                await add_history_async_func(
                    session_id, "grid_search_failed", fail_event_data.model_dump()
                )
            except Exception as hist_e:
                logger.warning(
                    f"[Job {job_id}] Failed to add grid search failed history: {hist_e}"
                )


# --- Tool Registration --- #


def register_evaluation_tools(
    mcp: FastMCP,
    config: Dict[str, Any],
    start_async_job_func: Callable[
        ..., Coroutine[Any, Any, Dict[str, str]]
    ],  # 戻り値修正: JobStartResponse辞書
    add_history_async_func: Callable[..., Awaitable[None]],  # 戻り値修正
):
    """評価とグリッドサーチ関連のツールをMCPツールとして登録します。"""
    logger.info("Registering evaluation and grid search tools...")

    # --- データセット一覧取得ツール --- #
    @mcp.tool("list_datasets")
    async def list_datasets_tool() -> List[Dict[str, Any]]:
        """使用可能なデータセットの一覧とその基本情報を取得します。"""
        try:
            datasets = get_available_datasets(config)
            logger.info(f"データセット一覧を取得しました: {len(datasets)}件")
            return datasets
        except Exception as e:
            logger.error(
                f"データセット一覧取得中にエラーが発生しました: {e}", exc_info=True
            )
            return [{"error": f"データセット一覧取得中にエラー: {str(e)}"}]

    # --- Helper to start evaluation/grid search jobs --- #
    async def _start_evaluation_job(
        task_coroutine_factory: Callable[..., Coroutine],
        tool_name: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:  # 戻り値は JobStartResponse 形式 (Dict)
        """評価/グリッドサーチタスクを開始し、JobStartResponse 辞書を返すヘルパー"""
        job_description = f"{tool_name} job" + (
            f" for session {session_id}" if session_id else ""
        )
        logger.info(f"Requesting to start {job_description} with args: {kwargs}")
        try:
            # ファクトリに関数を渡してコルーチンオブジェクトを作成
            # job_idにはプレースホルダーを渡す（実際のIDはstart_async_job_funcで生成される）
            task_coroutine = task_coroutine_factory(job_id="placeholder", **kwargs)
            # start_async_job_func を呼び出してジョブを開始し、結果 (JobStartResponse辞書) を取得
            # tool_name と session_id も渡すように変更
            job_start_response = await start_async_job_func(
                task_coroutine, tool_name=tool_name, session_id=session_id
            )
            logger.info(
                f"Successfully requested {job_description}. Job info: {job_start_response}"
            )
            return job_start_response  # JobStartResponse 形式の辞書を返す
        except Exception as e:
            logger.error(f"Failed to start {job_description}: {e}", exc_info=True)
            # エラー時も JobStartResponse に準じた形式を返す方が良いかもしれない
            return {
                "error": f"Failed to start {tool_name} job: {e}",
                "status": JobStatus.FAILED.value,
            }

    # --- run_evaluation ツール --- #
    run_evaluation_factory = partial(
        _run_evaluate_async,
        output_base_dir=get_output_base_dir(config),
        add_history_async_func=add_history_async_func,
        config=config,
    )

    # ★ 入力スキーマを適用
    @mcp.tool("run_evaluation")
    async def run_evaluation_tool(
        # スキーマに合わせてシグネチャを完全に書き換え
        detector_name: str,
        dataset_name: Optional[str] = None,
        code_version: Optional[str] = None,
        detector_params: Optional[Dict[str, Any]] = None,
        save_plots: bool = False,
        save_results_json: bool = True,
        session_id: Optional[str] = None,
        output_dir: Optional[str] = None,  # ユーザー指定用として残す (スキーマにはない)
    ) -> Dict[str, Any]:  # JobStartResponse 形式
        """指定された検出器とデータセットで評価ジョブを開始します。"""
        # dataset_nameが指定されていない場合はconfig['evaluation']['default_dataset']を使用
        if dataset_name is None:
            dataset_name = config.get("evaluation", {}).get(
                "default_dataset", "synthesized_v1"
            )
            logger.info(
                f"dataset_nameが指定されていないため、デフォルト値 '{dataset_name}' を使用します"
            )

        # スキーマで検証された引数 + output_dir を kwargs にまとめる
        kwargs = {
            "detector_name": detector_name,
            "dataset_name": dataset_name,
            # audio_path, ref_path は dataset_name から導出される想定
            "num_procs": config.get("resource_limits", {}).get(
                "evaluation_num_procs"
            ),  # configから取得
            "session_id": session_id,
            "code_version": code_version,
            "detector_params": detector_params,
            "save_plots": save_plots,
            "save_results_json": save_results_json,
            "user_output_dir": output_dir,  # スキーマ外の引数を kwargs に含める
        }
        return await _start_evaluation_job(
            run_evaluation_factory,
            "run_evaluation",
            **kwargs,  # スキーマ検証済み引数 + user_output_dir を渡す
        )

    # --- run_grid_search ツール --- #
    run_grid_search_factory = partial(
        _execute_grid_search_async,
        output_base_dir=get_output_base_dir(config),
        add_history_async_func=add_history_async_func,
        config=config,
    )

    # ★ 入力スキーマを適用
    @mcp.tool("run_grid_search")
    async def run_grid_search_tool(
        # スキーマに合わせてシグネチャを完全に書き換え
        grid_config: Dict[str, Any],
        skip_existing: bool = False,
        best_metric: Optional[str] = "note.f_measure",
        code_version: Optional[str] = None,
        session_id: Optional[str] = None,
        output_dir: Optional[str] = None,  # ユーザー指定用 (スキーマにはない)
    ) -> Dict[str, Any]:  # JobStartResponse 形式
        """グリッドサーチジョブを開始します。"""
        kwargs = {
            "grid_config": grid_config,
            "num_procs": config.get("resource_limits", {}).get(
                "grid_search_num_procs"
            ),  # configから取得
            "skip_existing": skip_existing,
            "best_metric": best_metric,
            "session_id": session_id,
            "code_version": code_version,
            "user_output_dir": output_dir,  # スキーマ外の引数を kwargs に含める
        }
        return await _start_evaluation_job(
            run_grid_search_factory, "run_grid_search", **kwargs
        )

    logger.info("Evaluation and grid search tools registered.")


# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.")
