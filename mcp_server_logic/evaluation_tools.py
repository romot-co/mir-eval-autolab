import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine, List

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP
from src.evaluation.evaluation_runner import run_evaluation as run_evaluation_core # 同期関数
from src.evaluation.grid_search.core import run_grid_search as run_grid_search_core # 同期関数
from src.utils.path_utils import get_output_dir, ensure_dir
from src.utils.json_utils import NumpyEncoder
from src.utils.exception_utils import EvaluationError, GridSearchError

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

# --- Tool Registration --- #

def register_evaluation_tools(
    mcp: FastMCP,
    config: Dict[str, Any],
    start_async_job_func: Callable[..., str],
    add_history_sync_func: Callable # Needed by task funcs
):
    """評価およびグリッドサーチ関連のMCPツールを登録"""
    logger.info("Registering evaluation and grid search tools...")

    # Helper to curry args for async job start
    def _start_eval_job(task_func: Callable, tool_name: str, session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Pass config and sync history adder to the task function
        kwargs['session_id'] = session_id # Ensure session_id is in kwargs for the task
        
        # Extract necessary config values for the specific task
        task_specific_config_args = {}
        if task_func == _run_evaluate:
            task_specific_config_args['evaluation_results_dir'] = Path(config['paths']['evaluation_results'])
            # Add other config values needed by _run_evaluate here
        elif task_func == _execute_grid_search:
             task_specific_config_args['grid_search_results_dir'] = Path(config['paths']['grid_search_results'])
             # Add other config values needed by _execute_grid_search here
        
        # Pass only necessary args, not the whole config dict
        # Note: add_history_sync_func is already passed via start_async_job_func in register_tools
        job_id = start_async_job_func(
            task_func, 
            tool_name, 
            session_id, 
            # config, # Pass config to start_async_job_func if needed by core job logic (logging etc.)
            # add_history_sync_func, # Passed implicitly by register_tools
            **task_specific_config_args, # Pass specific config values
            **kwargs # Pass original tool arguments
        )
        # Add request history immediately
        if session_id:
            try:
                 add_history_sync_func(session_id, f"{tool_name}_request", {"params": kwargs, "job_id": job_id})
            except Exception as e:
                 logger.warning(f"Failed to add request history for {tool_name} (Session: {session_id}): {e}")
        return {"job_id": job_id, "status": "pending"}

    # Register tools
    # run_evaluation の引数をラップ関数で受け取り、_start_eval_job に渡す
    @mcp.tool("run_evaluation")
    async def run_evaluation_tool(
        detector_name: str,
        dataset_name: Optional[str] = None,
        audio_path: Optional[str] = None,
        ref_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        num_procs: Optional[int] = None,
        include_note_metrics: bool = True,
        include_pitch_metrics: bool = True,
        include_frame_metrics: bool = True,
        skip_existing: bool = False,
        session_id: str = ""
    ) -> Dict[str, Any]:
        # Validate args
        if not dataset_name and not (audio_path and ref_path):
            raise ValueError("`dataset_name` or (`audio_path` and `ref_path`) must be specified.")
        # Collect args for the task function
        task_kwargs = locals()
        task_kwargs.pop('session_id') # Remove session_id as it's handled by _start_eval_job
        task_kwargs.pop('self', None) # Remove self if present (might happen in classes)
        return _start_eval_job(_run_evaluate, "run_evaluation", session_id, **task_kwargs)

    # run_grid_search の引数をラップ関数で受け取り、_start_eval_job に渡す
    @mcp.tool("run_grid_search")
    async def run_grid_search_tool(
        config_path: str,
        output_dir: Optional[str] = None,
        num_procs: Optional[int] = None,
        skip_existing: bool = False,
        session_id: str = ""
    ) -> Dict[str, Any]:
        task_kwargs = locals()
        task_kwargs.pop('session_id')
        task_kwargs.pop('self', None)
        return _start_eval_job(_execute_grid_search, "run_grid_search", session_id, **task_kwargs)

    logger.info("Evaluation and grid search tools registered.")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 