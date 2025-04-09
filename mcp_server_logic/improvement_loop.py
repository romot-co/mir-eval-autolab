import logging
import json
import time
from typing import Dict, Any, Optional, Callable, Coroutine
from pathlib import Path

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP
# 各ツールモジュールの同期タスク関数をインポート
from .llm_tools import _run_suggest_parameters, _run_generate_hypotheses, _run_improve_code # LLM関連
from .evaluation_tools import _execute_grid_search # グリッドサーチ実行
from .code_tools import _run_get_code, _run_save_code, get_detector_path # コード取得/保存 ヘルパー関数をインポート
# 科学自動化戦略クラス
from src.science_automation.exploration_strategy import ExplorationStrategy
from src.utils.exception_utils import MirexError, LLMError, ConfigError
# DBアクセス関数
from . import db_utils

logger = logging.getLogger('mcp_server.improvement_loop')

# --- Synchronous Task Functions for Loop Control --- #

def _run_parameter_optimization(job_id: str, config: Dict[str, Any], add_history_sync_func: Callable, detector_name: str, source_code: str, audio_dir: Optional[str] = None, reference_dir: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
    """パラメータ最適化プロセスを実行 (同期、LLM提案 -> グリッドサーチ)"""
    logger.info(f"[Job {job_id}] Starting parameter optimization for '{detector_name}'... Session: {session_id}")

    if session_id:
        try:
            add_history_sync_func(session_id, "parameter_optimization_started", {"detector_name": detector_name})
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add optimization started history: {hist_e}")

    try:
        # 1. LLM にパラメータ範囲を提案させる (_run_suggest_parameters を直接呼び出す)
        logger.debug(f"[Job {job_id}] Suggesting parameters via LLM...")
        suggestion_result = _run_suggest_parameters(job_id, config, add_history_sync_func, source_code, context="Suggest parameters for grid search", session_id=session_id)

        suggested_params = suggestion_result.get('params', [])
        if not suggested_params or not isinstance(suggested_params, list):
            warning_msg = "LLM did not suggest any valid parameters for optimization."
            logger.warning(f"[Job {job_id}] {warning_msg}")
            if session_id:
                try: add_history_sync_func(session_id, "parameter_optimization_skipped", {"reason": warning_msg})
                except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add skipped history: {hist_e}")
            return {"status": "skipped", "message": warning_msg, "suggestion_result": suggestion_result}

        # 2. グリッドサーチ設定を構築
        grid_params_config = {}
        for param_info in suggested_params:
            name = param_info.get('name')
            range_list = param_info.get('suggested_range')
            step = param_info.get('step')
            # 簡単なバリデーションと範囲生成 (より堅牢な処理が必要)
            if name and range_list and len(range_list) == 2:
                try:
                    start, end = float(range_list[0]), float(range_list[1])
                    num_steps = 5 # 固定ステップ数 or LLM提案?
                    if step:
                        num_steps = max(2, int((end - start) / float(step)) + 1)
                    # 線形 or 対数スケール？ -> ここでは線形と仮定
                    values = [start + i * (end - start) / (num_steps - 1) for i in range(num_steps)]
                    # 型を元の値に合わせる試み (int or float)
                    current_val = param_info.get('current_value')
                    if isinstance(current_val, int):
                         values = [int(round(v)) for v in values]
                    elif isinstance(current_val, float):
                         values = [round(v, 4) for v in values] # 小数点4桁程度

                    grid_params_config[name] = sorted(list(set(values))) # 重複除去してソート
                except (ValueError, TypeError) as e:
                    logger.warning(f"[Job {job_id}] Could not parse range/step for param '{name}': {e}")
            else:
                logger.warning(f"[Job {job_id}] Skipping param '{name}' due to missing range/name.")

        if not grid_params_config:
             warning_msg = "Could not construct grid search config from LLM suggestions."
             logger.warning(f"[Job {job_id}] {warning_msg}")
             if session_id:
                 try: add_history_sync_func(session_id, "parameter_optimization_skipped", {"reason": warning_msg})
                 except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add skipped history: {hist_e}")
             return {"status": "skipped", "message": warning_msg}

        # 3. グリッドサーチを実行 (_execute_grid_search を直接呼び出す)
        logger.debug(f"[Job {job_id}] Executing grid search with {len(grid_params_config)} parameter(s)...")
        # グリッドサーチ用の設定ファイルを作成する必要がある
        # または _execute_grid_search が辞書を受け取れるように修正？ -> ここでは一時ファイルを作成
        import tempfile, yaml
        temp_config_path = None
        try:
             grid_config_data = {
                 "detector_name": detector_name,
                 "audio_dir": audio_dir or config['paths']['audio'],
                 "reference_dir": reference_dir or config['paths']['reference'],
                 "reference_pattern": "*.csv", # 設定可能にすべき
                 "param_grid": grid_params_config,
                 "best_metric": "note.f_measure" # 設定可能にすべき
             }
             with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False, encoding='utf-8') as tmp_f:
                  yaml.dump(grid_config_data, tmp_f, default_flow_style=False)
                  temp_config_path = tmp_f.name
             logger.info(f"[Job {job_id}] Created temporary grid search config: {temp_config_path}")

             grid_search_kwargs = {
                 "config_path": temp_config_path,
                 "session_id": session_id,
                 # num_procs など、他の引数も config から取得・設定
                 "num_procs": config.get('resource_limits', {}).get('max_concurrent_jobs', 1)
             }
             grid_results = _execute_grid_search(job_id, config, add_history_sync_func, **grid_search_kwargs)
             logger.info(f"[Job {job_id}] Grid search completed.")

             final_result = {
                 "status": "completed",
                 "suggestion_result": suggestion_result,
                 "grid_search_config": grid_params_config,
                 "grid_search_results": grid_results
             }
             if session_id:
                  try:
                      add_history_sync_func(session_id, "parameter_optimization_complete", final_result)
                  except Exception as hist_e:
                      logger.warning(f"[Job {job_id}] Failed to add optimization complete history: {hist_e}")
             return final_result

        finally:
             # 一時設定ファイルを削除
             if temp_config_path and os.path.exists(temp_config_path):
                  try:
                       os.unlink(temp_config_path)
                       logger.debug(f"[Job {job_id}] Deleted temporary grid config: {temp_config_path}")
                  except Exception as e:
                       logger.warning(f"[Job {job_id}] Failed to delete temporary grid config {temp_config_path}: {e}")

    except Exception as e:
        logger.error(f"[Job {job_id}] Error during parameter optimization: {e}", exc_info=True)
        if session_id:
            try:
                add_history_sync_func(session_id, "parameter_optimization_failed", {"error": str(e)})
            except Exception as hist_e:
                logger.warning(f"[Job {job_id}] Failed to add optimization failed history: {hist_e}")
        raise MirexError(f"Parameter optimization failed: {e}") from e

def _run_suggest_exploration_strategy(job_id: str, config: Dict[str, Any], add_history_sync_func: Callable, session_id: str, current_performance: Optional[float] = None) -> Dict[str, Any]:
    """次の探索戦略を提案する (同期)"""
    logger.info(f"[Job {job_id}] Suggesting next exploration strategy... Session: {session_id}")
    try:
        # DBからセッション履歴を取得
        db_path = Path(config['paths']['db']) / db_utils.DB_FILENAME
        row = db_utils._db_fetch_one(db_path, "SELECT history FROM sessions WHERE session_id = ?", (session_id,))
        if not row:
             raise ValueError(f"Session {session_id} not found for strategy suggestion.")
        try:
            history = json.loads(row["history"] or '[]')
        except json.JSONDecodeError:
             logger.error(f"[Job {job_id}] Failed to decode session history for {session_id}")
             history = []

        # ExplorationStrategy インスタンス化
        try:
             strategy_instance = ExplorationStrategy(history=history)
             # 必要に応じて現在の状態を設定 (例: current_performance)
             if current_performance is not None:
                  # strategy_instance にパフォーマンスを設定するメソッドがあれば呼び出す
                  pass
        except Exception as e:
             logger.error(f"[Job {job_id}] Failed to initialize ExplorationStrategy: {e}", exc_info=True)
             raise ConfigError("Failed to initialize ExplorationStrategy") from e

        # 停滞検出と戦略提案
        is_stagnating = strategy_instance.detect_stagnation()
        if is_stagnating:
            next_action = strategy_instance.suggest_strategy_change()
            reason = "Stagnation detected"
        else:
            # 停滞していない場合は通常の次のアクションを取得
            next_action = strategy_instance.get_next_action(current_performance)
            reason = "Normal progression"

        result = {
            "strategy_suggestion": next_action,
            "reason": reason,
            "is_stagnating": is_stagnating
        }

        try:
            add_history_sync_func(session_id, "strategy_suggested", result)
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add strategy suggested history: {hist_e}")

        logger.info(f"[Job {job_id}] Strategy suggestion: {next_action.get('action', 'N/A')}, Reason: {reason}")
        return result

    except Exception as e:
        logger.error(f"[Job {job_id}] Error suggesting exploration strategy: {e}", exc_info=True)
        if session_id:
             try:
                  add_history_sync_func(session_id, "strategy_suggestion_failed", {"error": str(e)})
             except Exception as hist_e:
                  logger.warning(f"[Job {job_id}] Failed to add strategy error history: {hist_e}")
        raise MirexError(f"Failed to suggest strategy: {e}") from e

def register_loop_tools(
    mcp: FastMCP,
    config: Dict[str, Any],
    start_async_job_func: Callable[..., str],
    add_history_sync_func: Callable
):
    """自動改善ループと関連戦略ツールのMCPツールを登録"""
    logger.info("Registering improvement loop and strategy tools...")

    # Helper for starting async jobs, similar to other modules
    def _start_loop_job(task_func: Callable, tool_name: str, session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        kwargs['session_id'] = session_id # Ensure session_id is passed to task
        job_id = start_async_job_func(task_func, tool_name, session_id, config, add_history_sync_func, **kwargs)
        if session_id:
            try: add_history_sync_func(session_id, f"{tool_name}_request", {"params": kwargs, "job_id": job_id})
            except Exception as e: logger.warning(f"Failed to add request history for {tool_name}: {e}")
        return {"job_id": job_id, "status": "pending"}

    # optimize_parameters ツール
    @mcp.tool("optimize_parameters")
    async def optimize_parameters_tool(
        detector_name: str,
        code_version: Optional[str] = None, # 最適化対象のコードバージョン
        auto_suggest: bool = True, # LLMによるパラメータ提案を行うか
        audio_dir: Optional[str] = None, # データセット指定用
        reference_dir: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """指定された検出器のパラメータ最適化 (LLM提案+グリッドサーチ) を実行します。"""
        if not auto_suggest:
            # If auto_suggest is False, we expect the caller to manually run grid search later
            # This tool focuses on the LLM suggestion + grid search combo
            # Maybe return an error or just don't do anything? Let's return an error for now.
            raise ValueError("optimize_parameters tool currently only supports auto_suggest=True (LLM parameter suggestion).")

        # 最適化対象のコードを取得
        try:
            # Use _run_get_code task function to get code content (consistency with save_code)
            # This runs it as a separate (short) job, maybe less efficient but uses the same infra
            # Alternatively, call get_detector_path and read directly like in the loop task?
            # Let's try direct read for efficiency here.
            logger.debug(f"Getting code for '{detector_name}' version '{code_version or 'latest'}' for optimization...")
            code_path = get_detector_path(config, detector_name, version=code_version, session_id=session_id)
            with open(code_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            logger.debug(f"Got code from {code_path}")
        except (FileNotFoundError, ConfigError, FileError) as e:
            logger.error(f"Failed to get code '{detector_name}' (Version: {code_version or 'latest'}) for optimization: {e}")
            # Return a job-like failure structure
            return {"job_id": None, "status": "failed", "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error getting code: {e}", exc_info=True)
            return {"job_id": None, "status": "failed", "error": f"Unexpected error getting code: {e}"}

        task_kwargs = {
            "detector_name": detector_name,
            "code_version": code_version, # _run_parameter_optimization 内でコード取得時に使用
            "source_code": source_code, # Pass the obtained source code directly to the task function
            "audio_dir": audio_dir,
            "reference_dir": reference_dir,
            "session_id": session_id
        }
        # _run_parameter_optimization は source_code を要求するため、ここで None を渡すか、
        # タスク関数側で get_code を呼び出すように修正が必要。
        # -> タスク関数側で get_code を呼び出すように修正する。
        return _start_loop_job(_run_parameter_optimization, "optimize_parameters", session_id, **task_kwargs)

    # suggest_exploration_strategy ツール
    @mcp.tool("suggest_exploration_strategy")
    async def suggest_exploration_strategy_tool(
        session_id: str,
        current_performance: Optional[float] = None # 現在のパフォーマンス指標
    ) -> Dict[str, Any]:
        """現在のセッション履歴とパフォーマンスに基づいて、次の探索戦略を提案します。"""
        if not session_id:
             raise ValueError("session_id is required.")
        task_kwargs = {
            "session_id": session_id,
            "current_performance": current_performance
        }
        return _start_loop_job(_run_suggest_exploration_strategy, "suggest_exploration_strategy", session_id, **task_kwargs)

    logger.info("Improvement loop and strategy tools registered.")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 