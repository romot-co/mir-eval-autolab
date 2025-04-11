import logging
import json
import time
from typing import Dict, Any, Optional, Callable, Coroutine, Awaitable
from pathlib import Path
import asyncio
import os # 一時ファイル削除のため
from functools import partial

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP
# 各ツールモジュールの同期タスク関数をインポート (一部は直接実行しない)
# from .llm_tools import _run_suggest_parameters, _run_generate_hypotheses, _run_improve_code # LLM関連
# from .evaluation_tools import _execute_grid_search # グリッドサーチ実行
# LLMと評価の非同期ラッパーをインポート (あるいはコア関数を run_in_executor で呼び出す)
from .llm_tools import _run_suggest_parameters_async # 修正: llm_tools の非同期関数を使う
from .evaluation_tools import _execute_grid_search_async
from .code_tools import _run_get_code, _run_save_code, get_detector_path # コード取得/保存 ヘルパー関数をインポート
# 科学自動化戦略クラス
# from src.science_automation.exploration_strategy import ExplorationStrategy # 元の絶対インポートに戻す <- コメントアウト
# from ..science_automation.exploration_strategy import ExplorationStrategy # 相対インポートをコメントアウト
from src.utils.exception_utils import MirexError, LLMError, ConfigError, StateManagementError # StateManagementError追加
from src.utils.path_utils import is_safe_path_component # インポート追加
# DBアクセス関数
from . import db_utils
# --- 追加: 履歴イベントスキーマ --- #
from .schemas import (
    ParameterOptimizationStartedData,
    ParameterOptimizationSkippedData,
    ParameterOptimizationCompleteData,
    ParameterOptimizationFailedData,
    StrategySuggestedData,
    SuggestExplorationStrategyInput,
    OptimizeParametersInput
)

logger = logging.getLogger('mcp_server.improvement_loop')

# --- 修正: 非同期 Task Functions --- #

async def _run_parameter_optimization(
    job_id: str,
    config: Dict[str, Any],
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    detector_name: str,
    source_code: str,
    audio_dir: Optional[str] = None,
    reference_dir: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """パラメータ最適化プロセスを実行 (非同期、LLM提案 -> グリッドサーチ)"""
    logger.info(f"[Job {job_id}] Starting parameter optimization for '{detector_name}'... Session: {session_id}")
    db_path = Path(config['paths']['db_path']) # db_path を取得

    if session_id:
        try:
            # --- 修正: スキーマを使用 --- #
            event_data = ParameterOptimizationStartedData(
                job_id=job_id,
                detector_name=detector_name
            )
            await add_history_async_func(session_id, "parameter_optimization_started", event_data.model_dump(exclude_none=True))
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add optimization started history: {hist_e}")

    suggestion_result: Optional[Dict[str, Any]] = None # スコープ外でも参照可能にする
    grid_params_config: Optional[Dict[str, Any]] = None

    try:
        # 1. LLM にパラメータ範囲を提案させる
        logger.debug(f"[Job {job_id}] Suggesting parameters via LLM...")
        # llm_tools の非同期関数を直接呼び出す
        suggestion_result = await _run_suggest_parameters_async(
            job_id=job_id,
            add_history_async_func=add_history_async_func,
            llm_config=config.get('llm', {}),
            resource_limits=config.get('resource_limits', {}),
            db_path=db_path,
            session_id=session_id,
            source_code=source_code,
            context="Suggest parameters for grid search"
        )

        suggested_params = suggestion_result.get('params', [])
        if not suggested_params or not isinstance(suggested_params, list):
            warning_msg = "LLM did not suggest any valid parameters for optimization."
            logger.warning(f"[Job {job_id}] {warning_msg}")
            if session_id:
                try:
                    # --- 修正: スキーマを使用 --- #
                    event_data = ParameterOptimizationSkippedData(
                        job_id=job_id,
                        reason=warning_msg,
                        suggestion_result=suggestion_result
                    )
                    await add_history_async_func(session_id, "parameter_optimization_skipped", event_data.model_dump(exclude_none=True))
                except Exception as hist_e:
                    logger.warning(f"[Job {job_id}] Failed to add skipped history: {hist_e}")
            return {"status": "skipped", "message": warning_msg, "suggestion_result": suggestion_result}

        # 2. グリッドサーチ設定を構築
        grid_params_config = {}
        for param_info in suggested_params:
            name = param_info.get('name')
            range_list = param_info.get('suggested_range')
            step = param_info.get('step')
            param_type = param_info.get('type', 'float') # 型情報も考慮 (int, float, categorical?)
            values = param_info.get('suggested_values') # 直接値リストが提案される場合も考慮

            if name and values:
                # suggested_values があればそれを使う
                grid_params_config[name] = values
                logger.debug(f"Using suggested values for '{name}': {values}")
            elif name and range_list and len(range_list) == 2:
                # suggested_range から値を生成
                try:
                    start, end = float(range_list[0]), float(range_list[1])
                    num_steps = 5 # デフォルトステップ数
                    if step:
                        try:
                            num_steps = max(2, int((end - start) / float(step)) + 1)
                        except ValueError:
                            logger.warning(f"Invalid step value for param '{name}': {step}. Using default {num_steps} steps.")
                    else:
                        # ステップが提案されない場合のデフォルト処理
                        num_steps = 5 if param_type == 'float' else min(5, int(end - start) + 1) if param_type == 'int' else 5

                    # 線形スケールで値を生成
                    if num_steps > 1:
                         gen_values = [start + i * (end - start) / (num_steps - 1) for i in range(num_steps)]
                    else:
                         gen_values = [start] # ステップ数が1の場合

                    # 型変換
                    if param_type == 'int':
                        gen_values = [int(round(v)) for v in gen_values]
                    else: # float (default)
                        gen_values = [round(v, 4) for v in gen_values] # 小数点4桁程度

                    grid_params_config[name] = sorted(list(set(gen_values))) # 重複除去してソート
                except (ValueError, TypeError) as e:
                    logger.warning(f"[Job {job_id}] Could not parse range/step for param '{name}': {e}")
            else:
                logger.warning(f"[Job {job_id}] Skipping param '{name}' due to missing range/name or values.")

        if not grid_params_config:
             warning_msg = "Could not construct grid search config from LLM suggestions."
             logger.warning(f"[Job {job_id}] {warning_msg}")
             if session_id:
                 try:
                     # --- 修正: スキーマを使用 --- #
                     event_data = ParameterOptimizationSkippedData(
                         job_id=job_id,
                         reason=warning_msg,
                         suggestion_result=suggestion_result
                     )
                     await add_history_async_func(session_id, "parameter_optimization_skipped", event_data.model_dump(exclude_none=True))
                 except Exception as hist_e:
                     logger.warning(f"[Job {job_id}] Failed to add skipped history: {hist_e}")
             return {"status": "skipped", "message": warning_msg}

        # 3. グリッドサーチを実行
        logger.debug(f"[Job {job_id}] Executing grid search with {len(grid_params_config)} parameter(s)...")
        import tempfile, yaml
        temp_config_path = None
        try:
             grid_config_data = {
                 "detector_name": detector_name,
                 "audio_dir": audio_dir or config['paths']['audio'],
                 "reference_dir": reference_dir or config['paths']['reference'],
                 "reference_pattern": config.get('evaluation', {}).get('reference_pattern', '*.csv'),
                 "param_grid": grid_params_config,
                 "best_metric": config.get('evaluation', {}).get('grid_search_metric', 'note.f_measure')
             }
             # 一時ファイルに書き出す
             with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False, encoding='utf-8') as tmp_f:
                  yaml.dump(grid_config_data, tmp_f, default_flow_style=False)
                  temp_config_path = tmp_f.name
             logger.info(f"[Job {job_id}] Created temporary grid search config: {temp_config_path}")

             # グリッドサーチ実行関数を呼び出す
             grid_search_kwargs = {
                 "config_path": temp_config_path,
                 "session_id": session_id,
                 "num_procs": config.get('resource_limits', {}).get('max_concurrent_jobs', 1)
             }
             grid_results = await _execute_grid_search_async(
                 job_id=job_id,
                 output_base_dir=Path(config['paths']['output_base']),
                 add_history_async_func=add_history_async_func,
                 global_config=config,
                 llm_client=None, # グリッドサーチ自体はLLMクライアント不要
                 **grid_search_kwargs
             )

             final_result = {
                 "status": "completed",
                 "suggestion_result": suggestion_result, # LLMの提案内容も返す
                 "grid_search_config": grid_params_config, # 実行したグリッド設定
                 "grid_search_results": grid_results # グリッドサーチの結果
             }
             if session_id:
                  try:
                      # --- 修正: スキーマを使用 --- #
                      event_data = ParameterOptimizationCompleteData(
                          job_id=job_id,
                          best_params=grid_results.get('best_params'),
                          best_score=grid_results.get('best_score'),
                          search_space=grid_params_config,
                          grid_search_results=grid_results
                      )
                      await add_history_async_func(session_id, "parameter_optimization_complete", event_data.model_dump(exclude_none=True))
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

    except MirexError as me: # MirexError はそのままリスロー
        logger.error(f"[Job {job_id}] MirexError during parameter optimization: {me}")
        if session_id:
            try:
                # --- 修正: スキーマを使用 --- #
                event_data = ParameterOptimizationFailedData(
                    job_id=job_id,
                    error=str(me),
                    type='MirexError'
                )
                await add_history_async_func(session_id, "parameter_optimization_failed", event_data.model_dump(exclude_none=True))
            except Exception as hist_e:
                logger.warning(f"[Job {job_id}] Failed to add failed history: {hist_e}")
        raise me
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"[Job {job_id}] Unexpected {error_type} during parameter optimization: {e}", exc_info=True)
        if session_id:
            try:
                # --- 修正: スキーマを使用 --- #
                event_data = ParameterOptimizationFailedData(
                    job_id=job_id,
                    error=str(e),
                    type=error_type
                )
                await add_history_async_func(session_id, "parameter_optimization_failed", event_data.model_dump(exclude_none=True))
            except Exception as hist_e:
                logger.warning(f"[Job {job_id}] Failed to add failed history: {hist_e}")
        # StateManagementError などのカスタム例外にするか検討
        raise MirexError(f"Parameter optimization failed unexpectedly: {e}") from e

# --- コメントアウト --- #
# async def _run_suggest_exploration_strategy(
#     job_id: str,
#     config: Dict[str, Any],
#     add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
#     session_id: str
#     # current_performance: Optional[float] = None # 削除: 履歴から取得するため不要
# ) -> Dict[str, Any]:
#     """次の探索戦略を提案する (非同期)"""
#     logger.info(f"[Job {job_id}] Suggesting next exploration strategy... Session: {session_id}")
#     db_path = Path(config['paths']['db_path'])
#     strategy_config = config.get('strategy', {}) # ExplorationStrategy 用の設定を取得
# 
#     try:
#         # DBからセッション履歴を取得
#         row = await db_utils.db_fetch_one_async(db_path, "SELECT history FROM sessions WHERE session_id = ?", (session_id,))
#         if not row:
#              # セッションが見つからない場合は StateManagementError を発生させる
#              raise StateManagementError(f"Session {session_id} not found for strategy suggestion.")
#         try:
#             history = json.loads(row["history"] or '[]')
#             if not isinstance(history, list): history = []
#         except json.JSONDecodeError:
#              logger.error(f"[Job {job_id}] Failed to decode session history for {session_id}")
#              history = []
# 
#         # ExplorationStrategy インスタンス化 (config も渡す)
#         try:
#              # --- 修正: config を渡す --- #
#              strategy_instance = ExplorationStrategy(history=history, config=strategy_config)
#         except Exception as e:
#              logger.error(f"[Job {job_id}] Failed to initialize ExplorationStrategy: {e}", exc_info=True)
#              # 初期化失敗は設定エラーに近いので ConfigError とする
#              raise ConfigError("Failed to initialize ExplorationStrategy") from e
# 
#         # 次のアクションを取得 (最新の評価結果は引数で渡さず、strategy_instance が内部で分析する)
#         next_action = strategy_instance.get_next_action()
# 
#         # 提案されたアクションを実行アクションとして履歴に記録する（この関数内で行うべきか？）
#         # -> 呼び出し元の auto_improver.py などで行う方が責務分離的に良いかもしれない。
#         #    ここでは提案を返すことに集中する。
#         result = {
#             "strategy_suggestion": next_action,
#             # "reason": reason, # get_next_action が説明を含むため不要
#             # "is_stagnating": is_stagnating # get_next_action が停滞判断するため不要
#         }
# 
#         try:
#             # --- 修正: スキーマを使用 --- #
#             event_data = StrategySuggestedData(
#                 job_id=job_id,
#                 strategy_suggestion=next_action
#             )
#             await add_history_async_func(session_id, "strategy_suggested", event_data.model_dump(exclude_none=True))
#         except Exception as hist_e:
#             logger.warning(f"[Job {job_id}] Failed to add strategy suggested history: {hist_e}")
# 
#         logger.info(f"[Job {job_id}] Strategy suggestion: Action={next_action.get('action', 'N/A')}, Phase={next_action.get('phase', 'N/A')}")
#         return result
# 
#     except StateManagementError as sme:
#         logger.error(f"[Job {job_id}] Session/DB error suggesting exploration strategy: {sme}")
#         # Session not found などのDBエラーはそのままリスロー
#         raise sme
#     except ConfigError as ce:
#         logger.error(f"[Job {job_id}] Configuration error suggesting exploration strategy: {ce}")
#         # ExplorationStrategy 初期化エラーなどもリスロー
#         raise ce
#     except Exception as e:
#         error_type = type(e).__name__
#         logger.error(f"[Job {job_id}] Unexpected {error_type} suggesting exploration strategy: {e}", exc_info=True)
#         # 予期せぬエラーは MirexError でラップして処理を継続可能にするか検討
#         # ここでは一旦、カスタム例外を発生させてジョブ失敗とする
#         raise MirexError(f"Failed to suggest exploration strategy unexpectedly: {e}") from e

def register_loop_tools(
    mcp: FastMCP,
    config: Dict[str, Any],
    start_async_job_func: Callable[..., Coroutine[Any, Any, str]], # 戻り値は job_id (str) の想定
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]]
):
    """改善ループに関連する非同期タスクをMCPツールとして登録します。"""
    logger.info("Registering improvement loop tools...")

    # --- Helper function to start jobs ---
    async def _start_loop_job(
        task_coroutine_factory: Callable[..., Coroutine],
        tool_name: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """指定されたコルーチンファクトリからタスクを開始し、job_idを返すヘルパー"""
        job_description = f"{tool_name} job" + (f" for session {session_id}" if session_id else "")
        logger.info(f"Requesting to start {job_description} with args: {kwargs}")
        try:
            # ファクトリを呼び出してコルーチンを作成
            task_coroutine = task_coroutine_factory(**kwargs)
            # start_async_job_func を呼び出してジョブを開始
            job_id = await start_async_job_func(task_coroutine, job_description)
            logger.info(f"Successfully started {job_description} with Job ID: {job_id}")
            return {"job_id": job_id}
        except Exception as e:
            logger.error(f"Failed to start {job_description}: {e}", exc_info=True)
            # エラーを返す（MCPツールは通常JSONシリアライズ可能な辞書を返す）
            return {"error": f"Failed to start {tool_name} job: {e}"}

    # --- Register Parameter Optimization Tool ---
    parameter_optimization_factory = partial(
        _run_parameter_optimization,
        config=config,
        add_history_async_func=add_history_async_func
    )

    @mcp.tool("optimize_parameters", input_schema=OptimizeParametersInput)
    async def optimize_parameters_tool(
        detector_name: str,
        code_version: Optional[str] = None,
        audio_dir: Optional[str] = None,
        reference_dir: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """LLM提案とグリッドサーチを組み合わせたパラメータ最適化ジョブを開始します。"""
        # 内部でコード取得を行うため、source_code は不要
        return await _start_loop_job(
            parameter_optimization_factory,
            "optimize_parameters",
            session_id=session_id,
            detector_name=detector_name,
            code_version=code_version,
            audio_dir=audio_dir,
            reference_dir=reference_dir,
            session_id=session_id # _run 関数にも session_id を渡す
        )

    # --- Register Suggest Exploration Strategy Tool (コメントアウト) --- #
    # suggest_strategy_factory = partial(
    #     _run_suggest_exploration_strategy,
    #     config=config,
    #     add_history_async_func=add_history_async_func
    # )
    #
    # @mcp.tool("suggest_exploration_strategy", input_schema=SuggestExplorationStrategyInput)
    # async def suggest_exploration_strategy_tool(
    #     session_id: str,
    #     # current_performance: Optional[float] = None # 削除
    # ) -> Dict[str, Any]:
    #     """現在のセッション履歴に基づいて次の探索戦略を提案するジョブを開始します。"""
    #     return await _start_loop_job(
    #         suggest_strategy_factory,
    #         "suggest_exploration_strategy",
    #         session_id=session_id,
    #         session_id=session_id # _run 関数にも session_id を渡す
    #     )

    logger.info("Improvement loop tools registered.")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 