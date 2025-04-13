# src/mcp_server_logic/llm_tools.py

import os
import json
import logging
import re
import time
import random
# import httpx # 削除：HTTP クライアントは不要
import asyncio
from typing import Dict, Any, Optional, Callable, Coroutine, Type, List, Union, Tuple, Awaitable
# import abc # 削除：抽象基底クラスは不要
import traceback
from pathlib import Path
from functools import partial
import ast # コード構文チェック用

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ValidationError, Field # Field を追加
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound

from src.utils.json_utils import NumpyEncoder
# from src.utils.exception_utils import retry_on_exception, LLMError, ConfigError, StateManagementError # retry_on_exception 削除
from src.utils.exception_utils import LLMError, ConfigError, StateManagementError, MirexError # MirexError を追加
from src.utils.path_utils import get_workspace_dir, get_output_base_dir, get_dataset_paths, get_project_root, ensure_dir, get_db_dir, validate_path_within_allowed_dirs
from . import db_utils
from . import session_manager # 履歴記録、セッション情報取得のため
from . import schemas # Pydantic スキーマ
from src.utils.exception_utils import ConfigError, FileError

logger = logging.getLogger('mcp_server.llm_tools')

# --- Jinja2 Environment Setup ---
# Use Path object for template directory
prompt_template_dir = Path(__file__).parent / "prompts"
try:
    jinja_env = Environment(
        loader=FileSystemLoader(prompt_template_dir),
        autoescape=select_autoescape(['html', 'xml']), # Basic autoescaping
        trim_blocks=True, lstrip_blocks=True # Template readability improvements
    )
    logger.info(f"Jinja2 environment loaded from: {prompt_template_dir}")
except Exception as e:
    logger.critical(f"Failed to initialize Jinja2 environment from {prompt_template_dir}: {e}", exc_info=True)
    jinja_env = None # Indicate failure

# --- Prompt Rendering Helper ---
async def _render_prompt(
    template_name: str,
    context: Dict[str, Any],
    session_id: Optional[str] = None,
    config: Optional[dict] = None,
    db_path: Optional[Path] = None
) -> str:
    if not jinja_env:
        raise MirexError("Jinja2 environment is not initialized.", error_type="ConfigurationError") # LLMErrorからMirexErrorに変更

    session_history_summary = "No session history available."
    if session_id and config and db_path:
        try:
            session_history_summary = await session_manager.get_session_summary_for_prompt(
                config=config, db_path=db_path, session_id=session_id
            )
        except Exception as e:
            logger.error(f"Error fetching session summary for prompt context: {e}", exc_info=True)
            session_history_summary = f"Error fetching session history: {e}"

    full_context = context.copy()
    full_context['session_history_summary'] = session_history_summary

    try:
        template = jinja_env.get_template(f"{template_name}.j2")
        # Rendering is CPU-bound but typically fast enough to run directly in async context
        prompt = template.render(full_context)
        logger.debug(f"Rendered prompt template '{template_name}.j2'")
        return prompt
    except TemplateNotFound:
        logger.error(f"Prompt template not found: {template_name}.j2 in {prompt_template_dir}")
        raise MirexError(f"Prompt template '{template_name}.j2' not found.", error_type="ConfigurationError")
    except Exception as e:
        logger.error(f"Error rendering prompt template '{template_name}': {e}", exc_info=True)
        raise MirexError(f"Error rendering prompt template '{template_name}': {e}", error_type="PromptRenderingError") from e

# --- LLM Tool Async Task Functions ---

# 共通のプロンプト生成処理をまとめるヘルパー
async def _generate_prompt_task_base(
    job_id: str,
    config: dict,
    add_history_func: Callable[..., Awaitable[None]],
    session_id: str,
    template_name: str,
    context: Dict[str, Any],
    history_event_prefix: str,
    start_event_schema: Type[BaseModel],
    complete_event_schema: Type[BaseModel],
    fail_event_schema: Type[BaseModel],
    **kwargs # Specific start event data
) -> schemas.PromptData:
    """プロンプト生成タスクの共通処理"""
    logger.info(f"[Job {job_id}] Starting prompt generation ({template_name})... Session: {session_id}")
    start_time = time.monotonic()
    
    # DBディレクトリのパスを安全に取得
    try:
        db_dir = get_db_dir(config)
        # 許可されたディレクトリを設定（ワークスペースディレクトリ内のDBディレクトリのみ）
        allowed_dirs = [get_workspace_dir(config)]
        # DBパスを検証
        validated_db_dir = validate_path_within_allowed_dirs(
            db_dir,
            allowed_base_dirs=allowed_dirs,
            check_existence=True,
            check_is_file=False,  # ディレクトリを期待
            allow_absolute=True   # 絶対パスを許可
        )
        db_path = validated_db_dir / db_utils.DB_FILENAME
        # DBファイルも検証
        db_path = validate_path_within_allowed_dirs(
            db_path,
            allowed_base_dirs=allowed_dirs,
            check_existence=False,  # DBファイルが存在しない場合もあり得る
            check_is_file=None,     # 存在チェックを行わないため不要
            allow_absolute=True     # 絶対パスを許可
        )
    except (ConfigError, FileError, ValueError) as e:
        logger.error(f"[Job {job_id}] Failed to validate DB path: {e}")
        raise MirexError(f"Failed to validate DB path: {e}", error_type="PathValidationError") from e

    # 開始履歴
    try:
        start_event_data = start_event_schema(job_id=job_id, **kwargs)
        await add_history_func(session_id, f"{history_event_prefix}_prompt_generation_started", start_event_data.model_dump())
    except Exception as hist_e:
        logger.warning(f"[Job {job_id}] Failed to add {history_event_prefix}_prompt_generation_started history: {hist_e}")

    generated_prompt = ""
    try:
        generated_prompt = await _render_prompt(template_name, context, session_id, config, db_path)

        elapsed_time = time.monotonic() - start_time
        logger.info(f"[Job {job_id}] Prompt generation ({template_name}) successful ({elapsed_time:.2f}s).")

        result_data = schemas.PromptData(prompt=generated_prompt)

        # 完了履歴
        try:
            complete_event_data = complete_event_schema(job_id=job_id, prompt=generated_prompt)
            await add_history_func(session_id, f"{history_event_prefix}_prompt_generation_complete", complete_event_data.model_dump())
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add {history_event_prefix}_prompt_generation_complete history: {hist_e}")

        return result_data

    except MirexError as mirex_err: # PromptRenderingError や ConfigurationError をキャッチ
        elapsed_time = time.monotonic() - start_time
        logger.error(f"[Job {job_id}] MirexError during prompt generation ({template_name}) ({elapsed_time:.2f}s): {mirex_err}", exc_info=False)
        try:
            fail_event_data = fail_event_schema(
                job_id=job_id, error=str(mirex_err), error_type=mirex_err.error_type or 'PromptGenerationError',
                context_used=context # Context を含める
            )
            await add_history_func(session_id, f"{history_event_prefix}_prompt_generation_failed", fail_event_data.model_dump())
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add {history_event_prefix}_prompt_generation_failed history (MirexError): {hist_e}")
        raise mirex_err # Re-raise MirexError
    except Exception as e:
        elapsed_time = time.monotonic() - start_time
        error_type = type(e).__name__
        logger.error(f"[Job {job_id}] Unexpected {error_type} during prompt generation ({template_name}) ({elapsed_time:.2f}s): {e}", exc_info=True)
        try:
            fail_event_data = fail_event_schema(
                job_id=job_id, error=str(e), error_type=error_type, context_used=context
            )
            await add_history_func(session_id, f"{history_event_prefix}_prompt_generation_failed", fail_event_data.model_dump())
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add {history_event_prefix}_prompt_generation_failed history (Unexpected): {hist_e}")
        # 予期せぬエラーは MirexError でラップして再raiseする
        raise MirexError(f"Unexpected error during prompt generation: {e}", error_type="UnexpectedPromptGenerationError") from e


async def _run_get_improve_code_prompt_async(
    job_id: str,
    config: dict,
    add_history_func: Callable[..., Awaitable[None]],
    session_id: str,
    code: str,
    suggestion: str,
    original_code_version_hash: Optional[str] = None
) -> schemas.PromptData:
    """コード改善プロンプトを生成する"""
    context = {"code_to_improve": code, "improvement_suggestion": suggestion}
    return await _generate_prompt_task_base(
        job_id, config, add_history_func, session_id,
        template_name="improve_code",
        context=context,
        history_event_prefix="improve_code",
        start_event_schema=schemas.ImproveCodePromptGenerationStartedData,
        complete_event_schema=schemas.ImproveCodePromptGenerationCompleteData,
        fail_event_schema=schemas.ImproveCodePromptGenerationFailedData,
        original_code_version_hash=original_code_version_hash,
        improvement_suggestion=suggestion
    )

async def _run_get_suggest_parameters_prompt_async(
    job_id: str,
    config: dict,
    add_history_func: Callable[..., Awaitable[None]],
    session_id: str,
    detector_code: str,
    analysis_results: Optional[Dict[str, Any]] = None,
    current_metrics: Optional[Dict[str, Any]] = None,
    cycle_state: Optional[Dict[str, Any]] = None,
) -> schemas.PromptData:
    """パラメータ提案プロンプトを生成する"""
    context = {
        "detector_code": detector_code,
        "analysis_results": analysis_results,
        "current_metrics": current_metrics,
        "cycle_state": cycle_state,
    }
    return await _generate_prompt_task_base(
        job_id, config, add_history_func, session_id,
        template_name="suggest_parameters",
        context=context,
        history_event_prefix="parameter_suggestion",
        start_event_schema=schemas.ParameterSuggestionPromptGenerationStartedData,
        complete_event_schema=schemas.ParameterSuggestionPromptGenerationCompleteData,
        fail_event_schema=schemas.ParameterSuggestionPromptGenerationFailedData,
    )

async def _run_get_analyze_evaluation_prompt_async(
    job_id: str,
    config: dict,
    add_history_func: Callable[..., Awaitable[None]],
    session_id: str,
    evaluation_results: Dict[str, Any],
    # detector_code: Optional[str] = None, # コードは含めない
) -> schemas.PromptData:
    """評価結果分析プロンプトを生成する"""
    context = {'evaluation_results': evaluation_results}
    return await _generate_prompt_task_base(
        job_id, config, add_history_func, session_id,
        template_name="analyze_evaluation_results",
        context=context,
        history_event_prefix="analyze_evaluation",
        start_event_schema=schemas.AnalyzeEvaluationPromptGenerationStartedData,
        complete_event_schema=schemas.AnalyzeEvaluationPromptGenerationCompleteData,
        fail_event_schema=schemas.AnalyzeEvaluationPromptGenerationFailedData,
        evaluation_results_summary=str(evaluation_results)[:500] # 簡単なサマリー
    )

async def _run_get_suggest_exploration_strategy_prompt_async(
    job_id: str,
    config: dict,
    add_history_func: Callable[..., Awaitable[None]],
    session_id: str,
) -> schemas.PromptData:
    """探索戦略提案プロンプトを生成する"""
    # DBパスを安全に取得して検証
    try:
        db_dir = get_db_dir(config)
        allowed_dirs = [get_workspace_dir(config)]
        validated_db_dir = validate_path_within_allowed_dirs(
            db_dir,
            allowed_base_dirs=allowed_dirs,
            check_existence=True,
            check_is_file=False,
            allow_absolute=True
        )
        db_path = validated_db_dir / db_utils.DB_FILENAME
        db_path = validate_path_within_allowed_dirs(
            db_path,
            allowed_base_dirs=allowed_dirs,
            check_existence=False,
            check_is_file=None,
            allow_absolute=True
        )
    except (ConfigError, FileError, ValueError) as e:
        logger.error(f"[Job {job_id}] Failed to validate DB path for strategy prompt: {e}")
        raise MirexError(f"Failed to validate DB path for strategy prompt: {e}", error_type="PathValidationError") from e
    
    try:
        session_info = await session_manager.get_session_info(config, db_path, session_id)
        # model_validate の代わりに ** で展開して SessionCycleState インスタンスを作成
        cycle_state_data = session_info.cycle_state if session_info.cycle_state else {}
        cycle_state_obj = schemas.SessionCycleState(**cycle_state_data)
        context = {
            "session_id": session_id, "base_algorithm": session_info.base_algorithm,
            "user_goal": session_info.improvement_goal, "cycle_state": cycle_state_obj.model_dump(),
            "current_metrics": session_info.current_metrics,
        }
    except Exception as e:
        logger.error(f"Failed to get session info for strategy prompt context (session: {session_id}): {e}", exc_info=True)
        raise MirexError(f"Could not retrieve session info for strategy prompt: {e}") from e

    return await _generate_prompt_task_base(
        job_id, config, add_history_func, session_id,
        template_name="suggest_exploration_strategy",
        context=context,
        history_event_prefix="strategy_suggestion",
        start_event_schema=schemas.StrategySuggestionPromptGenerationStartedData,
        complete_event_schema=schemas.StrategySuggestionPromptGenerationCompleteData,
        fail_event_schema=schemas.StrategySuggestionPromptGenerationFailedData,
    )

async def _run_get_generate_hypotheses_prompt_async(
    job_id: str,
    config: dict,
    add_history_func: Callable[..., Awaitable[None]],
    session_id: str,
    num_hypotheses: int,
    analysis_results: Optional[Dict[str, Any]] = None,
    current_metrics: Optional[Dict[str, Any]] = None,
) -> schemas.PromptData:
    """改善仮説生成プロンプトを生成する"""
    # DBパスを安全に取得して検証
    try:
        db_dir = get_db_dir(config)
        allowed_dirs = [get_workspace_dir(config)]
        validated_db_dir = validate_path_within_allowed_dirs(
            db_dir,
            allowed_base_dirs=allowed_dirs,
            check_existence=True,
            check_is_file=False,
            allow_absolute=True
        )
        db_path = validated_db_dir / db_utils.DB_FILENAME
        db_path = validate_path_within_allowed_dirs(
            db_path,
            allowed_base_dirs=allowed_dirs,
            check_existence=False,
            check_is_file=None,
            allow_absolute=True
        )
    except (ConfigError, FileError, ValueError) as e:
        logger.error(f"[Job {job_id}] Failed to validate DB path for hypotheses prompt: {e}")
        raise MirexError(f"Failed to validate DB path for hypotheses prompt: {e}", error_type="PathValidationError") from e
    
    user_goal = "Not available"
    try:
        session_info = await session_manager.get_session_info(config, db_path, session_id)
        user_goal = session_info.improvement_goal
    except Exception as e:
        logger.warning(f"Could not fetch user goal for hypotheses prompt (session: {session_id}): {e}")

    context = {
        "evaluation_results": current_metrics, "analysis_results": analysis_results,
        "num_hypotheses": num_hypotheses, "user_goal": user_goal,
    }
    return await _generate_prompt_task_base(
        job_id, config, add_history_func, session_id,
        template_name="suggest_improvement_hypothesis",
        context=context,
        history_event_prefix="hypotheses_generation",
        start_event_schema=schemas.HypothesesGenerationPromptGenerationStartedData,
        complete_event_schema=schemas.HypothesesGenerationPromptGenerationCompleteData,
        fail_event_schema=schemas.HypothesesGenerationPromptGenerationFailedData,
        num_hypotheses_requested=num_hypotheses
    )

async def _run_get_assess_improvement_prompt_async(
    job_id: str,
    config: dict,
    add_history_func: Callable[..., Awaitable[None]],
    session_id: str,
    original_detector_code: str,
    improved_detector_code: str,
    evaluation_results_before: Dict[str, Any],
    evaluation_results_after: Dict[str, Any],
    hypothesis_tested: Optional[str] = None,
    user_goal: Optional[str] = None,
    previous_feedback: Optional[str] = None,
) -> schemas.PromptData:
    """改善評価プロンプトを生成する"""
    context = {
        "original_detector_code": original_detector_code, "improved_detector_code": improved_detector_code,
        "evaluation_results_before": evaluation_results_before, "evaluation_results_after": evaluation_results_after,
        "hypothesis_tested": hypothesis_tested, "user_goal": user_goal, "previous_feedback": previous_feedback,
    }
    return await _generate_prompt_task_base(
        job_id, config, add_history_func, session_id,
        template_name="assess_improvement",
        context=context,
        history_event_prefix="assess_improvement",
        start_event_schema=schemas.AssessImprovementPromptGenerationStartedData,
        complete_event_schema=schemas.AssessImprovementPromptGenerationCompleteData,
        fail_event_schema=schemas.AssessImprovementPromptGenerationFailedData,
        hypothesis_tested=hypothesis_tested,
    )

# --- MCPツール登録 (Refactored) ---
def register_llm_tools(
    mcp: FastMCP,
    config: dict,
    start_async_job_func: Callable[..., Awaitable[Dict[str, str]]],
    add_history_async_func: Callable[..., Awaitable[None]]
):
    logger.info("Registering LLM Prompt Generation tools...")

    # DBパス解決ロジックはジョブ開始時に移動 (仕様書 2.1.1)
    # def get_db_path():
    #     try:
    #         db_dir = get_db_dir(config)
    #         return db_dir / db_utils.DB_FILENAME
    #     except (ConfigError, FileError) as path_err:
    #         logger.error(f"Failed to determine DB path for LLM tool job: {path_err}")
    #         raise path_err

    async def _start_prompt_gen_job(
        task_coroutine_factory: Callable[..., Coroutine],
        tool_name: str, # Example: "get_improve_code_prompt"
        session_id: Optional[str] = None,
        **kwargs # Specific args for the task factory
    ) -> Dict[str, Any]: # JobStartResponse 形式
        job_description = f"Prompt Generation job ({tool_name})" + (f" for session {session_id}" if session_id else "")
        logger.info(f"Requesting to start {job_description} with args: {kwargs}")
        try:
            # ★ db_path はここで解決しない (task_coroutine_factory 内で解決される)
            # db_path = get_db_path()
            # db_path は config 経由で渡される想定

            # add_history_func を部分適用
            task_coroutine_partial = partial(
                task_coroutine_factory,
                config=config,
                add_history_func=add_history_async_func, # Inject add_history_func
                **kwargs
            )
            # ジョブを開始 (start_async_job_func は外部から渡される)
            job_start_response = await start_async_job_func(
                task_coroutine_partial,
                tool_name=tool_name, # Use specific prompt gen tool name
                session_id=session_id
            )
            logger.info(f"Successfully requested {job_description}. Job info: {job_start_response}")
            return job_start_response
        except Exception as e:
            logger.error(f"Failed to start {job_description}: {e}", exc_info=True)
            # JobStartResponse 形式でエラーを返す
            return {"job_id": None, "status": schemas.JobStatus.FAILED.value, "error": f"Failed to start {tool_name} job: {e}"}

    # Improve Code Prompt Tool
    @mcp.tool("get_improve_code_prompt")
    async def get_improve_code_prompt_tool(session_id: str, code: str, suggestion: str, original_code_version_hash: Optional[str] = None) -> Dict[str, Any]:
        """コード改善のためのプロンプトを生成するジョブを開始します。"""
        return await _start_prompt_gen_job(
            _run_get_improve_code_prompt_async, "get_improve_code_prompt", session_id,
            code=code, suggestion=suggestion, original_code_version_hash=original_code_version_hash
        )

    # Suggest Parameters Prompt Tool
    @mcp.tool("get_suggest_parameters_prompt")
    async def get_suggest_parameters_prompt_tool(session_id: str, detector_code: str, analysis_results: Optional[Dict[str, Any]] = None, current_metrics: Optional[Dict[str, Any]] = None, cycle_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """パラメータ提案のためのプロンプトを生成するジョブを開始します。"""
        return await _start_prompt_gen_job(
            _run_get_suggest_parameters_prompt_async, "get_suggest_parameters_prompt", session_id,
            detector_code=detector_code, analysis_results=analysis_results, current_metrics=current_metrics, cycle_state=cycle_state
        )

    # Analyze Evaluation Prompt Tool
    @mcp.tool("get_analyze_evaluation_prompt")
    async def get_analyze_evaluation_prompt_tool(session_id: str, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """評価結果分析のためのプロンプトを生成するジョブを開始します。"""
        return await _start_prompt_gen_job(
            _run_get_analyze_evaluation_prompt_async, "get_analyze_evaluation_prompt", session_id,
            evaluation_results=evaluation_results
        )

    # Suggest Exploration Strategy Prompt Tool
    @mcp.tool("get_suggest_exploration_strategy_prompt")
    async def get_suggest_exploration_strategy_prompt_tool(session_id: str) -> Dict[str, Any]:
        """探索戦略提案のためのプロンプトを生成するジョブを開始します。"""
        return await _start_prompt_gen_job(
            _run_get_suggest_exploration_strategy_prompt_async, "get_suggest_exploration_strategy_prompt", session_id
        )

    # Generate Hypotheses Prompt Tool
    @mcp.tool("get_generate_hypotheses_prompt")
    async def get_generate_hypotheses_prompt_tool(session_id: str, num_hypotheses: int = 3, analysis_results: Optional[Dict[str, Any]] = None, current_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """改善仮説生成のためのプロンプトを生成するジョブを開始します。"""
        return await _start_prompt_gen_job(
            _run_get_generate_hypotheses_prompt_async, "get_generate_hypotheses_prompt", session_id,
            num_hypotheses=num_hypotheses, analysis_results=analysis_results, current_metrics=current_metrics
        )

    # Assess Improvement Prompt Tool
    @mcp.tool("get_assess_improvement_prompt")
    async def get_assess_improvement_prompt_tool(session_id: str, original_detector_code: str, improved_detector_code: str, evaluation_results_before: Dict[str, Any], evaluation_results_after: Dict[str, Any], hypothesis_tested: Optional[str] = None, user_goal: Optional[str] = None, previous_feedback: Optional[str] = None) -> Dict[str, Any]:
        """改善評価のためのプロンプトを生成するジョブを開始します。"""
        return await _start_prompt_gen_job(
            _run_get_assess_improvement_prompt_async, "get_assess_improvement_prompt", session_id,
            original_detector_code=original_detector_code, improved_detector_code=improved_detector_code, evaluation_results_before=evaluation_results_before, evaluation_results_after=evaluation_results_after, hypothesis_tested=hypothesis_tested, user_goal=user_goal, previous_feedback=previous_feedback
        )

    # --- 評価分析ツール ---
    @mcp.tool("analyze_evaluation")
    async def analyze_evaluation_tool(
        session_id: str,
        evaluation_results: Dict[str, Any],
        detector_code: Optional[str] = None,
        user_goal: Optional[str] = None
    ) -> Dict[str, Any]:
        """評価結果を分析し、強みと弱み、改善点を特定します。"""
        try:
            logger.info(f"評価分析ツール実行: セッション {session_id}")
            # 評価分析プロンプト生成の非同期タスクを実行
            task_factory = partial(
                _run_get_analyze_evaluation_prompt_async,
                config=config,
                add_history_func=add_history_async_func,
                session_id=session_id,
                evaluation_results=evaluation_results
            )
            
            # プロンプト生成ジョブを開始
            job_response = await _start_prompt_gen_job(
                task_factory,
                "get_analyze_evaluation_prompt",
                session_id=session_id
            )
            return job_response
            
        except Exception as e:
            logger.error(f"評価分析ツール実行中にエラーが発生: {e}", exc_info=True)
            return {"job_id": None, "status": schemas.JobStatus.FAILED.value, "error": f"Failed to run analyze_evaluation tool: {e}"}
    
    # --- 仮説生成ツール ---
    @mcp.tool("generate_hypotheses")
    async def generate_hypotheses_tool(
        session_id: str,
        num_hypotheses: int = 3,
        analysis_results: Optional[Dict[str, Any]] = None,
        current_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """アルゴリズム改善のための仮説を生成します。"""
        try:
            logger.info(f"仮説生成ツール実行: セッション {session_id}, 仮説数 {num_hypotheses}")
            # 仮説生成プロンプト生成の非同期タスクを実行
            task_factory = partial(
                _run_get_generate_hypotheses_prompt_async,
                config=config,
                add_history_func=add_history_async_func,
                session_id=session_id,
                num_hypotheses=num_hypotheses,
                analysis_results=analysis_results,
                current_metrics=current_metrics
            )
            
            # プロンプト生成ジョブを開始
            job_response = await _start_prompt_gen_job(
                task_factory,
                "get_generate_hypotheses_prompt",
                session_id=session_id
            )
            return job_response
            
        except Exception as e:
            logger.error(f"仮説生成ツール実行中にエラーが発生: {e}", exc_info=True)
            return {"job_id": None, "status": schemas.JobStatus.FAILED.value, "error": f"Failed to run generate_hypotheses tool: {e}"}
    
    # --- 戦略提案ツール ---
    @mcp.tool("suggest_exploration_strategy")
    async def suggest_exploration_strategy_tool(session_id: str) -> Dict[str, Any]:
        """次に取るべきアクションの戦略を提案します。"""
        try:
            logger.info(f"戦略提案ツール実行: セッション {session_id}")
            # 戦略提案プロンプト生成の非同期タスクを実行
            task_factory = partial(
                _run_get_suggest_exploration_strategy_prompt_async,
                config=config,
                add_history_func=add_history_async_func,
                session_id=session_id
            )
            
            # プロンプト生成ジョブを開始
            job_response = await _start_prompt_gen_job(
                task_factory,
                "get_suggest_exploration_strategy_prompt",
                session_id=session_id
            )
            return job_response
            
        except Exception as e:
            logger.error(f"戦略提案ツール実行中にエラーが発生: {e}", exc_info=True)
            return {"job_id": None, "status": schemas.JobStatus.FAILED.value, "error": f"Failed to run suggest_exploration_strategy tool: {e}"}
    
    # --- 改善効果評価ツール ---
    @mcp.tool("assess_improvement")
    async def assess_improvement_tool(
        session_id: str,
        original_code: str,
        improved_code: str,
        evaluation_results_before: Dict[str, Any],
        evaluation_results_after: Dict[str, Any],
        hypothesis_tested: Optional[str] = None,
        user_goal: Optional[str] = None,
        previous_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """コード改善の効果を評価します。"""
        try:
            logger.info(f"改善効果評価ツール実行: セッション {session_id}")
            # 改善評価プロンプト生成の非同期タスクを実行
            task_factory = partial(
                _run_get_assess_improvement_prompt_async,
                config=config,
                add_history_func=add_history_async_func,
                session_id=session_id,
                original_detector_code=original_code,
                improved_detector_code=improved_code,
                evaluation_results_before=evaluation_results_before,
                evaluation_results_after=evaluation_results_after,
                hypothesis_tested=hypothesis_tested,
                user_goal=user_goal,
                previous_feedback=previous_feedback
            )
            
            # プロンプト生成ジョブを開始
            job_response = await _start_prompt_gen_job(
                task_factory,
                "get_assess_improvement_prompt",
                session_id=session_id
            )
            return job_response
            
        except Exception as e:
            logger.error(f"改善効果評価ツール実行中にエラーが発生: {e}", exc_info=True)
            return {"job_id": None, "status": schemas.JobStatus.FAILED.value, "error": f"Failed to run assess_improvement tool: {e}"}

    logger.info("LLMプロンプト生成ツールと追加MCPツールの登録完了")

async def initialize_llm_client_async(api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
    """LLMクライアントを非同期に初期化
    
    Args:
        api_key: APIキー（Noneの場合は環境変数から取得）
        model: 使用するモデル名
        
    Returns:
        初期化されたLLMクライアントインスタンス
    """
    from src.cli.llm_client import AnthropicClient
    return AnthropicClient(api_key=api_key, model=model)

# --- Import Guard ---
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.")