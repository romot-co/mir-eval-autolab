# mcp_server_logic/llm_tools.py

import os
import json
import logging
import re
import time
import random
import httpx # requests の代わりに非同期 httpx を使用
import asyncio
from typing import Dict, Any, Optional, Callable, Coroutine, Type, List, Union, Tuple, Awaitable # Typeを追加 -> Awaitable追加
import abc # 抽象基底クラスのためにインポート
import traceback # エラーログ用に追加
from pathlib import Path # Pathオブジェクトのため
from functools import partial # Import partial

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP
from src.utils.json_utils import NumpyEncoder # JSONエンコーダー (旧 JsonNumpyEncoder)
from src.utils.exception_utils import retry_on_exception, LLMError, ConfigError # 例外とリトライ
# from anthropic import Anthropic, APIError, APIStatusError # httpx を使うため不要に
from . import db_utils # 履歴保存用 DBアクセス関数
# session_manager から非同期の履歴追加関数をインポート
# 注意: 循環インポートを避けるため、実際の呼び出しはregister関数経由で行うか、
#       依存関係を見直す必要があるかもしれない。ここでは関数シグネチャのみ利用。
from .session_manager import add_session_history # ★ 非同期関数を直接インポート (依存性注入が理想)
from src.utils.path_utils import validate_path_within_allowed_dirs, get_workspace_dir, get_output_base_dir, get_dataset_paths, get_project_root, ensure_dir, get_db_dir # Ensure all needed funcs are imported

# Phase 2 imports
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound
import pydantic
# --- 追加: 履歴イベントスキーマ --- #
from .schemas import (
    # Tool Input Schemas
    ImproveCodeInput, SuggestParametersInput, AnalyzeEvaluationInput,
    # Tool Output/Result Schemas
    ParameterSuggestion,
    CodeCritique,
    EvaluationAnalysis,
    ImproveCodeResultData, # ★ 追加
    # History Event Data Schemas
    ImproveCodeStartedData,
    ImproveCodeCompleteData,
    ImproveCodeFailedData,
    ParameterSuggestionStartedData,
    ParameterSuggestionCompleteData,
    ParameterSuggestionFailedData,
    AnalyzeEvaluationStartedData,
    AnalyzeEvaluationCompleteData,
    AnalyzeEvaluationFailedData,
    # HypothesesGeneratedData, # TODO: スキーマ定義後に追加
    # AssessImprovementData, # TODO: スキーマ定義後に追加
)

logger = logging.getLogger('mcp_server.llm_tools')

# --- Jinja2 Environment Setup (Phase 2) --- #
# Use Path object for template directory
prompt_template_dir = Path(__file__).parent / "prompts"
try:
    jinja_env = Environment(
        loader=FileSystemLoader(prompt_template_dir),
        autoescape=select_autoescape(['html', 'xml']) # Basic autoescaping, adjust if needed
    )
    logger.info(f"Jinja2 environment loaded from: {prompt_template_dir}")
except Exception as e:
    logger.critical(f"Failed to initialize Jinja2 environment from {prompt_template_dir}: {e}", exc_info=True)
    # Fallback or raise critical error?
    # For now, let it potentially fail later if templates are needed.
    jinja_env = None # Indicate failure

# --- Helper Functions --- #

def extract_code_from_text(text: str) -> Optional[str]:
    """テキストからPythonコードブロックを抽出します。"""
    if not text: return None

    pattern_python = r"```python\s*([\s\S]*?)\s*```"
    match_python = re.search(pattern_python, text)
    if match_python: return match_python.group(1).strip()

    pattern_plain = r"```\s*([\s\S]*?)\s*```"
    match_plain = re.search(pattern_plain, text)
    if match_plain:
        code_candidate = match_plain.group(1).strip()
        # JSONでないことを確認する簡易チェック
        likely_json = code_candidate.lstrip().startswith('{') and code_candidate.rstrip().endswith('}')
        if likely_json:
            try:
                json.loads(code_candidate)
                return None # JSONなのでコードではない
            except json.JSONDecodeError:
                pass # JSONではないのでコードの可能性あり
        # Pythonキーワードを含むか簡易チェック
        if any(keyword in code_candidate for keyword in ["def ", "class ", "import ", "from "]):
            return code_candidate

    # フォールバック：全体がコードかもしれない場合
    if text.strip() and any(keyword in text for keyword in ["def ", "class ", "import ", "from "]):
        lines = text.split('\n')
        python_like_lines = 0
        for line in lines[:10]:
            stripped = line.strip()
            if stripped.startswith(("def ", "class ", "import ", "from ", "#")) or (len(line) > 0 and line[0].isspace()):
                python_like_lines += 1
        if python_like_lines >= min(2, len(lines)):
            return text.strip()

    logger.debug("テキストからPythonコードブロックを抽出できませんでした。")
    return None

def mock_llm_response(prompt: str, request_json: bool = False) -> Any:
    """Mock LLM response implementation."""
    logger.info("モックLLM応答を生成します。")
    time.sleep(random.uniform(0.5, 1.5)) # 処理時間を模倣
    code = extract_code_from_text(prompt)
    if not code: code = "# Mock improvement: プロンプトから元のコードが見つかりません\nprint('モック改善コードよりこんにちは！')"
    issues_str = "一般的なパフォーマンスと特定のエッジケース"
    if "リバーブ" in prompt: issues_str = "リバーブ処理"
    elif "ノイズ" in prompt: issues_str = "ノイズ耐性"
    elif "和音" in prompt: issues_str = "ポリフォニー検出"
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    modified_code = f"""\
# Mock LLM Improvement ({current_time})
# Target: {issues_str}
# - 感度パラメータをわずかに調整しました。
# - エラー処理のプレースホルダを追加しました。

{code}

# Mock modification: ダミー関数追加
def _mock_helper_function_{int(time.time())}():
    pass

# Mock modification: 基本的なエラー処理プレースホルダ
try:
    pass
except Exception as e:
    print(f"モックエラーハンドラがキャッチ: {{e}}")
"""
    lines = modified_code.splitlines()
    for idx, line in enumerate(lines):
        # パラメータ調整の例
        if "threshold" in line.lower() and "=" in line:
             parts = line.split('=')
             try:
                 current_val_str = parts[1].split('#')[0].strip()
                 current_val = float(current_val_str)
                 new_val = max(0.01, current_val * random.uniform(0.8, 1.2)) # ランダムに調整
                 lines[idx] = f"{parts[0]}= {new_val:.3f} # Mock adjustment"
                 break
             except ValueError: pass
    modified_code = "\n".join(lines)
    if request_json:
        return {
            "improved_code": modified_code,
            "summary": f"Mock improvement targeting {issues_str}.",
            "changes": ["Adjusted sensitivity parameters (mock)", "Added basic error handling placeholder", "Added dummy helper function."]
        }
    else:
        return modified_code

# --- LLM Client Abstraction --- #

class BaseLLMClient(abc.ABC):
    """LLMクライアントの抽象ベースクラス"""
    @abc.abstractmethod
    async def generate(self, prompt: str, request_json: bool = False) -> Any:
        """指定されたプロンプトでLLMにテキスト生成をリクエストする (非同期)"""
        pass

class ClaudeClient(BaseLLMClient):
    """Anthropic Claude API用クライアント (httpxを使用)"""
    def __init__(self, llm_config: Dict[str, Any], timeouts: Dict[str, Any]):
        self.api_key = llm_config.get('api_key', os.environ.get("ANTHROPIC_API_KEY"))
        self.model = llm_config.get('model', os.environ.get("CLAUDE_MODEL", "claude-3-opus-20240229"))
        self.api_base = llm_config.get('api_base', os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com"))
        self.api_version = llm_config.get('api_version', os.environ.get("CLAUDE_API_VERSION", "2023-06-01"))
        self.timeout = timeouts.get('llm', 180)
        self.max_tokens = llm_config.get('max_tokens', 4096)

        # Desktop mode detection (should be derived from config/env)
        self.is_desktop = llm_config.get('desktop_mode', os.environ.get("CLAUDE_DESKTOP_MODE", "false").lower() in ["true", "1", "yes"])
        self.desktop_url = llm_config.get('desktop_url', os.environ.get("CLAUDE_DESKTOP_URL", "http://localhost:5000"))

        if not self.api_key and not self.is_desktop:
            logger.warning("ClaudeClient: ANTHROPIC_API_KEY not set and not in desktop mode. Mock responses will be used.")
            self.use_mock = True
        else:
            self.use_mock = False

    # リトライデコレータを generate メソッドに適用
    @retry_on_exception(logger=logger, log_message_prefix="Claude API リトライ試行")
    async def generate(self, prompt: str, request_json: bool = False) -> Any:
        """Claude APIを非同期で呼び出し、応答を取得します。（リトライ機能付き）"""
        if self.use_mock:
            # モック応答は同期的だが、async 関数内で実行するために run_in_executor を使用
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, mock_llm_response, prompt, request_json)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if self.is_desktop:
                    headers = {"Content-Type": "application/json", "anthropic-version": self.api_version}
                    endpoint = f"{self.desktop_url}/v1/messages"
                    logger.info(f"Claude Desktop モードでAPIリクエスト実行: {endpoint}")
                else:
                    if not self.api_key: raise LLMError("Anthropic API Key is missing.")
                    headers = {"Content-Type": "application/json", "X-API-Key": self.api_key, "anthropic-version": self.api_version}
                    endpoint = f"{self.api_base}/v1/messages"
                    logger.info(f"Anthropic Cloud APIモードでリクエスト実行: {endpoint}")

                system_prompt_json = "Please provide the output strictly in valid JSON format."
                messages = [{"role": "user", "content": prompt}]
                payload = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages}
                if request_json: payload["system"] = system_prompt_json

                logger.debug(f"Sending prompt to Claude (request_json={request_json}): {prompt[:200]}...")
                response = await client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status() # HTTPエラーチェック

            data = response.json()
            # Claude API v3 の応答形式に合わせる
            if not data.get("content") or not isinstance(data["content"], list) or not data["content"][0].get("text"):
                 logger.error(f"Claude応答形式が予期しません: {data}")
                 raise LLMError("Claude応答のcontentが見つからないか不正です。")

            llm_response_text = data["content"][0]["text"]
            logger.debug(f"Received Claude response: {llm_response_text[:200]}...")

            if request_json:
                try:
                    # JSONの抽出 (マークダウンブロック考慮)
                    if llm_response_text.strip().startswith("```json"):
                         processed_text = llm_response_text.strip()[7:].strip()
                         end_idx = processed_text.find("```")
                         if end_idx > 0: processed_text = processed_text[:end_idx].strip()
                    elif llm_response_text.strip().startswith("```"):
                         processed_text = llm_response_text.strip()[3:].strip()
                         end_idx = processed_text.find("```")
                         if end_idx > 0: processed_text = processed_text[:end_idx].strip()
                    else:
                         processed_text = llm_response_text

                    # 応答からJSONを抽出
                    json_response = json.loads(processed_text)
                    return json_response
                except json.JSONDecodeError as json_err:
                    logger.warning(f"Failed to find code block, returning raw response: {llm_response_text[:100]}...")
                    # コードブロックが見つからない場合は、応答全体をそのまま返す
                    code = extract_code_from_text(llm_response_text)
                    if not code:
                        logger.warning("Could not extract code even after attempt, returning raw text.")
                    return {"error": "JSON parse failed and no code block found", "raw_response": llm_response_text[:500]}
            else:
                # ★ JSONを要求しない場合、コード抽出を非同期化
                loop = asyncio.get_running_loop()
                code = loop.run_in_executor(None, extract_code_from_text, llm_response_text)
                return code if code else llm_response_text

        except httpx.TimeoutException as e:
            logger.error(f"Claude API呼び出しがタイムアウト ({self.timeout}秒)")
            raise LLMError(f"Claude API call timed out: {e}") from e
        except httpx.RequestError as req_err:
            logger.error(f"Claude APIリクエストエラー: {req_err}")
            err_details = str(req_err)
            if req_err.response is not None:
                try: err_details += f"\nResponse ({req_err.response.status_code}): {req_err.response.text}"
                except Exception: pass
            raise LLMError(f"Claude API request error: {err_details}") from req_err
        # except APIError as api_err: # Anthropic SDKの例外もキャッチ (もしSDKを使う場合)
        #      logger.error(f"Anthropic API Error: {api_err}")
        #      raise LLMError(f"Anthropic API Error: {api_err}") from api_err
        except Exception as e:
            logger.error(f"予期せぬClaude APIエラー: {e}", exc_info=True)
            raise LLMError(f"Unexpected Claude API error: {e}") from e


# --- LLM Client Factory --- #
# (コメント: サーバー起動時に生成し、DIするのが望ましい)
_llm_client_instance: Optional[BaseLLMClient] = None

def get_llm_client(llm_config: Dict[str, Any], timeouts: Dict[str, Any]) -> BaseLLMClient:
    """設定に基づいてLLMクライアントのインスタンスを取得または作成します。"""
    global _llm_client_instance
    if _llm_client_instance is None:
        client_type = llm_config.get('client_type', 'ClaudeClient') # configから取得
        if client_type == 'ClaudeClient':
            _llm_client_instance = ClaudeClient(llm_config, timeouts)
        # elif client_type == 'OpenAIClient':
        #     # _llm_client_instance = OpenAIClient(llm_config, timeouts) # 実装例
        else:
            logger.warning(f"サポートされていないLLMクライアントタイプ '{client_type}'。ClaudeClient(またはモック)にフォールバックします。")
            _llm_client_instance = ClaudeClient(llm_config, timeouts)
        logger.info(f"LLMクライアントを初期化しました: {type(_llm_client_instance).__name__}")
    return _llm_client_instance


# --- Helper function to get LLM Client --- #
# (コメント: 上記 Factory または DI に統合すべき)
def _get_llm_client(config: dict) -> BaseLLMClient:
    llm_config = config.get('llm', {})
    api_provider = llm_config.get('provider', 'anthropic').lower()
    api_key = llm_config.get('api_key')
    model_name = llm_config.get('model', 'claude-3-opus-20240229')

    if not api_key:
        raise ConfigError("LLM API key is not configured.")

    if api_provider == 'anthropic':
        return ClaudeClient(api_key=api_key, default_model=model_name)
    # Add other providers like OpenAI here
    # elif api_provider == 'openai':
    #     return OpenAIClient(api_key=api_key, default_model=model_name)
    else:
        raise ConfigError(f"Unsupported LLM provider: {api_provider}")

# --- Helper function to render prompt (Phase 2) --- #
async def _render_prompt(
    template_name: str,
    context: Dict[str, Any],
    session_id: Optional[str] = None,
    config: Optional[dict] = None, # Needed for history summary
    db_path: Optional[Path] = None # Needed for history summary
) -> str:
    """Jinja2 テンプレートを非同期でレンダリングし、履歴サマリーを含める"""
    if not jinja_env:
        raise LLMError("Jinja2 environment is not initialized.")

    # セッション履歴サマリーを取得 (非同期で)
    session_history_summary = "No session history available or feature disabled."
    if session_id and config and db_path:
        try:
            # session_manager モジュールからヘルパー関数を呼び出す (依存関係注意)
            from .session_manager import get_session_summary_for_prompt
            session_history_summary = await get_session_summary_for_prompt(
                config=config,
                db_path=db_path,
                session_id=session_id
                # max_summary_length=config.get('strategy', {}).get('prompt_history_limit', 2000)
            )
        except ImportError:
            logger.error("Could not import get_session_summary_for_prompt from session_manager. Skipping history summary.")
        except Exception as e:
            logger.error(f"Error fetching session summary for prompt: {e}", exc_info=True)
            session_history_summary = f"Error fetching session history: {e}"

    # コンテキストに履歴サマリーを追加
    full_context = context.copy()
    full_context['session_history_summary'] = session_history_summary

    try:
        template = jinja_env.get_template(f"{template_name}.j2")
        # ★ レンダリングは同期的だが、CPUバウンドの可能性があるので Executor を使う
        loop = asyncio.get_running_loop()
        # template.render は同期関数
        prompt = template.render(full_context)
        logger.debug(f"Rendered prompt ({template_name}):\n{prompt[:500]}...")
        return prompt
    except TemplateNotFound:
        logger.error(f"Prompt template not found: {template_name}.j2 in {prompt_template_dir}")
        raise LLMError(f"Prompt template '{template_name}.j2' not found.")
    except Exception as e:
        logger.error(f"Error rendering prompt template '{template_name}': {e}", exc_info=True)
        raise LLMError(f"Error rendering prompt template '{template_name}': {e}") from e

# --- LLM Tool Task Functions (Refactored for Phase 2) --- #

async def _run_improve_code(
    job_id: str, # job_id を追加
    config: dict,
    add_history_func: Callable[..., Awaitable[None]], # ★ 戻り値を None に
    # --- スキーマに合わせて引数変更 --- #
    session_id: str,
    code: str, # detector_code -> code
    suggestion: str, # user_goal や hypotheses を統合した指示
    # evaluation_results: Optional[Dict[str, Any]] = None, # 必要なら追加
    # --- ここまで --- #
    original_code_version_hash: Optional[str] = None # 履歴用
) -> ImproveCodeResultData: # ★ 戻り値をスキーマに
    """LLMにコード改善を依頼し、結果の ImproveCodeResultData を返す（非同期、履歴記録強化）"""
    logger.info(f"[Job {job_id}] Starting code improvement... Session: {session_id}")
    start_time = time.time()
    db_path = Path(config['paths']['db_path'])
    llm_client = _get_llm_client(config)
    # hypothesis_to_test は suggestion に含まれる想定

    # --- 開始履歴 --- #
    try:
        start_event_data = ImproveCodeStartedData(
            job_id=job_id,
            original_code_version_hash=original_code_version_hash,
            # hypothesis_to_test は suggestion の一部として記録するか？
            # prompt_used は後で記録
        )
        await add_history_func(session_id, "improve_code_started", start_event_data.model_dump(exclude_none=True))
    except Exception as hist_e:
        logger.warning(f"[Job {job_id}] Failed to add improve_code_started history: {hist_e}")

    prompt = "" # スコープ外で参照できるように初期化
    raw_response: Optional[str] = None
    try:
        # プロンプトをレンダリング
        context = {
            "code_to_improve": code,
            "improvement_suggestion": suggestion,
            # evaluation_results や user_goal も必要ならコンテキストに追加
        }
        prompt = await _render_prompt("improve_code", context, session_id, config, db_path)

        # LLM呼び出し (コード文字列を期待)
        raw_response = await llm_client.generate(prompt, request_json=False)

        # レスポンスからコードを抽出
        improved_code = extract_code_from_text(raw_response)
        if not improved_code:
            raise LLMError("LLM response did not contain a valid Python code block.", raw_response=raw_response)

        # 構文チェック
        import ast
        try:
            ast.parse(improved_code)
        except SyntaxError as se:
            raise LLMError(f"Generated code has syntax errors: {se}", raw_response=raw_response, error_type="SyntaxError") from se

        elapsed_time = time.time() - start_time
        logger.info(f"[Job {job_id}] Code improvement successful ({elapsed_time:.2f}s). Session: {session_id}")

        # TODO: LLM応答から変更サマリーを抽出するロジック
        changes_summary = f"Improved code based on suggestion: {suggestion[:50]}..." # 仮

        # 結果をスキーマに格納
        result_data = ImproveCodeResultData(code=improved_code, summary=changes_summary)

        # --- 完了履歴 --- #
        try:
            complete_event_data = ImproveCodeCompleteData(
                job_id=job_id,
                original_code_version_hash=original_code_version_hash,
                new_code_version_hash=None, # save_code ツールが責任を持つ
                hypothesis_tested=None, # suggestion から抽出 or 別途管理
                changes_summary=changes_summary,
                llm_response_raw=raw_response,
            )
            await add_history_func(session_id, "improve_code_complete", complete_event_data.model_dump(exclude_none=True))
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add improve_code_complete history: {hist_e}")

        return result_data

    except LLMError as llm_err:
        elapsed_time = time.time() - start_time
        logger.error(f"[Job {job_id}] LLM error during code improvement ({elapsed_time:.2f}s): {llm_err}", exc_info=False)
        try:
            fail_event_data = ImproveCodeFailedData(
                job_id=job_id,
                original_code_version_hash=original_code_version_hash,
                # hypothesis_tested=hypothesis_to_test,
                error=str(llm_err),
                error_type=llm_err.error_type or 'LLMError',
                prompt_used=prompt,
                llm_response_raw=llm_err.raw_response or raw_response
            )
            await add_history_func(session_id, "improve_code_failed", fail_event_data.model_dump(exclude_none=True))
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add improve_code_failed history (LLMError): {hist_e}")
        raise llm_err
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_type = type(e).__name__
        logger.error(f"[Job {job_id}] Unexpected {error_type} during code improvement ({elapsed_time:.2f}s): {e}", exc_info=True)
        try:
            fail_event_data = ImproveCodeFailedData(
                job_id=job_id,
                original_code_version_hash=original_code_version_hash,
                # hypothesis_tested=hypothesis_to_test,
                error=str(e),
                error_type=error_type,
                prompt_used=prompt,
                llm_response_raw=raw_response
            )
            await add_history_func(session_id, "improve_code_failed", fail_event_data.model_dump(exclude_none=True))
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add improve_code_failed history (Unexpected): {hist_e}")
        raise LLMError(f"Unexpected error during code improvement: {e}") from e

async def _run_suggest_parameters(
    job_id: str,
    config: dict,
    add_history_func: Callable[..., Awaitable[None]], # ★ 戻り値を None に
    # --- スキーマに合わせて引数変更 --- #
    session_id: str,
    analysis_results: Optional[Dict[str, Any]] = None,
    current_metrics: Optional[Dict[str, Any]] = None,
    cycle_state: Optional[Dict[str, Any]] = None, # SessionCycleState 辞書
    # detector_code: str, # 不要？ サーバー側で取得可能か
    # user_goal: Optional[str] = None
    # --- ここまで --- #
) -> ParameterSuggestion: # ★ 戻り値をスキーマに
    """LLMにパラメータ提案を依頼し、結果の ParameterSuggestion を返す（非同期、履歴スキーマ使用）"""
    logger.info(f"[Job {job_id}] Starting parameter suggestion... Session: {session_id}")
    start_time = time.time()
    db_path = Path(config['paths']['db_path'])
    llm_client = _get_llm_client(config)

    # --- 開始履歴 --- #
    try:
        start_event_data = ParameterSuggestionStartedData(job_id=job_id)
        await add_history_func(session_id, "parameter_suggestion_started", start_event_data.model_dump(exclude_none=True))
    except Exception as hist_e:
        logger.warning(f"[Job {job_id}] Failed to add parameter_suggestion_started history: {hist_e}")

    prompt = "" # スコープ外で参照できるように初期化
    raw_response: Optional[Any] = None
    try:
        # TODO: 必要なら detector_code をDBから取得
        detector_code = "# Detector code not fetched in this version"
        # プロンプトをレンダリング
        context = {
            "detector_code": detector_code,
            "analysis_results": analysis_results,
            "current_metrics": current_metrics,
            "cycle_state": cycle_state,
            # "user_goal": user_goal
        }
        prompt = await _render_prompt("suggest_parameters", context, session_id, config, db_path)

        # LLM呼び出し (JSON応答を期待)
        raw_response = await llm_client.generate(prompt, request_json=True)

        # Pydanticモデルで検証
        try:
            validated_response = ParameterSuggestion.model_validate(raw_response)
        except pydantic.ValidationError as val_err:
            logger.warning(f"LLM response failed validation for ParameterSuggestion: {val_err}")
            raise LLMError(f"LLM response validation failed: {val_err}", raw_response=raw_response, error_type="ValidationError") from val_err

        elapsed_time = time.time() - start_time
        logger.info(f"[Job {job_id}] Parameter suggestion successful ({elapsed_time:.2f}s). Session: {session_id}")

        # --- 完了履歴 --- #
        try:
            complete_event_data = ParameterSuggestionCompleteData(
                job_id=job_id,
                suggestion=validated_response,
                prompt_used=prompt,
                llm_response_raw=raw_response
            )
            await add_history_func(session_id, "parameter_suggestion_complete", complete_event_data.model_dump(exclude_none=True))
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add parameter_suggestion_complete history: {hist_e}")

        return validated_response # ★ スキーマオブジェクトを返す

    except LLMError as llm_err:
        elapsed_time = time.time() - start_time
        logger.error(f"[Job {job_id}] LLM error during parameter suggestion ({elapsed_time:.2f}s): {llm_err}", exc_info=False)
        try:
            fail_event_data = ParameterSuggestionFailedData(
                job_id=job_id,
                error=str(llm_err),
                error_type=llm_err.error_type or 'LLMError',
                prompt_used=prompt,
                llm_response_raw=llm_err.raw_response or raw_response
            )
            await add_history_func(session_id, "parameter_suggestion_failed", fail_event_data.model_dump(exclude_none=True))
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add parameter_suggestion_failed history (LLMError): {hist_e}")
        raise llm_err
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_type = type(e).__name__
        logger.error(f"[Job {job_id}] Unexpected {error_type} during parameter suggestion ({elapsed_time:.2f}s): {e}", exc_info=True)
        try:
            fail_event_data = ParameterSuggestionFailedData(
                job_id=job_id,
                error=str(e),
                error_type=error_type,
                prompt_used=prompt,
                llm_response_raw=raw_response
            )
            await add_history_func(session_id, "parameter_suggestion_failed", fail_event_data.model_dump(exclude_none=True))
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add parameter_suggestion_failed history (Unexpected): {hist_e}")
        raise LLMError(f"Unexpected error during parameter suggestion: {e}") from e

async def _run_analyze_evaluation_results(
    job_id: str,
    config: dict,
    add_history_func: Callable[..., Awaitable[None]], # ★ 戻り値を None に
    # --- スキーマに合わせて引数変更 --- #
    session_id: str,
    evaluation_results: Dict[str, Any],
    # detector_code: Optional[str] = None, # 不要？
    # user_goal: Optional[str] = None
    # --- ここまで --- #
) -> EvaluationAnalysis: # ★ 戻り値をスキーマに
    """評価結果を分析し、EvaluationAnalysis を返す（非同期、履歴スキーマ使用）"""
    logger.info(f"[Job {job_id}] Starting evaluation analysis... Session: {session_id}")
    start_time = time.time()
    db_path = Path(config['paths']['db_path'])
    llm_client = _get_llm_client(config)
    schema_class = EvaluationAnalysis

    # --- 開始履歴 --- #
    try:
        start_event_data = AnalyzeEvaluationStartedData(job_id=job_id)
        await add_history_func(session_id, "analyze_evaluation_results_started", start_event_data.model_dump(exclude_none=True))
    except Exception as hist_e:
        logger.warning(f"[Job {job_id}] Failed to add analyze_evaluation_results_started history: {hist_e}")

    prompt = "" # スコープ外で参照できるように初期化
    raw_response: Optional[Any] = None
    try:
        # TODO: 必要なら detector_code をDBから取得
        detector_code = "# Detector code not fetched in this version"
        # プロンプトをレンダリング
        context = {
            'evaluation_results': evaluation_results,
            'detector_code': detector_code,
            # 'user_goal': user_goal
        }
        prompt = await _render_prompt('analyze_evaluation_results', context, session_id, config, db_path)

        # LLM呼び出し (JSON応答を期待)
        raw_response = await llm_client.generate(prompt, request_json=True)

        # Pydanticモデルで検証
        try:
            validated_response = schema_class.model_validate(raw_response)
            logger.info(f"[Job {job_id}] LLM analysis passed validation")
        except pydantic.ValidationError as e:
            logger.error(f"[Job {job_id}] LLM analysis failed validation: {e}")
            raise LLMError(f"LLM analysis response failed validation: {e}", raw_response=raw_response, error_type="ValidationError") from e
        except Exception as e:
            logger.error(f"[Job {job_id}] Error parsing/validating LLM analysis response: {e}", exc_info=True)
            raise LLMError(f"Failed to parse or validate LLM analysis JSON response: {e}", raw_response=raw_response, error_type="ParsingError")

        elapsed_time = time.time() - start_time
        logger.info(f"[Job {job_id}] Evaluation analysis successful ({elapsed_time:.2f}s). Session: {session_id}")

        # --- 完了履歴 --- #
        try:
            complete_event_data = AnalyzeEvaluationCompleteData(
                job_id=job_id,
                analysis=validated_response,
                prompt_used=prompt,
                llm_response_raw=raw_response
            )
            await add_history_func(session_id, 'analyze_evaluation_results_complete', complete_event_data.model_dump(exclude_none=True))
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add analyze_evaluation_results_complete history: {hist_e}")

        return validated_response # ★ スキーマオブジェクトを返す

    except LLMError as llm_err:
        elapsed_time = time.time() - start_time
        logger.error(f"[Job {job_id}] LLM error during evaluation analysis ({elapsed_time:.2f}s): {llm_err}", exc_info=False)
        try:
            fail_event_data = AnalyzeEvaluationFailedData(
                job_id=job_id,
                error=str(llm_err),
                error_type=llm_err.error_type or 'LLMError',
                prompt_used=prompt,
                llm_response_raw=llm_err.raw_response or raw_response
            )
            await add_history_func(session_id, 'analyze_evaluation_results_failed', fail_event_data.model_dump(exclude_none=True))
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add analyze_evaluation_results_failed history (LLMError): {hist_e}")
        raise llm_err
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_type = type(e).__name__
        logger.error(f"[Job {job_id}] Unexpected {error_type} during evaluation analysis ({elapsed_time:.2f}s): {e}", exc_info=True)
        try:
            fail_event_data = AnalyzeEvaluationFailedData(
                job_id=job_id,
                error=str(e),
                error_type=error_type,
                prompt_used=prompt,
                llm_response_raw=raw_response
            )
            await add_history_func(session_id, 'analyze_evaluation_results_failed', fail_event_data.model_dump(exclude_none=True))
        except Exception as hist_e:
            logger.warning(f"[Job {job_id}] Failed to add analyze_evaluation_results_failed history (Unexpected): {hist_e}")
        raise LLMError(f"Unexpected error during evaluation analysis: {e}") from e

# --- MCPツール登録 --- #

def register_llm_tools(
    mcp: FastMCP,
    config: dict,
    start_async_job_func: Callable[..., Awaitable[Dict[str, str]]], # ★ 戻り値修正
    add_history_async_func: Callable[..., Awaitable[None]] # ★ 戻り値修正
):
    """LLM関連タスクをMCPツールとして登録します。"""
    logger.info("Registering LLM tools...")

    # --- Helper to start jobs --- #
    async def _start_llm_job(
        task_coroutine_factory: Callable[..., Coroutine],
        tool_name: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]: # ★ JobStartResponse 形式
        """LLMタスクを開始し、JobStartResponse 辞書を返すヘルパー"""
        job_description = f"LLM {tool_name} job" + (f" for session {session_id}" if session_id else "")
        logger.info(f"Requesting to start {job_description} with args: {kwargs}")
        try:
            # job_id は start_async_job_func で生成される想定
            task_coroutine = task_coroutine_factory(**kwargs)
            # ★ tool_name, session_id を渡すように変更
            job_start_response = await start_async_job_func(task_coroutine, tool_name=tool_name, session_id=session_id)
            logger.info(f"Successfully requested {job_description}. Job info: {job_start_response}")
            return job_start_response
        except Exception as e:
            logger.error(f"Failed to start {job_description}: {e}", exc_info=True)
            from .schemas import JobStatus # Local import to avoid potential circular dep
            return {"error": f"Failed to start {tool_name} job: {e}", "status": JobStatus.FAILED.value}

    # --- improve_code ツール --- #
    improve_code_factory = partial(
        _run_improve_code,
        config=config,
        add_history_func=add_history_async_func
    )
    # ★ 入力スキーマ適用
    @mcp.tool(
        name="improve_code",
        description="Asks the LLM to improve the provided Python code based on a suggestion. Returns the improved code.",
        input_schema=ImproveCodeInput # ★ スキーマ指定
    )
    async def improve_code_tool(
        # ★ シグネチャをスキーマに合わせる
        session_id: str,
        code: str,
        suggestion: str,
        # detector_code: str, # 旧引数削除
        # evaluation_results: Optional[Dict[str, Any]] = None,
        # user_goal: Optional[str] = None,
        # hypotheses: Optional[list[str]] = None,
        # previous_feedback: Optional[str] = None,
        original_code_version_hash: Optional[str] = None # 履歴用なので残す
    ) -> Dict[str, Any]: # ★ JobStartResponse 形式
        # スキーマ検証済み引数を渡す
        kwargs = {
            "session_id": session_id,
            "code": code,
            "suggestion": suggestion,
            "original_code_version_hash": original_code_version_hash
        }
        return await _start_llm_job(
            improve_code_factory,
            "improve_code",
            session_id=session_id,
            **kwargs
        )

    # --- suggest_parameters ツール --- #
    suggest_parameters_factory = partial(
        _run_suggest_parameters,
        config=config,
        add_history_func=add_history_async_func
    )
    # ★ 入力スキーマ適用
    @mcp.tool(
        name="suggest_parameters",
        description="Asks the LLM to suggest parameters to explore based on analysis. Returns a JSON object with suggestions.",
        input_schema=SuggestParametersInput # ★ スキーマ指定
    )
    async def suggest_parameters_tool(
        # ★ シグネチャをスキーマに合わせる
        session_id: str,
        analysis_results: Optional[Dict[str, Any]] = None,
        current_metrics: Optional[Dict[str, Any]] = None,
        cycle_state: Optional[Dict[str, Any]] = None,
        # detector_code: str, # 旧引数削除
        # evaluation_results: Optional[Dict[str, Any]] = None,
        # user_goal: Optional[str] = None
    ) -> Dict[str, Any]: # ★ JobStartResponse 形式
        kwargs = {
            "session_id": session_id,
            "analysis_results": analysis_results,
            "current_metrics": current_metrics,
            "cycle_state": cycle_state,
        }
        return await _start_llm_job(
            suggest_parameters_factory,
            "suggest_parameters",
            session_id=session_id,
            **kwargs
        )

    # --- analyze_evaluation_results ツール --- #
    analyze_evaluation_factory = partial(
        _run_analyze_evaluation_results,
        config=config,
        add_history_func=add_history_async_func
    )
    # ★ 入力スキーマ適用
    @mcp.tool(
        name="analyze_evaluation_results",
        description="Asks the LLM to analyze evaluation results and provide a structured analysis (JSON).",
        input_schema=AnalyzeEvaluationInput # ★ スキーマ指定
    )
    async def analyze_evaluation_results_tool(
        # ★ シグネチャをスキーマに合わせる
        session_id: str,
        evaluation_results: Dict[str, Any],
        # detector_code: Optional[str] = None, # 旧引数削除
        # user_goal: Optional[str] = None
    ) -> Dict[str, Any]: # ★ JobStartResponse 形式
        kwargs = {
            "session_id": session_id,
            "evaluation_results": evaluation_results,
        }
        return await _start_llm_job(
            analyze_evaluation_factory,
            "analyze_evaluation_results",
            session_id=session_id,
            **kwargs
        )

    # TODO: Add registration for generate_hypotheses, assess_improvement tools

    logger.info("LLM tools registered.")

# --- Helper to remove helper functions if not needed outside --- #
# del mock_llm_response
# del extract_code_from_text

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.")
