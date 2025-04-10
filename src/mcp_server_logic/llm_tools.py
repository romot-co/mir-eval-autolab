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

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP
from .serialization_utils import JsonNumpyEncoder # JSONエンコーダー
from src.utils.exception_utils import retry_on_exception, LLMError # 例外とリトライ
# from anthropic import Anthropic, APIError, APIStatusError # httpx を使うため不要に
from . import db_utils # 履歴保存用 DBアクセス関数
# session_manager から非同期の履歴追加関数をインポート
# 注意: 循環インポートを避けるため、実際の呼び出しはregister関数経由で行うか、
#       依存関係を見直す必要があるかもしれない。ここでは関数シグネチャのみ利用。
# from .session_manager import add_session_history
from src.utils.path_utils import validate_path_within_allowed_dirs, get_workspace_dir, get_output_base_dir, get_dataset_paths, get_project_root, ensure_dir, get_db_dir # Ensure all needed funcs are imported

logger = logging.getLogger('mcp_server.llm_tools')

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

                    # JSONパースは同期的なので Executor を使う
                    loop = asyncio.get_running_loop()
                    json_response = await loop.run_in_executor(None, json.loads, processed_text)

                    if not isinstance(json_response, (dict, list)):
                        logger.warning(f"Claude JSON応答が辞書またはリストではありません: {type(json_response)}")
                        raise json.JSONDecodeError("応答が有効なJSONオブジェクト/配列ではありません", processed_text, 0)
                    return json_response
                except json.JSONDecodeError as json_err:
                    logger.warning(f"Claudeが要求にも関わらず有効なJSONを返しませんでした。エラー: {json_err}. Raw text: '{llm_response_text[:100]}...'")
                    # JSONパース失敗時のフォールバック
                    loop = asyncio.get_running_loop()
                    code = await loop.run_in_executor(None, extract_code_from_text, llm_response_text)
                    if code:
                        return {"improved_code": code, "summary": "Raw text response (JSON parse failed).", "changes": [], "error": "JSON parse failed, extracted code as fallback."}
                    else:
                        return {"error": "JSON parse failed and no code block found", "raw_response": llm_response_text[:500]}
            else:
                # JSONを要求しない場合、コード抽出を試みる
                loop = asyncio.get_running_loop()
                code = await loop.run_in_executor(None, extract_code_from_text, llm_response_text)
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


# --- Async Task Functions (修正版) --- #

async def _run_improve_code(
    job_id: str,
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]], # 履歴追加関数 (非同期)
    llm_config: Dict[str, Any], # LLM設定
    timeouts: Dict[str, Any], # タイムアウト設定
    db_path: Path, # DBパス
    session_id: Optional[str], # セッションID
    original_code: str, # 対象コード
    target_issue: Optional[str] = None, # 改善目標
    context: Optional[Dict[str, Any]] = None # 追加コンテキスト
) -> Dict[str, Any]:
    """LLMを使用してコードを改善する非同期タスク"""
    logger.info(f"[Job {job_id}] Running improve_code task... Session: {session_id}")
    get_timestamp = time.time

    # 履歴追加: 開始
    if session_id:
        try:
             await add_history_async_func(session_id, "llm_improve_started", {"target_issue": target_issue, "context": context})
        except Exception as hist_e:
             logger.warning(f"[Job {job_id}] Failed to add improve_code started history: {hist_e}")

    try:
        llm_client = get_llm_client(llm_config, timeouts)
        # プロンプト生成
        prompt = f"""Improve the following Python code. Address the target issue: '{target_issue or "General improvement"}'.
Context: {json.dumps(context, cls=JsonNumpyEncoder) if context else "None"}

Original Code:
```python
{original_code}
```

Return ONLY the improved Python code block within ```python ... ```, optionally preceded by a brief summary of changes (max 3 bullet points)."
"""

        # LLM呼び出し (コードブロックを期待)
        raw_response = await llm_client.generate(prompt, request_json=False)

        # コードブロック抽出
        improved_code = extract_code_from_text(raw_response)

        if not improved_code:
            logger.warning(f"[Job {job_id}] LLM did not return a valid Python code block. Using raw response as fallback.")
            # フォールバックとして、生の応答を使うか、エラーにするか？
            # ここでは生の応答を含めてエラー扱いとする
            result = {
                 "status": "failed",
                 "error": "LLM did not return a valid Python code block.",
                 "raw_response": raw_response[:1000] # 応答の一部を記録
            }
            if session_id:
                try:
                     await add_history_async_func(session_id, "llm_improve_failed", {"error": result["error"], "raw_response_preview": result["raw_response"][:200]})
                except Exception as hist_e:
                    logger.warning(f"[Job {job_id}] Failed to add history for LLM code block error: {hist_e}")
            return result

        # 成功時の結果
        result = {
            "status": "completed",
            "improved_code": improved_code,
            # "summary": "..." # 応答からサマリーを抽出するロジックを追加可能
        }

        # 履歴追加: 完了
        if session_id:
            try:
                await add_history_async_func(session_id, "llm_improve_complete", {
                    "result_preview": improved_code[:200] + "..." # コードの一部を記録
                    # "summary": result.get("summary")
                })
            except Exception as hist_e:
                logger.warning(f"[Job {job_id}] Failed to add improve_code complete history: {hist_e}")

        return result

    except LLMError as llm_err:
        logger.error(f"[Job {job_id}] LLMError during improve_code: {llm_err}", exc_info=True)
        if session_id:
            try: await add_history_async_func(session_id, "llm_improve_failed", {"error": f"LLMError: {llm_err}"})
            except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add history for LLMError: {hist_e}")
        raise # LLMError はそのまま上位に伝播させる (job_worker で failed にする)
    except Exception as e:
        logger.error(f"[Job {job_id}] Unexpected error during improve_code: {e}", exc_info=True)
        if session_id:
            try: await add_history_async_func(session_id, "llm_improve_failed", {"error": f"Unexpected Error: {e}"})
            except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add history for unexpected error: {hist_e}")
        raise # 他の例外も上位に伝播

async def _run_suggest_improvement(
    job_id: str,
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    llm_config: Dict[str, Any],
    timeouts: Dict[str, Any],
    db_path: Path,
    session_id: Optional[str],
    code: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """コードの改善点を提案する非同期タスク"""
    logger.info(f"[Job {job_id}] Running suggest_improvement task... Session: {session_id}")
    get_timestamp = time.time

    if session_id:
        try: await add_history_async_func(session_id, "llm_suggest_started", {"context": context})
        except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add suggest_improvement started history: {hist_e}")

    try:
        llm_client = get_llm_client(llm_config, timeouts)
        prompt = f"""Analyze the following Python code and suggest specific improvements, focusing on performance, readability, and robustness. Provide suggestions as a JSON list of objects, each with 'suggestion' (str) and 'priority' (int, 1-3, 3=high).
Context: {json.dumps(context, cls=JsonNumpyEncoder) if context else "None"}

Code:
```python
{code}
```

Respond ONLY with the valid JSON list.
"""
        # LLM呼び出し (JSONを期待)
        suggestions = await llm_client.generate(prompt, request_json=True)

        if not isinstance(suggestions, list):
            logger.warning(f"[Job {job_id}] LLM did not return a valid JSON list for suggestions. Response type: {type(suggestions)}")
            result = {"status": "failed", "error": "LLM did not return a valid JSON list.", "raw_response": str(suggestions)[:1000]}
            if session_id:
                 try: await add_history_async_func(session_id, "llm_suggest_failed", {"error": result["error"], "raw_response_preview": result["raw_response"][:200]})
                 except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add history for LLM list error: {hist_e}")
            return result

        result = {"status": "completed", "suggestions": suggestions}

        if session_id:
             try: await add_history_async_func(session_id, "llm_suggest_complete", {"suggestion_count": len(suggestions)})
             except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add suggest_improvement complete history: {hist_e}")

        return result

    except LLMError as llm_err:
        logger.error(f"[Job {job_id}] LLMError during suggest_improvement: {llm_err}", exc_info=True)
        if session_id:
            try: await add_history_async_func(session_id, "llm_suggest_failed", {"error": f"LLMError: {llm_err}"})
            except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add history for LLMError: {hist_e}")
        raise
    except Exception as e:
        logger.error(f"[Job {job_id}] Unexpected error during suggest_improvement: {e}", exc_info=True)
        if session_id:
            try: await add_history_async_func(session_id, "llm_suggest_failed", {"error": f"Unexpected Error: {e}"})
            except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add history for unexpected error: {hist_e}")
        raise

async def _run_summarize_code(
    job_id: str,
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    llm_config: Dict[str, Any],
    timeouts: Dict[str, Any],
    db_path: Path,
    session_id: Optional[str],
    code: str
) -> Dict[str, Any]:
    """コードの概要を生成する非同期タスク"""
    logger.info(f"[Job {job_id}] Running summarize_code task... Session: {session_id}")
    get_timestamp = time.time

    if session_id:
        try: await add_history_async_func(session_id, "llm_summarize_started", {})
        except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add summarize_code started history: {hist_e}")

    try:
        llm_client = get_llm_client(llm_config, timeouts)
        prompt = f"""Provide a concise summary (max 100 words) of the following Python code, explaining its purpose and main functionality.

Code:
```python
{code}
```

Respond ONLY with the summary text.
"""
        summary_text = await llm_client.generate(prompt, request_json=False)

        result = {"status": "completed", "summary": summary_text}

        if session_id:
            try: await add_history_async_func(session_id, "llm_summarize_complete", {"summary_preview": summary_text[:100] + "..."})
            except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add summarize_code complete history: {hist_e}")

        return result

    except LLMError as llm_err:
        logger.error(f"[Job {job_id}] LLMError during summarize_code: {llm_err}", exc_info=True)
        if session_id:
            try: await add_history_async_func(session_id, "llm_summarize_failed", {"error": f"LLMError: {llm_err}"})
            except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add history for LLMError: {hist_e}")
        raise
    except Exception as e:
        logger.error(f"[Job {job_id}] Unexpected error during summarize_code: {e}", exc_info=True)
        if session_id:
            try: await add_history_async_func(session_id, "llm_summarize_failed", {"error": f"Unexpected Error: {e}"})
            except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add history for unexpected error: {hist_e}")
        raise

async def _run_critique_code(
    job_id: str,
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    llm_config: Dict[str, Any],
    timeouts: Dict[str, Any],
    db_path: Path,
    session_id: Optional[str],
    code: str
) -> Dict[str, Any]:
    """コードの潜在的な問題を批評する非同期タスク"""
    logger.info(f"[Job {job_id}] Running critique_code task... Session: {session_id}")
    get_timestamp = time.time

    if session_id:
        try: await add_history_async_func(session_id, "llm_critique_started", {})
        except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add critique_code started history: {hist_e}")

    try:
        llm_client = get_llm_client(llm_config, timeouts)
        prompt = f"""Critique the following Python code for potential issues such as bugs, performance bottlenecks, security vulnerabilities, and style inconsistencies. Provide feedback as a JSON list of objects, each with 'issue' (str) and 'severity' (str: 'low', 'medium', 'high').

Code:
```python
{code}
```

Respond ONLY with the valid JSON list.
"""
        critique_list = await llm_client.generate(prompt, request_json=True)

        if not isinstance(critique_list, list):
            logger.warning(f"[Job {job_id}] LLM did not return a valid JSON list for critique. Response type: {type(critique_list)}")
            result = {"status": "failed", "error": "LLM did not return a valid JSON list.", "raw_response": str(critique_list)[:1000]}
            if session_id:
                try: await add_history_async_func(session_id, "llm_critique_failed", {"error": result["error"], "raw_response_preview": result["raw_response"][:200]})
                except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add history for LLM list error: {hist_e}")
            return result

        result = {"status": "completed", "critique": critique_list}

        if session_id:
            try: await add_history_async_func(session_id, "llm_critique_complete", {"critique_count": len(critique_list)})
            except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add critique_code complete history: {hist_e}")

        return result

    except LLMError as llm_err:
        logger.error(f"[Job {job_id}] LLMError during critique_code: {llm_err}", exc_info=True)
        if session_id:
            try: await add_history_async_func(session_id, "llm_critique_failed", {"error": f"LLMError: {llm_err}"})
            except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add history for LLMError: {hist_e}")
        raise
    except Exception as e:
        logger.error(f"[Job {job_id}] Unexpected error during critique_code: {e}", exc_info=True)
        if session_id:
            try: await add_history_async_func(session_id, "llm_critique_failed", {"error": f"Unexpected Error: {e}"})
            except Exception as hist_e: logger.warning(f"[Job {job_id}] Failed to add history for unexpected error: {hist_e}")
        raise


# --- Tool Registration --- #

def register_llm_tools(
    mcp: FastMCP,
    config: Dict[str, Any],
    start_async_job_func: Callable[..., Coroutine[Any, Any, str]],
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    # db_path は不要？ task関数に渡すように修正
):
    """MCPサーバーにLLM関連のツールを登録"""
    logger.info("Registering LLM tools...")

    llm_config = config.get('llm', {})
    timeouts = config.get('resource_limits', {})

    # DBパスを取得 (タスク関数に渡すため)
    try:
        db_path = get_db_dir(config)
    except (ConfigError, FileError) as e:
        logger.critical(f"Failed to get DB path during LLM tool registration: {e}. LLM tools may fail to record history.")
        # DBパスが取得できない場合でもツール自体は登録するが、警告を出す
        # 代替パスを設定 (非推奨)
        db_path = Path("./fallback_mcp_state.db")
        logger.error(f"Using fallback DB path for LLM tools: {db_path}")
        # raise ConfigError(f"Failed to initialize DB path for LLM tools: {e}") from e

    # パス検証用の許可リスト (LLMツールは通常コード文字列を扱うが、将来用に設定)
    allowed_base_dirs: List[Path] = []
    try:
        project_root = get_project_root()
        workspace_dir = get_workspace_dir(config)
        output_base_dir = get_output_base_dir(config)
        allowed_base_dirs = [project_root, workspace_dir, output_base_dir]
        logger.info(f"LLM tools: Allowed base directories for future path validation: {[str(p) for p in allowed_base_dirs]}")
    except (ConfigError, FileError) as e:
        logger.warning(f"Failed to determine allowed base directories for LLM tools: {e}")
        # 許可リストがなくてもツール自体は登録できる

    # --- Improve Code Tool --- #
    @mcp.tool("improve_code")
    async def improve_code_tool(
        original_code: str,
        target_issue: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        session_id: str = ""
    ) -> Dict[str, Any]:
        """LLMを使用して指定されたPythonコードを改善します。"""
        logger.info(f"Tool 'improve_code' called. Session: {session_id}")
        try:
            job_id = await start_async_job_func(
                _run_improve_code, # 非同期タスク関数
                "improve_code", # ツール名
                session_id,
                # --- _run_improve_code への引数 --- #
                job_id=None, # start_async_job_func が生成？
                add_history_async_func=add_history_async_func,
                llm_config=llm_config,
                timeouts=timeouts,
                db_path=db_path,
                session_id=session_id,
                original_code=original_code,
                target_issue=target_issue,
                context=context
            )
            return {"job_id": job_id, "status": "pending"}
        except (StateManagementError, Exception) as e:
            logger.error(f"Failed to start improve_code job: {e}", exc_info=True)
            return {"error": f"Failed to start job: {e}"}

    # --- Suggest Improvement Tool --- #
    @mcp.tool("suggest_improvement")
    async def suggest_improvement_tool(
        code: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: str = ""
    ) -> Dict[str, Any]:
        """LLMを使用してコードの改善点を提案します（JSON形式）。"""
        logger.info(f"Tool 'suggest_improvement' called. Session: {session_id}")
        try:
            job_id = await start_async_job_func(
                _run_suggest_improvement,
                "suggest_improvement",
                session_id,
                # --- _run_suggest_improvement への引数 --- #
                job_id=None,
                add_history_async_func=add_history_async_func,
                llm_config=llm_config,
                timeouts=timeouts,
                db_path=db_path,
                session_id=session_id,
                code=code,
                context=context
            )
            return {"job_id": job_id, "status": "pending"}
        except (StateManagementError, Exception) as e:
            logger.error(f"Failed to start suggest_improvement job: {e}", exc_info=True)
            return {"error": f"Failed to start job: {e}"}

    # --- Summarize Code Tool --- #
    @mcp.tool("summarize_code")
    async def summarize_code_tool(
        code: str,
        session_id: str = ""
    ) -> Dict[str, Any]:
        """LLMを使用してコードの概要を生成します。"""
        logger.info(f"Tool 'summarize_code' called. Session: {session_id}")
        try:
            job_id = await start_async_job_func(
                _run_summarize_code,
                "summarize_code",
                session_id,
                # --- _run_summarize_code への引数 --- #
                job_id=None,
                add_history_async_func=add_history_async_func,
                llm_config=llm_config,
                timeouts=timeouts,
                db_path=db_path,
                session_id=session_id,
                code=code
            )
            return {"job_id": job_id, "status": "pending"}
        except (StateManagementError, Exception) as e:
            logger.error(f"Failed to start summarize_code job: {e}", exc_info=True)
            return {"error": f"Failed to start job: {e}"}

    # --- Critique Code Tool --- #
    @mcp.tool("critique_code")
    async def critique_code_tool(
        code: str,
        session_id: str = ""
    ) -> Dict[str, Any]:
        """LLMを使用してコードの潜在的な問題を批評します（JSON形式）。"""
        logger.info(f"Tool 'critique_code' called. Session: {session_id}")
        try:
            job_id = await start_async_job_func(
                _run_critique_code,
                "critique_code",
                session_id,
                # --- _run_critique_code への引数 --- #
                job_id=None,
                add_history_async_func=add_history_async_func,
                llm_config=llm_config,
                timeouts=timeouts,
                db_path=db_path,
                session_id=session_id,
                code=code
            )
            return {"job_id": job_id, "status": "pending"}
        except (StateManagementError, Exception) as e:
            logger.error(f"Failed to start critique_code job: {e}", exc_info=True)
            return {"error": f"Failed to start job: {e}"}

    logger.info("LLM tools registered.")

# --- Helper to remove helper functions if not needed outside --- #
# del mock_llm_response
# del extract_code_from_text

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.")
