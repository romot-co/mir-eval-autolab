# mcp_server_logic/llm_tools.py

import os
import json
import logging
import re
import time
import random
import httpx # requests の代わりに非同期 httpx を使用
import asyncio
from typing import Dict, Any, Optional, Callable, Coroutine, Type, List, Union, Tuple # Typeを追加
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
    llm_config: Dict[str, Any],
    timeouts: Dict[str, Any],
    db_path: Path,
    prompt: str,
    add_history_func: Callable, # 非同期の履歴追加関数を受け取る
    session_id: str = ""
) -> Dict[str, Any]:
    """LLMを使用してコードを改善するジョブの実行関数 (非同期)"""
    logger.info(f"[Job {job_id}] コード改善を実行中... Session: {session_id}")
    try:
        llm_client = get_llm_client(llm_config, timeouts)
        response = await llm_client.generate(prompt, request_json=True)

        result = {}
        if isinstance(response, dict):
            result = response
            # 応答辞書内にコードが含まれているか確認し、なければ抽出試行
            if "improved_code" not in result or not result["improved_code"]:
                 loop = asyncio.get_running_loop()
                 found_code = None
                 # 他のキーも試す
                 for key in ["code", "solution", "python_code"]:
                      if key in result and isinstance(result[key], str):
                           found_code = result[key]; break
                 # それでも見つからなければ raw_response から抽出
                 if not found_code and "raw_response" in result:
                      found_code = await loop.run_in_executor(None, extract_code_from_text, result["raw_response"])
                 if found_code: result["improved_code"] = found_code
        elif isinstance(response, str):
            # 文字列応答の場合、コード抽出を試みる
            loop = asyncio.get_running_loop()
            extracted_code = await loop.run_in_executor(None, extract_code_from_text, response)
            result = {"improved_code": extracted_code, "summary": "LLMからRawテキスト応答"}
        else:
             raise LLMError(f"予期しないLLM応答タイプ: {type(response)}")

        # 改善コードが含まれているか最終チェック
        if "improved_code" not in result or not result["improved_code"]:
            error_msg = result.get("error", "LLM応答から有効なコードを抽出できませんでした")
            logger.error(f"[Job {job_id}] コード改善失敗: {error_msg}")
            if session_id:
                 try:
                     # 非同期の履歴追加関数を使用
                     await add_history_func(db_path=db_path, session_id=session_id, event_type="improve_code_failed", event_data={"error": error_msg, "prompt": prompt[:500]}) # プロンプトは切り詰める
                 except Exception as hist_e: logger.warning(f"[Job {job_id}] 失敗履歴の追加に失敗: {hist_e}")
            return {"error": error_msg, **result} # エラー情報を含む結果を返す

        # 成功履歴の追加
        if session_id:
            try:
                changes_summary = result.get("summary", "変更概要なし")
                modified_elements = result.get("changes", [])
                await add_history_func(
                    db_path=db_path, session_id=session_id, event_type="improve_code_result",
                    event_data={"summary": changes_summary, "modified_elements": modified_elements, "job_id": job_id}
                )
            except Exception as hist_e:
                logger.warning(f"[Job {job_id}] 成功履歴の追加に失敗: {hist_e}")

        return result

    except Exception as e:
        logger.error(f"[Job {job_id}] _run_improve_code でエラー: {e}", exc_info=True)
        if session_id:
             try:
                await add_history_func(db_path=db_path, session_id=session_id, event_type="improve_code_failed", event_data={"error": str(e), "prompt": prompt[:500]})
             except Exception as hist_e:
                logger.warning(f"[Job {job_id}] エラー時の履歴追加に失敗: {hist_e}")
        raise # エラーを再発生させ、ジョブを失敗させる

async def _run_analyze_evaluation_results(
    job_id: str,
    llm_config: Dict[str, Any],
    timeouts: Dict[str, Any],
    db_path: Path,
    add_history_func: Callable, # 非同期履歴追加関数
    evaluation_results_json: str,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """評価結果を分析するジョブの実行関数 (非同期)"""
    logger.info(f"[Job {job_id}] 評価結果を分析中... Session: {session_id}")
    try:
        loop = asyncio.get_running_loop()
        try:
             eval_results = await loop.run_in_executor(None, json.loads, evaluation_results_json)
        except json.JSONDecodeError as e:
             raise ValueError("評価結果のJSON形式が不正です。") from e

        # プロンプト文字列のインデントを修正
        prompt = f"""\
Analyze the following evaluation results and provide insights for improvement:

```json
{json.dumps(eval_results, indent=2)}
```

Suggest specific areas or parameters to focus on next. Return the analysis as a JSON object with keys like 'strengths', 'weaknesses', 'suggestions'.
"""

        llm_client = get_llm_client(llm_config, timeouts)
        analysis_result = await llm_client.generate(prompt, request_json=True) # JSON形式を要求

        if not isinstance(analysis_result, dict) or "error" in analysis_result:
            error_msg = f"評価分析失敗。LLM応答: {analysis_result}"
            logger.error(f"[Job {job_id}] {error_msg}")
            if session_id:
                 try: await add_history_func(db_path, session_id, "analysis_failed", {"error": error_msg})
                 except Exception: pass
            raise LLMError(error_msg)

        logger.info(f"[Job {job_id}] 評価分析成功。")
        if session_id:
             try: await add_history_func(db_path, session_id, "analysis_complete", {"analysis": analysis_result, "job_id": job_id})
             except Exception as hist_e: logger.warning(f"[Job {job_id}] 分析履歴の追加に失敗: {hist_e}")

        return analysis_result # 分析結果の辞書を返す

    except Exception as e:
        logger.error(f"[Job {job_id}] _run_analyze_evaluation_results でエラー: {e}", exc_info=True)
        if session_id:
            try: await add_history_func(db_path, session_id, "analysis_failed", {"error": str(e)})
            except Exception: pass
        raise

async def _run_generate_hypotheses(
    job_id: str,
    llm_config: Dict[str, Any],
    timeouts: Dict[str, Any],
    db_path: Path,
    add_history_func: Callable, # 非同期履歴追加関数
    context: str = "", # context はプロンプト作成用
    session_id: str = ""
) -> Dict[str, Any]:
    """評価結果に基づいて改善仮説を生成する (非同期)"""
    logger.info(f"[Job {job_id}] 改善仮説を生成中... Session: {session_id}")
    try:
        # プロンプト文字列のインデントを修正
        prompt = f"""\
Based on the recent context of an algorithm improvement session, generate 3 diverse scientific hypotheses about potential improvements or reasons for observed performance.

Context:
{context}

Each hypothesis should be a JSON object with 'title', 'description', 'evidence' (from context), and 'testability'. Return the hypotheses as a JSON list under the key 'hypotheses'.
"""

        llm_client = get_llm_client(llm_config, timeouts)
        hypotheses_result = await llm_client.generate(prompt, request_json=True)

        if not isinstance(hypotheses_result, dict) or "hypotheses" not in hypotheses_result or not isinstance(hypotheses_result["hypotheses"], list):
            error_msg = f"LLMが期待するJSON形式で仮説を返しませんでした: {hypotheses_result}"
            logger.error(f"[Job {job_id}] {error_msg}")
            if session_id:
                try: await add_history_func(db_path, session_id, "generate_hypotheses_failed", {"error": error_msg, "context": context})
                except Exception: pass
            return {"error": error_msg}

        logger.info(f"[Job {job_id}] {len(hypotheses_result['hypotheses'])}個の仮説を生成しました。")
        if session_id:
            try: await add_history_func(db_path, session_id, "generate_hypotheses_result", {"hypotheses": hypotheses_result["hypotheses"], "job_id": job_id})
            except Exception as hist_e: logger.warning(f"[Job {job_id}] 仮説履歴の追加に失敗: {hist_e}")

        return hypotheses_result # 仮説リストを含む辞書を返す

    except Exception as e:
        logger.error(f"[Job {job_id}] _run_generate_hypotheses でエラー: {e}", exc_info=True)
        if session_id:
            try: await add_history_func(db_path, session_id, "generate_hypotheses_failed", {"error": str(e), "context": context})
            except Exception: pass
        raise

async def _run_analyze_code_segment(
    job_id: str,
    llm_config: Dict[str, Any],
    timeouts: Dict[str, Any],
    db_path: Path,
    add_history_func: Callable, # 非同期履歴追加関数
    code_segment: str,
    question: str = "",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """コードセグメントを分析するジョブの実行関数 (非同期)"""
    logger.info(f"[Job {job_id}] コードセグメントを分析中... Session: {session_id}")
    try:
        if not question: question = "Explain this code segment and suggest potential improvements."
        # プロンプト文字列のインデントを修正
        prompt = f"""\
Analyze the following code segment:

```python
{code_segment}
```

Question: {question}
"""

        llm_client = get_llm_client(llm_config, timeouts)
        analysis_text = await llm_client.generate(prompt, request_json=False) # テキスト応答を期待
        result = {"analysis": analysis_text}

        if session_id:
             try: await add_history_func(db_path, session_id, "code_analysis_complete", {"question": question, "analysis_length": len(analysis_text), "job_id": job_id})
             except Exception as hist_e: logger.warning(f"[Job {job_id}] コード分析履歴の追加に失敗: {hist_e}")

        return result

    except Exception as e:
        logger.error(f"[Job {job_id}] _run_analyze_code_segment でエラー: {e}", exc_info=True)
        if session_id:
            try: await add_history_func(db_path, session_id, "code_analysis_failed", {"error": str(e)})
            except Exception: pass
        raise # Re-raise the exception

async def _run_suggest_parameters(
    job_id: str,
    llm_config: Dict[str, Any],
    timeouts: Dict[str, Any],
    db_path: Path,
    add_history_func: Callable, # 非同期履歴追加関数
    source_code: str,
    context: str = "",
    session_id: str = ""
) -> Dict[str, Any]:
    """ソースコードに基づいて最適化対象のパラメータを提案する (非同期)"""
    logger.info(f"[Job {job_id}] パラメータ最適化候補を提案中... Session: {session_id}")
    try:
        # プロンプト文字列のインデントを修正
        prompt = f"""\
Analyze the source code below and suggest 3-5 numerical parameters suitable for optimization (e.g., thresholds, coefficients). For each parameter, provide its current value (if discernible), a suggested range [min, max], and an optional step value. Output MUST be a JSON object with a key 'params' containing a list of objects, each with keys 'name', 'current_value', 'suggested_range', and optionally 'step'.

Source Code:
```python
{source_code}
```

Context: {context}
"""

        llm_client = get_llm_client(llm_config, timeouts)
        suggestions_result = await llm_client.generate(prompt, request_json=True)

        if not isinstance(suggestions_result, dict) or "params" not in suggestions_result or not isinstance(suggestions_result["params"], list):
            error_msg = f"LLMが期待するJSON形式でパラメータ候補を返しませんでした: {suggestions_result}"
            logger.error(f"[Job {job_id}] {error_msg}")
            if session_id:
                try: await add_history_func(db_path, session_id, "suggest_parameters_failed", {"error": error_msg, "context": context})
                except Exception: pass
            return {"error": error_msg}

        logger.info(f"[Job {job_id}] {len(suggestions_result['params'])}個のパラメータ候補を提案しました。")
        if session_id:
            try: await add_history_func(db_path, session_id, "suggest_parameters_result", {"params": suggestions_result["params"], "job_id": job_id})
            except Exception as hist_e: logger.warning(f"[Job {job_id}] パラメータ提案履歴の追加に失敗: {hist_e}")

        return suggestions_result # パラメータリストを含む辞書を返す

    except Exception as e:
        logger.error(f"[Job {job_id}] _run_suggest_parameters でエラー: {e}", exc_info=True)
        if session_id:
            try: await add_history_func(db_path, session_id, "suggest_parameters_failed", {"error": str(e), "context": context})
            except Exception: pass
        raise

# --- Tool Registration --- #

def register_llm_tools(
    mcp: FastMCP,
    config: Dict[str, Any],
    start_async_job_func: Callable[..., str],
    db_path: Path, # DBパスを直接受け取るように変更
    add_history_func: Callable # 非同期履歴追加関数も受け取る
):
    """LLM関連のMCPツールを登録"""
    logger.info("LLMツールを登録中...")

    # Helper to start async job with necessary context
    def _start_llm_job(task_func: Coroutine, tool_name: str, session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Extract necessary config parts for the LLM task
        llm_config = config.get('llm', {})
        timeouts = config.get('timeouts', {})

        # Pass specific config parts, db_path, and history func
        job_id = start_async_job_func(
            task_func, # The async task function itself
            tool_name=tool_name, # Tool name for logging/DB
            session_id_for_job=session_id, # Pass session_id for job tracking if needed
            # Arguments specific to the task function:
            llm_config=llm_config, # Pass specific config parts
            timeouts=timeouts,
            db_path=db_path,
            add_history_func=add_history_func, # Pass history func
            **kwargs # Original tool arguments
        )

        # Add request history asynchronously (fire-and-forget)
        if session_id:
            asyncio.create_task(
                add_history_func( # Use the imported async function
                    # config 引数は不要
                    db_path=db_path,
                    session_id=session_id,
                    event_type=f"{tool_name}_request",
                    event_data={"params": kwargs, "job_id": job_id}
                )
            )
        return {"job_id": job_id, "status": "pending"}

    # --- Register Tools ---
    # 各ツールは _start_llm_job を呼び出し、対応する _run_... 非同期タスク関数を渡す

    @mcp.tool("improve_code")
    async def improve_code_tool(prompt: str, session_id: str = "") -> Dict[str, Any]:
        """LLMを使用してコードを改善します。"""
        task_kwargs = {k: v for k, v in locals().items() if k not in ['self', 'session_id']}
        return _start_llm_job(_run_improve_code, "improve_code", session_id, **task_kwargs)

    @mcp.tool("analyze_evaluation_results")
    async def analyze_evaluation_results_tool(evaluation_results_json: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """評価結果JSONを分析し、改善のための洞察を提供します。"""
        task_kwargs = {k: v for k, v in locals().items() if k not in ['self', 'session_id']}
        return _start_llm_job(_run_analyze_evaluation_results, "analyze_evaluation_results", session_id, **task_kwargs)

    @mcp.tool("generate_hypotheses")
    async def generate_hypotheses_tool(context: str = "", session_id: str = "") -> Dict[str, Any]:
        """セッションコンテキストに基づいて改善仮説を生成します。"""
        if not session_id: raise ValueError("generate_hypothesesにはsession_idが必要です。")
        task_kwargs = {k: v for k, v in locals().items() if k not in ['self', 'session_id']}
        return _start_llm_job(_run_generate_hypotheses, "generate_hypotheses", session_id, **task_kwargs)

    @mcp.tool("analyze_code_segment")
    async def analyze_code_segment_tool(code_segment: str, question: str = "", session_id: Optional[str] = None) -> Dict[str, Any]:
        """指定されたコードセグメントを分析します。"""
        task_kwargs = {k: v for k, v in locals().items() if k not in ['self', 'session_id']}
        return _start_llm_job(_run_analyze_code_segment, "analyze_code_segment", session_id, **task_kwargs)

    @mcp.tool("suggest_parameters")
    async def suggest_parameters_tool(source_code: str, context: str = "", session_id: str = "") -> Dict[str, Any]:
        """ソースコードに基づいて最適化対象のパラメータを提案します。"""
        task_kwargs = {k: v for k, v in locals().items() if k not in ['self', 'session_id']}
        return _start_llm_job(_run_suggest_parameters, "suggest_parameters", session_id, **task_kwargs)

    logger.info("LLMツールを登録しました。")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.")
