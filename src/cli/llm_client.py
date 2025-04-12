"""LLMクライアントモジュール

このモジュールはCLIからLLMとの通信を管理し、MCPサーバーから生成されたプロンプトを使って
LLMによるコード改善やパラメータ提案などを行います。
"""

import os
import json
import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Anthropicクライアント
from anthropic import AsyncAnthropic, AnthropicError

# 例外ハンドリング
class LLMClientError(Exception):
    """LLMクライアントに関連するエラー"""
    def __init__(self, message: str, error_type: str = "ClientError", original_error: Optional[Exception] = None):
        self.message = message
        self.error_type = error_type
        self.original_error = original_error
        super().__init__(message)

class AnthropicClient:
    """AnthropicモデルとのAPIインターフェース"""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "claude-3-opus-20240229",
                 max_tokens: int = 4096,
                 temperature: float = 0.2,
                 timeout: int = 180):
        """初期化

        Args:
            api_key: Anthropic API キー（Noneの場合は環境変数から取得）
            model: 使用するモデル名
            max_tokens: 応答の最大トークン数
            temperature: 生成の温度（低いほど決定的、高いほど多様）
            timeout: API呼び出しのタイムアウト秒数
        """
        self.logger = logging.getLogger(__name__)
        
        # APIキーの取得（優先順位: 引数 > 環境変数）
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self.logger.error("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
            raise LLMClientError("API key is required for Anthropic client", error_type="ConfigurationError")
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # AsyncAnthropic クライアントの初期化
        self.client = AsyncAnthropic(api_key=api_key)
        self.logger.info(f"Initialized Anthropic client with model: {model}")
        
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      request_json: bool = False,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None) -> str:
        """LLMにプロンプトを送信して応答を生成

        Args:
            prompt: LLMに送信するプロンプト
            system_prompt: システムプロンプト（指示）
            request_json: JSONレスポースをリクエストする場合はTrue
            temperature: 生成温度（省略時はクラス初期化値）
            max_tokens: 最大トークン（省略時はクラス初期化値）

        Returns:
            生成されたテキスト応答 (request_json=True の場合はJSON文字列、失敗時は生テキスト)
        """
        self.logger.info(f"Sending prompt to {self.model} (length: {len(prompt)} chars)")
        
        # JSONレスポンスをリクエストする場合、システムプロンプトに含める
        effective_system_prompt = system_prompt or ""
        if request_json:
            if effective_system_prompt:
                effective_system_prompt += "\n\n"
            # JSONオブジェクトのみを返すように指示を明確化
            effective_system_prompt += "Format your entire response as a single valid JSON object. Do not include any text outside of the JSON object, including markdown formatting like ```json ... ```."
        
        try:
            start_time = time.time()
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                system=effective_system_prompt if effective_system_prompt else None, # 空文字列ではなくNoneを渡す
                messages=[
                    {"role": "user", "content": prompt}
                ],
                timeout=self.timeout # timeout を適用
            )
            elapsed_time = time.time() - start_time
            
            self.logger.info(f"Received response from {self.model} in {elapsed_time:.2f}s")
            content = response.content[0].text
            
            # JSONレスポンスをリクエストした場合、パースを試みる
            if request_json:
                json_str = None
                try:
                    # 1. ```json ... ``` 形式の抽出
                    import re # re をインポート
                    match_json_block = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
                    if match_json_block:
                        json_str = match_json_block.group(1).strip()
                        self.logger.debug("Extracted JSON string from ```json block.")
                    else:
                        # 2. テキスト全体から最初の '{' と最後の '}' を探す
                        start_index = content.find("{")
                        end_index = content.rfind("}")
                        if start_index != -1 and end_index != -1 and start_index < end_index:
                            json_str = content[start_index : end_index + 1]
                            self.logger.debug("Extracted JSON string by finding first '{' and last '}'.")
                        else:
                            self.logger.warning("Could not find JSON object structure ('{'...'}' or ```json ... ```) in the response.")
                            json_str = content # 抽出できない場合は生テキストを試す
                    
                    # 抽出した文字列をJSONとしてパース
                    parsed_json = json.loads(json_str)
                    # 有効なJSONであれば、整形されたJSON文字列を返す
                    return json.dumps(parsed_json, ensure_ascii=False, indent=2)
                
                except json.JSONDecodeError as parse_err:
                    # パース失敗時のログを詳細化
                    self.logger.warning(
                        f"Failed to parse JSON from response. Error: {parse_err}. "
                        f"Attempted string: '{json_str if json_str else content[:100]}'..." # 失敗した文字列の一部を表示
                    )
                    self.logger.warning("Returning raw text response instead.")
                    return content # パース失敗時は生テキストを返す
                except Exception as e: # その他の予期せぬエラー
                    self.logger.error(f"Unexpected error during JSON processing: {e}", exc_info=True)
                    return content # エラー時も生テキストを返す
            
            return content
            
        except AnthropicError as api_err:
            error_msg = f"Anthropic API error: {api_err}"
            self.logger.error(error_msg)
            raise LLMClientError(error_msg, error_type="APIError", original_error=api_err)
        except asyncio.TimeoutError as timeout_err: # asyncio.TimeoutError をキャッチ
            error_msg = f"Timeout error when calling Anthropic API (timeout: {self.timeout}s)"
            self.logger.error(error_msg)
            raise LLMClientError(error_msg, error_type="TimeoutError", original_error=timeout_err)
        except Exception as e:
            error_msg = f"Unexpected error when calling Anthropic API: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise LLMClientError(error_msg, error_type="UnexpectedError", original_error=e)
    
    async def extract_code_from_text(self, text: str) -> Optional[str]:
        """テキスト応答からコードブロックを抽出します

        Args:
            text: LLMからの応答テキスト

        Returns:
            抽出されたコード、抽出できない場合はNone
        """
        # Pythonコードブロックを探す (```python や ``` で囲まれたもの)
        import re
        
        # パターン1: ```python ... ``` 形式
        python_block_pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(python_block_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # パターン2: ``` ... ``` 形式（言語指定なし）
        generic_block_pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(generic_block_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # コードブロックが見つからない場合
        self.logger.warning("Could not extract code from LLM response")
        return None

# クライアント初期化ヘルパー
def initialize_llm_client(api_key: Optional[str] = None, model: str = "claude-3-opus-20240229") -> AnthropicClient:
    """LLMクライアントを初期化して返す

    Args:
        api_key: APIキー（Noneの場合は環境変数から取得）
        model: 使用するモデル名

    Returns:
        初期化されたLLMクライアント
    """
    return AnthropicClient(api_key=api_key, model=model)

# async def test_llm_client(): # テスト関数はテストファイルに移行
#     \"\"\"LLMクライアントの簡単なテスト関数\"\"\"
#     import asyncio
#     
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     
#     # 環境変数からAPIキーを取得
#     api_key = os.environ.get("ANTHROPIC_API_KEY")
#     
#     client = initialize_llm_client(api_key)
#     
#     try:
#         # 簡単なテストプロンプト
#         prompt = "短いpythonプログラムを書いて、Hello Worldと表示してください。"
#         
#         response = await client.generate(prompt)
#         print(f"Response:\n{response}")
#         
#         code = await client.extract_code_from_text(response)
#         if code:
#             print(f"\nExtracted code:\n{code}")
#         
#     except LLMClientError as e:
#         logger.error(f"Error: {e}")
#         if e.original_error:
#             logger.error(f"Original error: {e.original_error}")
# 
# if __name__ == "__main__": # テスト関数はテストファイルに移行
#     # テスト実行
#     asyncio.run(test_llm_client()) 