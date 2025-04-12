#!/usr/bin/env python
"""LLMクライアントのテスト用スクリプト"""

import pytest
import asyncio
import logging
import os
import sys
import json
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock # patch は不要

# このスクリプトの親ディレクトリをPYTHONPATHに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- テスト対象モジュールのインポート ---
# 必要なモジュールを直接インポート (テスト環境での存在を前提とする)
try:
    from src.cli.llm_client import initialize_llm_client, LLMClientError, AnthropicClient
    from anthropic import AsyncAnthropic, AnthropicError # 実際のクラスをインポート
except ImportError as e:
    pytest.skip(f"Skipping LLM client tests due to import error: {e}", allow_module_level=True)


# --- テスト用のモックレスポンスクラス ---
class MockResponse:
    def __init__(self, text_content):
        self._content = [MockContent(text_content)]
    @property
    def content(self): return self._content

class MockContent:
    def __init__(self, text_content): self._text = text_content
    @property
    def text(self): return self._text


# @pytest.mark.skipif(SKIP_TESTS, reason="必要なモジュールが見つかりません") # pytest.skip で代替
class TestLLMClient:

    # TL-01: APIキーがない場合の初期化エラー
    def test_initialize_llm_client_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(LLMClientError) as exc_info:
            initialize_llm_client(api_key=None)
        assert "API key is required" in str(exc_info.value)
        assert exc_info.value.error_type == "ConfigurationError"

    def test_initialize_llm_client_with_key(self, mocker):
        """APIキーを指定してクライアントが初期化されることをテスト"""
        # llm_client モジュール内の AsyncAnthropic をモック
        mock_async_anthropic_cls = mocker.patch('src.cli.llm_client.AsyncAnthropic')
        api_key = "test_api_key_arg"
        client = initialize_llm_client(api_key=api_key)

        # initialize_llm_client 内で AsyncAnthropic が呼び出されたか確認
        mock_async_anthropic_cls.assert_called_once_with(api_key=api_key)
        assert isinstance(client, AnthropicClient)
        # client.client がモックインスタンスであることを確認
        assert client.client == mock_async_anthropic_cls.return_value

    @pytest.fixture
    def mock_anthropic_create(self, mocker):
         """AsyncAnthropic クラスとそのインスタンスの messages.create をモックする"""
         # AsyncAnthropic クラス自体をモック
         mock_async_anthropic_cls = mocker.patch('src.cli.llm_client.AsyncAnthropic')
         # AsyncAnthropic クラスが呼ばれたときに返すインスタンスモックを取得
         mock_async_client_instance = mock_async_anthropic_cls.return_value
         # インスタンスモックの messages.create 属性に AsyncMock を設定
         mock_create = AsyncMock(return_value=MockResponse("Default Mock Response"))
         mock_async_client_instance.messages.create = mock_create
         return mock_create # この AsyncMock を返す

    @pytest.mark.asyncio
    async def test_llm_client_generate_basic(self, mock_anthropic_create):
        # initialize_llm_client を呼ぶと、モックされた AsyncAnthropic が使われ、
        # そのインスタンス (client.client) の messages.create は AsyncMock になっているはず
        client = initialize_llm_client(api_key="dummy_key")
        mock_anthropic_create.return_value = MockResponse("Test response text.")
        response = await client.generate("Test prompt")
        assert response == "Test response text."
        # await された AsyncMock の呼び出しを確認
        mock_anthropic_create.assert_awaited_once()
        call_args = mock_anthropic_create.await_args[1]
        assert call_args["messages"][0]["content"] == "Test prompt"
        assert call_args["model"] == client.model

    # TL-06: 温度/トークン数 上書き
    @pytest.mark.asyncio
    async def test_llm_client_generate_override_params(self, mock_anthropic_create):
        client = initialize_llm_client(api_key="dummy_key")
        mock_anthropic_create.return_value = MockResponse("Overridden response.")
        await client.generate("Test prompt", temperature=0.9, max_tokens=100)

        mock_anthropic_create.assert_awaited_once()
        call_args = mock_anthropic_create.await_args[1]
        assert call_args["temperature"] == 0.9
        assert call_args["max_tokens"] == 100

    # --- extract_code tests (変更なし) ---
    @pytest.mark.asyncio
    async def test_llm_client_extract_code_python(self):
        client = initialize_llm_client(api_key="dummy_key")
        text = "```python\nprint('hello')\n```"
        code = await client.extract_code_from_text(text)
        assert code == "print('hello')"

    @pytest.mark.asyncio
    async def test_llm_client_extract_code_generic(self):
        client = initialize_llm_client(api_key="dummy_key")
        text = "Some text\n```\ndef test():\n  pass\n```\nMore text"
        code = await client.extract_code_from_text(text)
        assert code == "def test():\n  pass"

    @pytest.mark.asyncio
    async def test_llm_client_extract_code_not_found(self, caplog):
        client = initialize_llm_client(api_key="dummy_key")
        text = "This text does not contain any code blocks."
        with caplog.at_level(logging.WARNING):
             code = await client.extract_code_from_text(text)
        assert code is None
        assert "Could not extract code" in caplog.text
    # --- End of extract_code tests ---

    @pytest.mark.asyncio
    async def test_llm_client_generate_json_request_success(self, mock_anthropic_create):
        mock_response_text = '{\n  "key": "value",\n  "number": 123\n}'
        mock_anthropic_create.return_value = MockResponse(mock_response_text)
        client = initialize_llm_client(api_key="dummy_key")
        response = await client.generate("Generate JSON", request_json=True)

        expected_json = json.dumps({"key": "value", "number": 123}, ensure_ascii=False, indent=2)
        assert response == expected_json

        mock_anthropic_create.assert_awaited_once()
        call_args = mock_anthropic_create.await_args[1]
        system_prompt = call_args.get("system", "")
        assert "JSON object" in system_prompt

    # TL-03: JSON要求時にパース失敗した場合
    @pytest.mark.asyncio
    async def test_llm_client_generate_json_request_parse_fail(self, mock_anthropic_create, caplog):
        invalid_json_text = "This is not JSON, but ```json {invalid json ```"
        mock_anthropic_create.return_value = MockResponse(invalid_json_text)
        client = initialize_llm_client(api_key="dummy_key")

        with caplog.at_level(logging.WARNING):
            response = await client.generate("Generate JSON", request_json=True)

        assert response == invalid_json_text
        assert "Failed to parse JSON" in caplog.text
        assert "Returning raw text response instead" in caplog.text

    @pytest.mark.asyncio
    async def test_llm_client_generate_api_error(self, mock_anthropic_create):
        """API呼び出し時にAnthropicErrorが発生するテスト"""
        # generate 内で捕捉される AnthropicError を発生させる
        # 実際の AnthropicError (anthropic ライブラリからインポート) を使用
        api_error_instance = AnthropicError("Invalid API Key")
        mock_anthropic_create.side_effect = api_error_instance
        client = initialize_llm_client(api_key="dummy_key")

        with pytest.raises(LLMClientError) as exc_info:
            await client.generate("Test prompt")

        assert "Anthropic API error" in str(exc_info.value)
        assert exc_info.value.error_type == "APIError"
        # except 節で捕捉したエラーが original_error に入ることを確認
        assert exc_info.value.original_error is api_error_instance

    # TL-02: API呼び出しタイムアウト
    @pytest.mark.asyncio
    async def test_llm_client_generate_timeout_error(self, mock_anthropic_create):
        """API呼び出し時にTimeoutErrorが発生するテスト"""
        # generate 内で捕捉される asyncio.TimeoutError を発生させる
        timeout_error = asyncio.TimeoutError("API call timed out")
        mock_anthropic_create.side_effect = timeout_error
        client = initialize_llm_client(api_key="dummy_key")
        client.timeout = 1 # タイムアウト値は generate 内の exception handling で使われる

        with pytest.raises(LLMClientError) as exc_info:
            await client.generate("Test prompt for timeout")

        assert "Timeout error when calling Anthropic API" in str(exc_info.value)
        assert exc_info.value.error_type == "TimeoutError"
        assert exc_info.value.original_error is timeout_error

    @pytest.mark.asyncio
    async def test_llm_client_generate_unexpected_error(self, mock_anthropic_create):
        """API呼び出し時に予期せぬエラーが発生するテスト"""
        # generate 内で捕捉される最後の Exception を発生させる
        unexpected_error = ValueError("Something unexpected happened")
        mock_anthropic_create.side_effect = unexpected_error
        client = initialize_llm_client(api_key="dummy_key")

        with pytest.raises(LLMClientError) as exc_info:
            await client.generate("Test prompt")

        assert "Unexpected error when calling Anthropic API" in str(exc_info.value)
        assert exc_info.value.error_type == "UnexpectedError"
        assert exc_info.value.original_error is unexpected_error 