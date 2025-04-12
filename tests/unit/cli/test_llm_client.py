#!/usr/bin/env python
"""LLMクライアントのテスト用スクリプト"""

import pytest
import asyncio
import logging
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock, Mock, PropertyMock

# このスクリプトの親ディレクトリをPYTHONPATHに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 実テストに先立ち、必要なモジュールをモックする
with patch.dict('sys.modules', {
    'anthropic': MagicMock(),
    'anthropic.AsyncAnthropic': MagicMock(),
    'anthropic.AnthropicError': type('AnthropicError', (Exception,), {})
}):
    try:
        # モック環境下でのインポート
        from src.cli.llm_client import initialize_llm_client, LLMClientError, AnthropicClient
        SKIP_TESTS = False
    except ImportError as e:
        print(f"Skipping LLM client tests due to import error: {e}")
        SKIP_TESTS = True

# テスト対象のクラスのダミー定義（スキップフラグが有効な場合）
if SKIP_TESTS:
    class LLMClientError(Exception): pass
    class AnthropicClient: pass
    def initialize_llm_client(*args, **kwargs): pass

# テスト用のモックレスポンスクラス
class MockResponse:
    """Anthropic APIレスポンスのモッククラス"""
    def __init__(self, text_content):
        self._content = [MockContent(text_content)]
    
    @property
    def content(self):
        return self._content

class MockContent:
    """Anthropic APIコンテンツのモッククラス"""
    def __init__(self, text_content):
        self._text = text_content
    
    @property
    def text(self):
        return self._text

@pytest.mark.skipif(SKIP_TESTS, reason="必要なモジュールが見つかりません")
class TestLLMClient:
    """LLMクライアントのテストクラス"""
    
    @pytest.fixture
    def mock_anthropic(self):
        """完全にモック化されたAnthropicクライアント"""
        # モックオブジェクトの作成
        mock_anthropic_module = MagicMock()
        mock_async_client = MagicMock()
        mock_messages = MagicMock()
        mock_messages.create = AsyncMock()
        
        # プロパティアクセスに対応するモックレスポンスを設定
        mock_messages.create.return_value = MockResponse("これはテスト応答です。\n```python\nprint('Hello, World!')\n```")
        
        mock_async_client.messages = mock_messages
        mock_anthropic_module.AsyncAnthropic.return_value = mock_async_client
        mock_anthropic_module.AnthropicError = type('AnthropicError', (Exception,), {})
        
        # 実際のインポートをモックに置き換え
        with patch.dict('sys.modules', {'anthropic': mock_anthropic_module}):
            yield mock_anthropic_module
    
    @pytest.mark.asyncio
    async def test_initialize_llm_client(self, mock_anthropic):
        """LLMクライアントの初期化をテストする"""
        # モックが設定された状態でimport
        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            from src.cli.llm_client import initialize_llm_client, AnthropicClient
            
            api_key = "test_api_key"
            client = initialize_llm_client(api_key)
            
            # Anthropicクライアントが正しく初期化されたことを確認
            mock_anthropic.AsyncAnthropic.assert_called_once_with(api_key=api_key)
            assert isinstance(client, AnthropicClient)
    
    @pytest.mark.asyncio
    async def test_llm_client_generate(self, mock_anthropic):
        """テキスト生成機能をテストする"""
        # モックが設定された状態でimport
        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            from src.cli.llm_client import initialize_llm_client
            
            # 標準的なテキスト応答を設定
            mock_anthropic.AsyncAnthropic.return_value.messages.create.return_value = MockResponse(
                "これはテスト応答です。\n```python\nprint('Hello, World!')\n```"
            )
            
            api_key = "test_api_key"
            client = initialize_llm_client(api_key)
            
            # 簡単なプロンプトでテスト
            prompt = "短いPythonプログラムを書いて、Hello Worldと表示してください。"
            response = await client.generate(prompt)
            
            # レスポンスが期待通りであることを確認
            assert "これはテスト応答です" in response
            assert "Hello, World" in response
            
            # APIが正しいパラメータで呼び出されたことを確認
            mock_anthropic.AsyncAnthropic.return_value.messages.create.assert_called_once()
            call_args = mock_anthropic.AsyncAnthropic.return_value.messages.create.call_args[1]
            assert call_args["messages"][0]["content"] == prompt
    
    @pytest.mark.asyncio
    async def test_llm_client_extract_code(self, mock_anthropic):
        """コード抽出機能をテストする"""
        # モックが設定された状態でimport
        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            from src.cli.llm_client import initialize_llm_client
            
            api_key = "test_api_key"
            client = initialize_llm_client(api_key)
            
            # コードブロックを含むテキスト
            text = """
            以下のコードを使用できます：
            
            ```python
            def hello():
                print("Hello, World!")
                
            hello()
            ```
            """
            
            code = await client.extract_code_from_text(text)
            
            # 正しくコードが抽出されていることを確認
            assert code is not None
            assert "def hello()" in code
            assert 'print("Hello, World!")' in code
    
    @pytest.mark.asyncio
    async def test_llm_client_json_request(self, mock_anthropic):
        """JSONリクエスト機能をテストする"""
        # モックが設定された状態でimport
        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            from src.cli.llm_client import initialize_llm_client
            
            # JSONレスポンス用にモックを設定
            mock_anthropic.AsyncAnthropic.return_value.messages.create.return_value = MockResponse(
                '{"name": "テスト太郎", "age": 30, "hobby": "プログラミング"}'
            )
            
            api_key = "test_api_key"
            client = initialize_llm_client(api_key)
            
            # JSONリクエストでテスト
            prompt = "名前、年齢、趣味を含む人物情報のJSONを生成してください"
            response = await client.generate(prompt, request_json=True)
            
            # レスポンスが期待通りであることを確認
            assert "name" in response
            assert "age" in response
            assert "hobby" in response
            
            # JSONリクエスト用のシステムメッセージが含まれていることを確認
            call_args = mock_anthropic.AsyncAnthropic.return_value.messages.create.call_args[1]
            system_message = call_args.get("system", "")
            assert "JSON" in system_message
    
    @pytest.mark.asyncio
    async def test_llm_client_error_handling(self, mock_anthropic):
        """エラーハンドリングをテストする"""
        # モックが設定された状態でimport
        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            from src.cli.llm_client import initialize_llm_client, LLMClientError
            
            # エラーを発生させるようにモックを設定
            api_error = mock_anthropic.AnthropicError("API key is invalid")
            mock_anthropic.AsyncAnthropic.return_value.messages.create.side_effect = api_error
            
            api_key = "test_api_key"
            client = initialize_llm_client(api_key)
            
            # エラーが適切にキャッチされ、LLMClientErrorとして再送出されることを確認
            with pytest.raises(LLMClientError) as exc_info:
                await client.generate("テストプロンプト")
            
            assert "API key is invalid" in str(exc_info.value) or "Anthropic API error" in str(exc_info.value) 