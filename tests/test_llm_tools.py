import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
import httpx

# Target module
from mcp_server_logic import llm_tools
from src.utils.exception_utils import LLMError

# --- Mock Configs --- #

@pytest.fixture
def mock_llm_config():
    return {
        "client_type": "ClaudeClient",
        "api_key": "TEST_API_KEY",
        "model": "claude-test-model",
        "api_base": "https://test.anthropic.com",
        "api_version": "2023-06-01",
        "max_tokens": 1024
    }

@pytest.fixture
def mock_timeouts():
    return {"llm": 30}

# --- Test BaseLLMClient (Abstract class, mostly for type checking) --- #
# No direct tests needed for the abstract class itself.

# --- Test ClaudeClient --- #

def test_claude_client_init(mock_llm_config, mock_timeouts):
    """Test ClaudeClient initialization with config."""
    client = llm_tools.ClaudeClient(mock_llm_config, mock_timeouts)
    assert client.api_key == "TEST_API_KEY"
    assert client.model == "claude-test-model"
    assert client.api_base == "https://test.anthropic.com"
    assert client.api_version == "2023-06-01"
    assert client.timeout == 30
    assert client.max_tokens == 1024
    assert not client.use_mock

@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ENV_KEY", "CLAUDE_MODEL": "env-model"})
def test_claude_client_init_env_override(mock_timeouts):
    """Test ClaudeClient initialization prioritizing env vars if config is missing."""
    # Simulate missing config values
    llm_config_empty = {"client_type": "ClaudeClient"}
    client = llm_tools.ClaudeClient(llm_config_empty, mock_timeouts)
    assert client.api_key == "ENV_KEY"
    assert client.model == "env-model"

def test_claude_client_init_no_key_mock(mock_timeouts):
    """Test ClaudeClient falls back to mock if no API key."""
    llm_config_no_key = {"client_type": "ClaudeClient"}
    with patch.dict(os.environ, {}, clear=True): # Ensure no env var key
        client = llm_tools.ClaudeClient(llm_config_no_key, mock_timeouts)
        assert client.use_mock

@pytest.mark.asyncio
async def test_claude_client_generate_mock(mock_llm_config, mock_timeouts):
    """Test generate uses mock response when use_mock is True."""
    client = llm_tools.ClaudeClient(mock_llm_config, mock_timeouts)
    client.use_mock = True # Force mock

    with patch("mcp_server_logic.llm_tools.mock_llm_response", return_value="Mocked response") as mock_func:
        result = await client.generate("Test prompt")
        assert result == "Mocked response"
        mock_func.assert_called_once_with("Test prompt", False)

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock) # Mock the httpx post call
async def test_claude_client_generate_success_text(mock_post, mock_llm_config, mock_timeouts):
    """Test successful text generation via Claude API."""
    client = llm_tools.ClaudeClient(mock_llm_config, mock_timeouts)
    client.use_mock = False

    # Mock the API response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "content": [{"type": "text", "text": "Generated text content"}]
    }
    mock_post.return_value = mock_response

    prompt = "Generate some text."
    result = await client.generate(prompt, request_json=False)

    assert result == "Generated text content"
    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    assert call_kwargs['url'] == f"{client.api_base}/v1/messages"
    assert call_kwargs['headers']["X-API-Key"] == client.api_key
    assert call_kwargs['json']["model"] == client.model
    assert call_kwargs['json']["messages"][0]["content"] == prompt
    assert "system" not in call_kwargs['json'] # No system prompt for text

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_claude_client_generate_success_json(mock_post, mock_llm_config, mock_timeouts):
    """Test successful JSON generation via Claude API."""
    client = llm_tools.ClaudeClient(mock_llm_config, mock_timeouts)
    client.use_mock = False
    expected_json = {"key": "value", "items": [1, 2]}

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    # Return the JSON string within the text field
    mock_response.json.return_value = {
        "content": [{"type": "text", "text": json.dumps(expected_json)}]
    }
    mock_post.return_value = mock_response

    prompt = "Generate JSON."
    result = await client.generate(prompt, request_json=True)

    assert result == expected_json
    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    assert call_kwargs['json']["system"] is not None # System prompt requested for JSON
    assert call_kwargs['json']["messages"][0]["content"] == prompt

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_claude_client_generate_json_fallback(mock_post, mock_llm_config, mock_timeouts):
    """Test JSON fallback to code extraction when JSON parse fails."""
    client = llm_tools.ClaudeClient(mock_llm_config, mock_timeouts)
    client.use_mock = False
    raw_text_with_code = "Here is the code:\n```python\ndef hello():\n  print('Hello')\n```\nThis should work." # Invalid JSON

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"content": [{"type": "text", "text": raw_text_with_code}]}
    mock_post.return_value = mock_response

    prompt = "Generate JSON, but fail."
    result = await client.generate(prompt, request_json=True)

    assert isinstance(result, dict)
    assert result["error"] == "JSON parse failed, extracted code as fallback."
    assert result["improved_code"] == "def hello():\n  print('Hello')"

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_claude_client_generate_api_error(mock_post, mock_llm_config, mock_timeouts):
    """Test handling of API errors (e.g., 4xx, 5xx)."""
    client = llm_tools.ClaudeClient(mock_llm_config, mock_timeouts)
    client.use_mock = False

    # Simulate an API error response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.text = "Bad Request Error Details"
    # Make raise_for_status raise the error
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request", request=MagicMock(), response=mock_response
    )
    mock_post.return_value = mock_response

    with pytest.raises(LLMError, match="Claude API request error: Bad Request"):
        await client.generate("Test prompt")

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Request timed out"))
async def test_claude_client_generate_timeout(mock_post, mock_llm_config, mock_timeouts):
    """Test handling of timeout errors."""
    client = llm_tools.ClaudeClient(mock_llm_config, mock_timeouts)
    client.use_mock = False

    with pytest.raises(LLMError, match="Claude API call timed out"):
        await client.generate("Test prompt")

# --- Test get_llm_client Factory --- #

@patch("mcp_server_logic.llm_tools.ClaudeClient")
def test_get_llm_client_claude(MockClaudeClient, mock_llm_config, mock_timeouts):
    """Test get_llm_client returns ClaudeClient instance."""
    llm_tools._llm_client_instance = None # Reset singleton
    client = llm_tools.get_llm_client(mock_llm_config, mock_timeouts)
    MockClaudeClient.assert_called_once_with(mock_llm_config, mock_timeouts)
    assert isinstance(client, MagicMock) # It returns the mocked instance

@patch("mcp_server_logic.llm_tools.ClaudeClient")
def test_get_llm_client_singleton(MockClaudeClient, mock_llm_config, mock_timeouts):
    """Test get_llm_client returns the same instance (singleton)."""
    llm_tools._llm_client_instance = None # Reset singleton
    client1 = llm_tools.get_llm_client(mock_llm_config, mock_timeouts)
    client2 = llm_tools.get_llm_client(mock_llm_config, mock_timeouts)
    MockClaudeClient.assert_called_once() # Should only be called once
    assert client1 is client2

# --- Test Helper Functions --- #

def test_extract_code_from_text():
    text1 = "Some text ```python\ndef hello():\n  pass\n``` more text"
    assert llm_tools.extract_code_from_text(text1) == "def hello():\n  pass"

    text2 = "```\nJust code here\n```"
    assert llm_tools.extract_code_from_text(text2) == "Just code here"

    text3 = "No code block here."
    assert llm_tools.extract_code_from_text(text3) is None

    text4 = "Code: `print('inline')`" # Does not extract inline code
    assert llm_tools.extract_code_from_text(text4) is None

# TODO: Add tests for mock_llm_response if needed
# TODO: Add tests for _run_improve_code, _run_generate_hypotheses, etc.
# These will require more mocking (db, session history, file system maybe) 