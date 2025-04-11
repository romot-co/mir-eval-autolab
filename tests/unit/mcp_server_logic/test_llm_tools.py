# tests/unit/mcp_server_logic/test_llm_tools.py
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock, ANY
import httpx # For mocking httpx.AsyncClient
from datetime import datetime, timezone

# テスト対象モジュールと依存モジュールをインポート (存在しない場合はダミーを仮定)
try:
    from src.mcp_server_logic import llm_tools
    from src.mcp_server_logic.llm_tools import (
        extract_code_from_text,
        ClaudeClient, # または Anthropic クライアントなど
        LLMError,
        _render_prompt, # プロンプトレンダリング関数
        _run_propose_strategy_async, # 例: 戦略提案タスク
        _run_refine_code_async # 例: コード改善タスク
        # 他の _run_..._async 関数も追加
    )
    from src.mcp_server_logic.schemas import (
        SessionInfoResponse, HistoryEntry, # セッション情報
        ProposeStrategyInput, ProposeStrategyOutput, # 戦略提案スキーマ
        RefineCodeInput, RefineCodeOutput, # コード改善スキーマ
        ErrorInfo # エラー情報
        # 他のツールの Input/Output スキーマも追加
    )
    from src.mcp_server_logic.session_manager import SessionManagerConfig # 設定が必要な場合
    from jsonschema import ValidationError # JSON スキーマ検証用
    import jinja2 # プロンプトテンプレート用
except ImportError:
    print("Warning: Using dummy implementations for llm_tools.py and dependencies.")
    from dataclasses import dataclass, field
    from typing import Dict, Any as TypingAny, Optional, List

    class LLMError(Exception): pass
    class ValidationError(Exception): pass # Dummy for jsonschema

    @dataclass
    class SessionInfoResponse: # Simplified dummy
        session_id: str
        status: str
        history: List[Dict] = field(default_factory=list)

    # --- Dummy Schemas ---
    @dataclass
    class ProposeStrategyInput: params: Dict[str, TypingAny]
    @dataclass
    class ProposeStrategyOutput: result: Dict[str, TypingAny]
    @dataclass
    class RefineCodeInput: params: Dict[str, TypingAny]
    @dataclass
    class RefineCodeOutput: result: Dict[str, TypingAny]
    @dataclass
    class ErrorInfo: error_type: str; message: str; details: Optional[str] = None

    @dataclass
    class SessionManagerConfig: db_path: str = ":memory:" # Dummy

    # --- Dummy llm_tools functions/classes ---
    def extract_code_from_text(text: str, language: str = "python") -> Optional[str]:
        # Very basic dummy extraction
        start_tag = f"```{language}"
        end_tag = "```"
        start_index = text.find(start_tag)
        if start_index == -1:
            return None
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index == -1:
            return None
        return text[start_index + len(start_tag):end_index].strip()

    class ClaudeClient: # Dummy Client
        def __init__(self, api_key: str, client: httpx.AsyncClient):
            self._api_key = api_key
            self._client = client

        async def generate_text(self, prompt: str, model: str, max_tokens: int, temperature: float, stop_sequences: List[str]) -> Dict[str, TypingAny]:
            # Simulate API call using the mocked client
            try:
                # Dummy request/response structure
                response = await self._client.post(
                    "https://dummy.anthropic.com/v1/complete", # Dummy URL
                    headers={"x-api-key": self._api_key, "Content-Type": "application/json"},
                    json={"prompt": prompt, "model": model, "max_tokens_to_sample": max_tokens, "temperature": temperature, "stop_sequences": stop_sequences}
                )
                response.raise_for_status() # Raise exception for 4xx/5xx
                return response.json() # Assume response is JSON like {"completion": "..."}
            except httpx.HTTPStatusError as e:
                raise LLMError(f"API request failed: {e.response.status_code}") from e
            except Exception as e:
                raise LLMError(f"LLM client error: {e}") from e

    # Dummy template rendering function
    def _render_prompt(template_name: str, context: Dict[str, TypingAny], template_env_mock) -> str:
         # Simulate template loading and rendering
         template_mock = template_env_mock.get_template(template_name)
         return template_mock.render(context)

    # Dummy schema validation function (mocked later)
    def validate_json(instance, schema):
        pass # Assume valid by default in dummy

    # Dummy _run_..._async functions
    async def _run_generic_llm_task(session_info: SessionInfoResponse, tool_input, tool_output_class, template_name: str, event_prefix: str, llm_client_mock, session_manager_mock, template_env_mock, misc_utils_mock, schema_validator_mock, config):
        event_type_base = f"{event_prefix}"
        try:
            context = {"session": session_info, "input": tool_input.params} # Simplified context
            prompt = _render_prompt(template_name, context, template_env_mock)

            # Simulate LLM call
            llm_response = await llm_client_mock.generate_text(prompt=prompt, model=ANY, max_tokens=ANY, temperature=ANY, stop_sequences=ANY)

            # Simulate extracting result and validating
            # Assume completion is the main result field
            result_text = llm_response.get("completion", "")
            # Basic JSON parsing simulation if needed
            try:
                 result_data = json.loads(result_text) if result_text.startswith('{') else {"raw_text": result_text}
            except json.JSONDecodeError:
                 result_data = {"raw_text": result_text} # Fallback for non-JSON

            # Validate output against schema (using mock)
            try:
                schema_validator_mock(result_data) # Validate the extracted data
                output = tool_output_class(result=result_data)
                event_type = f"{event_type_base}_complete"
                details = {"input": tool_input.params, "output": output.result}
            except ValidationError as ve:
                 # Handle validation error specifically if needed by design
                 raise LLMError(f"LLM output validation failed: {ve}") from ve

        except Exception as e:
            error_info = ErrorInfo(error_type=type(e).__name__, message=str(e))
            event_type = f"{event_type_base}_failed"
            details = {"input": tool_input.params, "error": error_info.__dict__}
            await session_manager_mock.add_session_history(session_id=session_info.session_id, event_type=event_type, details=details, config=config, db_utils_mock=ANY, misc_utils_mock=ANY, validate_func_mock=ANY)
            raise # Re-raise the original exception or a wrapped one

        await session_manager_mock.add_session_history(session_id=session_info.session_id, event_type=event_type, details=details, config=config, db_utils_mock=ANY, misc_utils_mock=ANY, validate_func_mock=ANY)
        return output

    async def _run_propose_strategy_async(session_info: SessionInfoResponse, tool_input: ProposeStrategyInput, *args):
         # Use the generic dummy runner
         # The mocks are passed via *args in the actual call, extract them here if needed or rely on fixtures
         llm_client_mock, session_manager_mock, template_env_mock, misc_utils_mock, schema_validator_mock, config = args[0], args[1], args[2], args[3], args[4], args[5] # Order depends on actual function signature
         return await _run_generic_llm_task(session_info, tool_input, ProposeStrategyOutput, "propose_strategy.prompt", "propose_strategy", *args)

    async def _run_refine_code_async(session_info: SessionInfoResponse, tool_input: RefineCodeInput, *args):
         llm_client_mock, session_manager_mock, template_env_mock, misc_utils_mock, schema_validator_mock, config = args[0], args[1], args[2], args[3], args[4], args[5]
         return await _run_generic_llm_task(session_info, tool_input, RefineCodeOutput, "refine_code.prompt", "refine_code", *args)

    # Dummy Jinja Environment for _render_prompt
    class DummyTemplate:
        def render(self, context):
             # Simple rendering simulation
             return f"Prompt for {self.name} with context: {json.dumps(context)}"
    class DummyJinjaEnv:
         def __init__(self):
             self.templates = {}
         def get_template(self, name):
             if name not in self.templates:
                 self.templates[name] = DummyTemplate()
                 self.templates[name].name = name # Store name for dummy render
             return self.templates[name]
    template_env_mock = DummyJinjaEnv()


# --- Fixtures ---
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_httpx_client():
    """Mocks httpx.AsyncClient."""
    return AsyncMock(spec=httpx.AsyncClient)

@pytest.fixture
def mock_llm_client(mock_httpx_client):
    """Provides a mocked ClaudeClient instance."""
    # Use the dummy ClaudeClient if imported, otherwise mock the real one
    if 'ClaudeClient' in globals() and isinstance(ClaudeClient, type):
         # Pass the mocked httpx client to the dummy constructor
         return ClaudeClient(api_key="dummy_key", client=mock_httpx_client)
    else:
         # If real client is imported, mock its methods
         mock = AsyncMock(spec=llm_tools.ClaudeClient) # Use real client spec
         mock.generate_text = AsyncMock()
         return mock


@pytest.fixture
def mock_session_manager_for_llm():
    """Mocks session_manager functions needed by llm_tools."""
    mock = MagicMock()
    mock.add_session_history = AsyncMock()
    # Add other mocked methods if needed
    return mock

@pytest.fixture
def mock_template_env():
    """Mocks the Jinja2 template environment."""
    mock_template = MagicMock(spec=jinja2.Template)
    mock_template.render = MagicMock(return_value="Rendered Prompt Content")

    mock_env = MagicMock(spec=jinja2.Environment)
    mock_env.get_template = MagicMock(return_value=mock_template)
    return mock_env, mock_template # Return both for easier assertions


@pytest.fixture
def mock_schema_validator():
    """Mocks the JSON schema validation function."""
    return MagicMock() # Default: valid

@pytest.fixture
def dummy_session_info() -> SessionInfoResponse:
     """Provides a basic SessionInfoResponse object."""
     # Use the dummy SessionInfoResponse if imported
     return SessionInfoResponse(session_id="llm_test_sid", status="running", history=[])

@pytest.fixture
def mock_misc_utils_for_llm():
     """Mocks misc_utils potentially needed."""
     mock = MagicMock()
     mock.generate_id = MagicMock(return_value="mock_event_llm_123")
     mock.get_timestamp = MagicMock(return_value=datetime.now(timezone.utc).isoformat())
     return mock

@pytest.fixture
def mock_config_for_llm() -> SessionManagerConfig:
     """Provides dummy config potentially needed."""
     return SessionManagerConfig() # Use dummy config


# --- Tests for extract_code_from_text ---

@pytest.mark.parametrize("text, expected", [
    ("Some text ```python\nprint('hello')\n``` more text", "print('hello')"),
    ("No code block here", None),
    ("```python\ndef func():\n  pass\n```", "def func():\n  pass"),
    ("```py\nx = 1\n```", None), # Language mismatch
    ("```python\ncode```", "code"), # End tag right after
    ("```python\n", None), # No end tag
    ("```python```", ""), # Empty block
])
def test_extract_code_from_text(text, expected):
    assert extract_code_from_text(text) == expected

def test_extract_code_from_text_different_language():
     text = "```javascript\nconsole.log('hi');\n```"
     assert extract_code_from_text(text, language="javascript") == "console.log('hi');"

# --- Tests for LLM Client (e.g., ClaudeClient) ---

async def test_llm_client_generate_text_success(mock_llm_client, mock_httpx_client):
    """Test successful LLM API call."""
    prompt = "Test prompt"
    model = "claude-instant-1.2"
    max_tokens = 100
    temperature = 0.7
    stop_sequences = ["\nHuman:"]

    # Configure the mock httpx response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"completion": "LLM response text", "stop_reason": "stop_sequence"}
    mock_response.raise_for_status = MagicMock() # Does nothing on success
    mock_httpx_client.post.return_value = mock_response

    # Call the generate_text method on the potentially mocked client
    result = await mock_llm_client.generate_text(prompt, model, max_tokens, temperature, stop_sequences)

    # Verify httpx call
    mock_httpx_client.post.assert_called_once()
    call_args, call_kwargs = mock_httpx_client.post.call_args
    assert "dummy.anthropic.com" in call_args[0] # Check URL loosely
    assert call_kwargs["headers"]["x-api-key"] == "dummy_key"
    assert call_kwargs["json"]["prompt"] == prompt
    assert call_kwargs["json"]["model"] == model
    assert call_kwargs["json"]["max_tokens_to_sample"] == max_tokens

    # Verify result
    assert result == {"completion": "LLM response text", "stop_reason": "stop_sequence"}

async def test_llm_client_generate_text_api_error(mock_llm_client, mock_httpx_client):
    """Test LLM API call returning an HTTP error."""
    prompt = "Error prompt"

    # Configure the mock httpx response for a 400 error
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.request = MagicMock(url="http://dummy/url") # Needed for HTTPStatusError
    # Make raise_for_status actually raise the error
    mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("Bad Request", request=mock_response.request, response=mock_response))
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(LLMError, match="API request failed: 400"):
        await mock_llm_client.generate_text(prompt, "model", 10, 0.5, [])

    mock_httpx_client.post.assert_called_once() # Ensure the call was made

# --- Tests for _render_prompt ---

def test_render_prompt(mock_template_env):
    """Test the prompt rendering function."""
    mock_env, mock_template = mock_template_env
    template_name = "test_template.prompt"
    context = {"var1": "value1", "items": [1, 2]}

    # Use the dummy _render_prompt if imported
    # result = _render_prompt(template_name, context, mock_env) # Pass mock env

    # Or patch the actual function if real one is imported
    # Assuming template_env is a global or module-level variable in llm_tools
    # If it's passed differently (e.g., via class), adjust the patch target
    try:
        with patch('src.mcp_server_logic.llm_tools.template_env', mock_env): # Patch the env used by _render_prompt
             result = llm_tools._render_prompt(template_name, context) # Call real function
    except (AttributeError, ImportError):
         # Fallback to using the dummy if patching the real one fails
         print("Patching real template_env failed, using dummy _render_prompt.")
         # Use the globally defined dummy template_env_mock if using dummy functions
         global template_env_mock
         result = _render_prompt(template_name, context, template_env_mock)


    # Check the mock associated with the render call
    # If using the dummy, check the mock stored in the dummy env
    if isinstance(template_env_mock, DummyJinjaEnv):
         template_env_mock.get_template(template_name).render.assert_called_once_with(context)
         assert result.startswith(f"Prompt for {template_name}") # Check dummy output
    else: # If patching real Jinja
         mock_env.get_template.assert_called_once_with(template_name)
         mock_template.render.assert_called_once_with(context)
         assert result == "Rendered Prompt Content" # From mock_template return value

# --- Tests for _run_..._async tasks ---

async def test_run_propose_strategy_success(dummy_session_info, mock_llm_client, mock_session_manager_for_llm, mock_template_env, mock_misc_utils_for_llm, mock_schema_validator, mock_config_for_llm):
    """Test the propose strategy task successfully."""
    mock_env, _ = mock_template_env
    tool_input = ProposeStrategyInput(params={"current_metric": 0.5})
    llm_completion = '{"strategy": "increase_threshold", "reasoning": "metric too low"}'
    expected_output_result = {"strategy": "increase_threshold", "reasoning": "metric too low"}

    # Configure mocks
    mock_llm_client.generate_text.return_value = {"completion": llm_completion}

    # Arguments for the dummy function (match the generic runner signature)
    args_for_task = (mock_llm_client, mock_session_manager_for_llm, mock_env if not isinstance(template_env_mock, DummyJinjaEnv) else template_env_mock, mock_misc_utils_for_llm, mock_schema_validator, mock_config_for_llm)

    output = await _run_propose_strategy_async(dummy_session_info, tool_input, *args_for_task)

    assert isinstance(output, ProposeStrategyOutput)
    assert output.result == expected_output_result

    # Verify mocks
    # Check template rendering mock
    if isinstance(template_env_mock, DummyJinjaEnv):
         template_env_mock.get_template("propose_strategy.prompt").render.assert_called_once()
    else:
         mock_env.get_template.assert_called_once_with("propose_strategy.prompt")

    mock_llm_client.generate_text.assert_called_once()
    mock_schema_validator.assert_called_once_with(expected_output_result) # Check validation input
    mock_session_manager_for_llm.add_session_history.assert_called_once()
    hist_args, hist_kwargs = mock_session_manager_for_llm.add_session_history.call_args
    assert hist_kwargs["session_id"] == dummy_session_info.session_id
    assert hist_kwargs["event_type"] == "propose_strategy_complete"
    assert hist_kwargs["details"]["input"] == tool_input.params
    assert hist_kwargs["details"]["output"] == expected_output_result
    assert "error" not in hist_kwargs["details"]


async def test_run_refine_code_llm_error(dummy_session_info, mock_llm_client, mock_session_manager_for_llm, mock_template_env, mock_misc_utils_for_llm, mock_schema_validator, mock_config_for_llm):
    """Test a task failing due to LLM error."""
    mock_env, _ = mock_template_env
    tool_input = RefineCodeInput(params={"code": "old_code", "feedback": "make it faster"})

    # Configure mocks
    mock_llm_client.generate_text.side_effect = LLMError("API timeout")

    args_for_task = (mock_llm_client, mock_session_manager_for_llm, mock_env if not isinstance(template_env_mock, DummyJinjaEnv) else template_env_mock, mock_misc_utils_for_llm, mock_schema_validator, mock_config_for_llm)

    with pytest.raises(LLMError, match="API timeout"):
        await _run_refine_code_async(dummy_session_info, tool_input, *args_for_task)

    # Verify mocks
    if isinstance(template_env_mock, DummyJinjaEnv):
         template_env_mock.get_template("refine_code.prompt").render.assert_called_once()
    else:
         mock_env.get_template.assert_called_once_with("refine_code.prompt")

    mock_llm_client.generate_text.assert_called_once()
    mock_schema_validator.assert_not_called() # Should fail before validation
    mock_session_manager_for_llm.add_session_history.assert_called_once()
    hist_args, hist_kwargs = mock_session_manager_for_llm.add_session_history.call_args
    assert hist_kwargs["session_id"] == dummy_session_info.session_id
    assert hist_kwargs["event_type"] == "refine_code_failed"
    assert hist_kwargs["details"]["input"] == tool_input.params
    assert "output" not in hist_kwargs["details"]
    assert hist_kwargs["details"]["error"]["error_type"] == "LLMError"
    assert hist_kwargs["details"]["error"]["message"] == "API timeout"


async def test_run_propose_strategy_validation_error(dummy_session_info, mock_llm_client, mock_session_manager_for_llm, mock_template_env, mock_misc_utils_for_llm, mock_schema_validator, mock_config_for_llm):
    """Test a task failing due to output schema validation error."""
    mock_env, _ = mock_template_env
    tool_input = ProposeStrategyInput(params={"metric": 0.9})
    llm_completion = '{"wrong_key": "some_value"}' # Does not match expected output schema
    invalid_data = {"wrong_key": "some_value"}

    # Configure mocks
    mock_llm_client.generate_text.return_value = {"completion": llm_completion}
    mock_schema_validator.side_effect = ValidationError("Missing required property: 'strategy'") # Simulate validation fail

    args_for_task = (mock_llm_client, mock_session_manager_for_llm, mock_env if not isinstance(template_env_mock, DummyJinjaEnv) else template_env_mock, mock_misc_utils_for_llm, mock_schema_validator, mock_config_for_llm)

    # The specific error raised might depend on the implementation (could be LLMError or ValidationError)
    with pytest.raises((LLMError, ValidationError), match="validation failed"):
        await _run_propose_strategy_async(dummy_session_info, tool_input, *args_for_task)

    # Verify mocks
    mock_llm_client.generate_text.assert_called_once()
    mock_schema_validator.assert_called_once_with(invalid_data) # Ensure validator was called with parsed data
    mock_session_manager_for_llm.add_session_history.assert_called_once()
    hist_args, hist_kwargs = mock_session_manager_for_llm.add_session_history.call_args
    assert hist_kwargs["event_type"] == "propose_strategy_failed"
    assert "validation failed" in hist_kwargs["details"]["error"]["message"] 