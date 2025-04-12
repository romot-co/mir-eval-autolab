import pytest
import json
import logging
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

# テスト対象のモジュールからのインポート
try:
    from src.cli.mirai import (
        poll_job_status,
        print_session_summary,
        determine_next_action,
        execute_analyze_evaluation,
        execute_generate_hypotheses,
        execute_improve_code,
        execute_run_evaluation,
        execute_optimize_parameters,
    )
    from src.cli.llm_client import LLMClientError
    from src.mcp_server_logic.schemas import JobStatus
    try:
        from mcp import McpError
    except ImportError:
        class McpError(Exception): pass
    class ToolError(Exception): pass

except ImportError as e:
    pytest.fail(f"Failed to import mirai or related modules: {e}. Ensure PYTHONPATH is correct.")

# --- 定数 ---
MOCK_SERVER_URL = "http://mock-mcp-server:1234"
MOCK_AUDIO_FILE = "test_audio.wav"
MOCK_REF_FILE = "test_ref.csv"
MOCK_OUTPUT_DIR = "test_output"
MOCK_CONFIG_FILE = "grid_config.yaml"
MOCK_DETECTOR_NAME = "TestDetector"
MOCK_DATASET_NAME = "test_dataset"
MOCK_SESSION_ID = "sess_test123"
MOCK_JOB_ID = "job_test456"
DEFAULT_LLM_MODEL = "claude-3-opus-20240229"
DEFAULT_POLL_INTERVAL = 5
DEFAULT_JOB_TIMEOUT = 600
DEFAULT_MAX_CYCLES = 10
DEFAULT_IMPROVEMENT_THRESHOLD = 0.01
DEFAULT_MAX_STAGNATION = 3
DEFAULT_SESSION_TIMEOUT = 3600

# --- 基本的なフィクスチャ ---

@pytest.fixture
def mock_config(mocker):
    """グローバル設定をモック化"""
    mock_cfg = {
        'evaluation': {'mir_eval_options': {'onset_tolerance': 0.05}},
        'visualization': {'default_plot_format': 'png'},
        'llm': {'model': DEFAULT_LLM_MODEL},
        'improve': {
            'poll_interval': DEFAULT_POLL_INTERVAL,
            'job_timeout': DEFAULT_JOB_TIMEOUT,
            'max_cycles': DEFAULT_MAX_CYCLES,
        },
        'strategy': {'improvement_threshold': DEFAULT_IMPROVEMENT_THRESHOLD, 'max_stagnation': DEFAULT_MAX_STAGNATION},
        'cleanup': {'session_timeout_seconds': DEFAULT_SESSION_TIMEOUT}
    }
    mocker.patch('src.cli.mirai.CONFIG', mock_cfg, create=True)
    mocker.patch('src.cli.mirai.load_global_config')
    return mock_cfg

@pytest.fixture
def mock_path_utils(mocker):
    """pathlib関数のモック化"""
    mocker.patch('pathlib.Path.exists', return_value=True)
    mocker.patch('pathlib.Path.is_file', return_value=True)
    mocker.patch('pathlib.Path.mkdir')
    return mocker

@pytest.fixture
def mock_open(mocker):
    """open関数のモック化"""
    return mocker.patch('builtins.open', mocker.mock_open(read_data='key: value'))

@pytest.fixture
def mock_yaml_load(mocker):
    """yaml.safe_loadのモック化"""
    return mocker.patch('yaml.safe_load', return_value={'mock_key': 'mock_value'})

@pytest.fixture
def mock_mcp_client(mocker):
    """MCPClientのモック化"""
    mock_client = MagicMock(name="MockMCPClientInstance")
    mock_client.run_tool = MagicMock(return_value={'job_id': MOCK_JOB_ID})
    mock_client.get_job_status = MagicMock(return_value={'status': JobStatus.COMPLETED.value, 'result': {}})
    mock_client.get_session_info = MagicMock(return_value={
        'session_id': MOCK_SESSION_ID, 
        'status': 'active', 
        'best_metrics': {'overall': {'f_measure': 0.8}}
    })
    mock_client.start_session = MagicMock(return_value={'session_id': MOCK_SESSION_ID, 'status': 'active'})
    mock_client.get_code = MagicMock(return_value={'code': 'print("mock code")'})
    mock_client.save_code = MagicMock(return_value={'version': 'v_mock'})
    mock_client.add_session_history = MagicMock(return_value=None)

    mocker.patch('src.cli.mirai.get_mcp_client', return_value=mock_client)
    return mock_client

@pytest.fixture
def mock_mcp_with_stages():
    """異なるステージの応答を返すMCPクライアント"""
    client = MagicMock(name="StagedMCPClientInstance")
    
    def configure_responses(statuses):
        client.get_job_status.side_effect = [
            {"status": status} for status in statuses
        ]
    
    client.configure = configure_responses
    return client

@pytest.fixture
def mock_llm_client(mocker):
    """LLMクライアントのモック化"""
    mock_llm = AsyncMock(name="MockLLMClientInstance")
    mock_llm.generate = AsyncMock(return_value=json.dumps({
        "action": "improve_code",
        "parameters": {"suggestion": "LLM mock suggestion", "code": "print('llm mock code')"}
    }))
    mock_llm.extract_code_from_text = AsyncMock(return_value="print('extracted llm code')")
    mocker.patch('src.cli.mirai.initialize_llm_client_async', return_value=mock_llm)
    return mock_llm

@pytest.fixture
def mock_typer_confirm(mocker):
    """typer.confirmのモック化"""
    return mocker.patch('typer.confirm', return_value=True)

@pytest.fixture
def mock_evaluate_detector(mocker):
    """evaluate_detector関数のモック化"""
    return mocker.patch(
        'src.cli.mirai.evaluate_detector',
        return_value={'metrics': {'note': {'f_measure': 0.9}}}
    )

# --- Improve Command Helpers ---

@pytest.fixture
def mock_determine_next_action(mocker):
    """determine_next_action関数のモック化"""
    mock_action = AsyncMock(return_value={
        "action": "improve_code",
        "parameters": {"suggestion": "Default mock suggestion", "code": "print('default code')"},
        "session_status": "running"
    })
    mocker.patch('src.cli.mirai.determine_next_action', mock_action)
    return mock_action

@pytest.fixture
def mock_execute_analyze_evaluation(mocker):
    """execute_analyze_evaluation関数のモック化"""
    return mocker.patch('src.cli.mirai.execute_analyze_evaluation', AsyncMock(return_value=True))

@pytest.fixture
def mock_execute_generate_hypotheses(mocker):
    """execute_generate_hypotheses関数のモック化"""
    return mocker.patch('src.cli.mirai.execute_generate_hypotheses', AsyncMock(return_value=True))

@pytest.fixture
def mock_execute_improve_code(mocker):
    """execute_improve_code関数のモック化"""
    return mocker.patch('src.cli.mirai.execute_improve_code', AsyncMock(return_value=True))

@pytest.fixture
def mock_execute_run_evaluation(mocker):
    """execute_run_evaluation関数のモック化"""
    return mocker.patch('src.cli.mirai.execute_run_evaluation', AsyncMock(return_value=True))

@pytest.fixture
def mock_execute_optimize_parameters(mocker):
    """execute_optimize_parameters関数のモック化"""
    return mocker.patch('src.cli.mirai.execute_optimize_parameters', AsyncMock(return_value=True))

@pytest.fixture
def mock_poll_job_status(mocker):
    """poll_job_status関数のモック化"""
    return mocker.patch('src.cli.mirai.poll_job_status', return_value={
        'status': JobStatus.COMPLETED.value,
        'result': {'summary': 'Mock job completed'}
    }) 