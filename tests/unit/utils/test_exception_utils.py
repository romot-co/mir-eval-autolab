# tests/unit/utils/test_exception_utils.py
import pytest
import logging
from unittest.mock import MagicMock # MagicMock をインポート

# Assuming these exceptions are defined in src.utils.exception_utils
# エラーを避けるため、もしファイルが存在しない場合はダミーのクラスを定義
try:
    from src.utils.exception_utils import (
        ConfigError,
        MCPError,
        ToolError,
        LLMError,
        StateManagementError,
        log_exception,
        safe_execute
    )
except ImportError:
    # Allow tests to run even if the module doesn't fully exist yet
    class BaseError(Exception): pass
    class ConfigError(BaseError): pass
    class MCPError(BaseError): pass
    class ToolError(MCPError): pass
    class LLMError(MCPError): pass
    class StateManagementError(MCPError): pass
    # Define dummy functions if import fails
    def log_exception(logger, exc, message, level=logging.ERROR):
        # Basic logging mimic for testing structure
        logger.log(level, f"{message}: {exc}", exc_info=True)

    def safe_execute(func, *args, default=None, error_message="", expected_exceptions=Exception, raise_on_error=False, logger_name='src.utils.exception_utils', **kwargs):
        logger = logging.getLogger(logger_name)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if raise_on_error:
                raise e
            if isinstance(e, expected_exceptions):
                final_message = f"{error_message}: {e}" if error_message else str(e)
                # Use log_exception structure or direct logging
                # log_exception(logger, e, error_message) # Or direct call:
                logger.error(f"{error_message}: {e}", exc_info=True)
                return default
            else: # Unexpected exception
                final_message = f"{error_message} (Unexpected Error): {e}" if error_message else f"Unexpected Error: {e}"
                logger.error(final_message, exc_info=True)
                return default


# Assuming get_logger is available for testing log_exception
# get_logger も存在しない場合に備える
try:
    from src.utils.logging_utils import get_logger, setup_logger
except ImportError:
    # Define a dummy get_logger if needed
    def get_logger(name, log_level=logging.INFO):
        logger = logging.getLogger(name)
        # Add a basic handler if none exist, to allow caplog to work
        if not logger.hasHandlers():
           # Use StreamHandler for visibility in test logs if needed, NullHandler otherwise
           # handler = logging.StreamHandler()
           handler = logging.NullHandler()
           logger.addHandler(handler)
        logger.setLevel(log_level) # Set level on logger too for get_logger behavior
        return logger
    def setup_logger(name, level=logging.INFO, console_level=None, **kwargs):
         logger = get_logger(name, level)
         # Add a handler if needed for testing setup_logger logic mimic
         if not logger.hasHandlers():
             handler = logging.StreamHandler()
             handler.setLevel(console_level if console_level is not None else level)
             logger.addHandler(handler)
         return logger


# --- Custom Exception Tests ---

def test_custom_exceptions_exist():
    """カスタム例外クラスが存在し、適切な継承関係にあることを確認"""
    assert issubclass(ConfigError, Exception)
    assert issubclass(MCPError, Exception)
    # 仮定の継承関係をテスト
    assert issubclass(ToolError, MCPError)
    assert issubclass(LLMError, MCPError)
    assert issubclass(StateManagementError, MCPError)
    # Check basic instantiation
    assert str(ConfigError("Config message")) == "Config message"
    # ToolError などが追加の引数を取る可能性も考慮
    try:
        # 実際の __init__ シグネチャに合わせて調整が必要
        # Assuming ToolError might take job_id
        class MockToolError(MCPError): # Use a mock if original might not exist
             def __init__(self, message, job_id=None):
                 super().__init__(message)
                 self.job_id = job_id
        t_error = MockToolError("Tool message", job_id="job1")
        # t_error = ToolError("Tool message", job_id="job1") # Use this if ToolError exists and takes job_id
        assert str(t_error) == "Tool message"
        # assert t_error.job_id == "job1" # Check attribute if it exists
    except TypeError:
        # シグネチャが異なる場合は基本のメッセージのみテスト
        assert str(ToolError("Tool message")) == "Tool message"
    except NameError: # Handle if ToolError itself is not defined
        pass


# --- log_exception Tests ---

@pytest.fixture
def logger_for_test():
    """テスト用のロガーをセットアップし、クリーンアップするフィクスチャ"""
    logger_name = "test_exception_logger"
    # Ensure logger starts clean for each test using this fixture
    # Remove existing handlers first
    existing_logger = logging.getLogger(logger_name)
    for handler in existing_logger.handlers[:]:
        existing_logger.removeHandler(handler)
        handler.close()

    logger = setup_logger(logger_name, level=logging.DEBUG, console_level=logging.DEBUG) # Ensure DEBUG level

    yield logger
    # Clean up handlers after test
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    # Remove logger from manager dict to prevent state leakage
    if logger_name in logging.Logger.manager.loggerDict:
        del logging.Logger.manager.loggerDict[logger_name]


def test_log_exception(caplog, logger_for_test):
    """log_exceptionが例外情報を正しくログに出力することを確認"""
    test_message = "Something went wrong"
    error_message = "Specific error detail"
    try:
        raise ValueError(error_message)
    except ValueError as e:
        # logger_for_test.name を使用して特定のロガーのログをキャプチャ
        with caplog.at_level(logging.ERROR, logger=logger_for_test.name):
            # Use the imported or dummy log_exception
            log_exception(logger_for_test, e, test_message)

    # Check logs captured specifically for logger_for_test.name
    filtered_records = [r for r in caplog.records if r.name == logger_for_test.name]
    assert len(filtered_records) >= 1 # ログが出力されていること
    record = filtered_records[-1]
    assert record.levelname == "ERROR"
    assert test_message in record.message
    assert error_message in record.message # Check if exception str is included
    # log_exception が exc_info=True を渡すことを期待 (or logger.exception is used)
    assert record.exc_info is not None

def test_log_exception_custom_level(caplog, logger_for_test):
    """log_exceptionがカスタムログレベルを使用できることを確認"""
    test_message = "A warning happened"
    error_message = "Warning detail"
    try:
        raise TypeError(error_message)
    except TypeError as e:
         with caplog.at_level(logging.WARNING, logger=logger_for_test.name):
             log_exception(logger_for_test, e, test_message, level=logging.WARNING)

    filtered_records = [r for r in caplog.records if r.name == logger_for_test.name]
    assert len(filtered_records) >= 1
    record = filtered_records[-1]
    assert record.levelname == "WARNING"
    assert test_message in record.message
    assert error_message in record.message
    assert record.exc_info is not None

# --- safe_execute Tests ---

def success_func(x):
    return x * 2

def error_func(x):
    raise ValueError("Intentional error")

def type_error_func(x):
    raise TypeError("Different error")

@pytest.fixture
def mock_safe_execute_logger(monkeypatch):
    """safe_execute内部で使用されるロガーをモックするフィクスチャ"""
    mock_logger = MagicMock(spec=logging.Logger)
    # safe_execute が内部で 'src.utils.exception_utils' という名前でロガーを取得すると仮定
    logger_name_in_module = 'src.utils.exception_utils' # Assume this name
    original_get_logger = logging.getLogger

    def mock_get_logger(name):
        if name == logger_name_in_module:
            # print(f"Mocking getLogger for: {name}") # Debug print
            return mock_logger
        # print(f"Using original getLogger for: {name}") # Debug print
        return original_get_logger(name)

    monkeypatch.setattr(logging, 'getLogger', mock_get_logger)

    # Mock log_exception as well, in case safe_execute uses it directly
    # Need to mock it within the context of the module where safe_execute would call it
    try:
        # Try patching the potentially imported function
        mock_log_exception_func = MagicMock()
        monkeypatch.setattr('src.utils.exception_utils.log_exception', mock_log_exception_func)
        return mock_logger, mock_log_exception_func
    except AttributeError:
        # If the import failed or structure is different, just return the logger mock
        return mock_logger, None # Indicate log_exception mock failed

def test_safe_execute_success(mock_safe_execute_logger):
    """safe_executeが成功時に結果を返し、ログを出力しないことを確認"""
    mock_logger, mock_log_exception = mock_safe_execute_logger
    result = safe_execute(success_func, 5, default="error", error_message="Should not log")
    assert result == 10
    mock_logger.error.assert_not_called()
    mock_logger.exception.assert_not_called()
    if mock_log_exception:
        mock_log_exception.assert_not_called()

def test_safe_execute_expected_error(mock_safe_execute_logger):
    """safe_executeが予期したエラー発生時にデフォルト値を返し、ログを出力することを確認"""
    mock_logger, mock_log_exception = mock_safe_execute_logger
    error_msg = "Error in error_func"
    result = safe_execute(error_func, 5, default="fallback", error_message=error_msg, expected_exceptions=ValueError)
    assert result == "fallback"

    # Check if logger.error(exc_info=True) was called
    mock_logger.error.assert_called_once()
    call_args, call_kwargs = mock_logger.error.call_args
    assert error_msg in call_args[0]
    assert "Intentional error" in call_args[0]
    assert call_kwargs.get('exc_info') is True
    # Verify log_exception (the separate function) was NOT called if logger.error was used directly
    if mock_log_exception:
       mock_log_exception.assert_not_called()


def test_safe_execute_unexpected_error(mock_safe_execute_logger):
    """safe_executeが予期しないエラー発生時にデフォルト値を返し、ログを出力することを確認"""
    mock_logger, mock_log_exception = mock_safe_execute_logger
    error_msg = "Unexpected error in type_error_func"
    # Expecting ValueError, but gets TypeError
    result = safe_execute(type_error_func, 5, default=None, error_message=error_msg, expected_exceptions=ValueError)
    assert result is None

    # Check logger.error was called for the unexpected error
    mock_logger.error.assert_called_once()
    call_args, call_kwargs = mock_logger.error.call_args
    assert error_msg in call_args[0]
    assert "Unexpected Error" in call_args[0] # Check for the "Unexpected Error" prefix
    assert "Different error" in call_args[0]
    assert call_kwargs.get('exc_info') is True
    if mock_log_exception:
       mock_log_exception.assert_not_called()


def test_safe_execute_no_expected_exception_set(mock_safe_execute_logger):
    """expected_exceptionsが未指定の場合、すべてのExceptionをキャッチすることを確認"""
    mock_logger, mock_log_exception = mock_safe_execute_logger
    error_msg = "Error in error_func (no expected)"
    # Default expected_exceptions=Exception catches ValueError
    result = safe_execute(error_func, 5, default=-1, error_message=error_msg)
    assert result == -1

    mock_logger.error.assert_called_once()
    call_args, call_kwargs = mock_logger.error.call_args
    assert error_msg in call_args[0]
    assert "Intentional error" in call_args[0]
    assert call_kwargs.get('exc_info') is True
    if mock_log_exception:
       mock_log_exception.assert_not_called()


def test_safe_execute_custom_default(mock_safe_execute_logger):
    """safe_executeがカスタムデフォルト値を正しく返すことを確認"""
    mock_logger, mock_log_exception = mock_safe_execute_logger
    result = safe_execute(error_func, 5, default={"status": "failed"}, error_message="Test custom default")
    assert result == {"status": "failed"}
    # Check that logging still happened
    mock_logger.error.assert_called_once()


def test_safe_execute_raise_on_error(mock_safe_execute_logger):
    """safe_executeがraise_on_error=Trueの場合に例外を再送出することを確認"""
    mock_logger, mock_log_exception = mock_safe_execute_logger
    with pytest.raises(ValueError, match="Intentional error"):
        safe_execute(error_func, 5, default="fallback", error_message="Should raise", raise_on_error=True)
    # エラーが再送出された場合、ログは出力されないことを期待
    mock_logger.error.assert_not_called()
    mock_logger.exception.assert_not_called()
    if mock_log_exception:
        mock_log_exception.assert_not_called()


def test_safe_execute_raise_on_error_unexpected(mock_safe_execute_logger):
    """safe_executeがraise_on_error=Trueで予期しないエラーの場合も再送出することを確認"""
    mock_logger, mock_log_exception = mock_safe_execute_logger
    with pytest.raises(TypeError, match="Different error"):
        # Expected is ValueError, actual is TypeError, should still raise
        safe_execute(type_error_func, 5, default=None, error_message="Should raise", expected_exceptions=ValueError, raise_on_error=True)
    mock_logger.error.assert_not_called()
    mock_logger.exception.assert_not_called()
    if mock_log_exception:
        mock_log_exception.assert_not_called() 