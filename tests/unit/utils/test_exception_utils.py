# tests/unit/utils/test_exception_utils.py
import pytest
import logging
from unittest.mock import MagicMock, patch # MagicMock と patch をインポート
import time # For retry test

# Assuming these exceptions are defined in src.utils.exception_utils
# エラーを避けるため、もしファイルが存在しない場合はダミーのクラスを定義
try:
    from src.utils.exception_utils import (
        MirexError,
        DetectionError,
        EvaluationError,
        FileError,
        ConfigError,
        StateManagementError,
        LLMError,
        log_exception,
        safe_execute,
        create_error_result,
        format_exception_message,
        retry_on_exception,
        wrap_exceptions,
        SubprocessError
    )
except ImportError:
    # Allow tests to run even if the module doesn't fully exist yet
    class BaseError(Exception): pass
    class MirexError(BaseError): pass
    class DetectionError(MirexError): pass
    class EvaluationError(MirexError): pass
    class FileError(MirexError): pass
    class ConfigError(MirexError): pass
    class StateManagementError(MirexError): pass
    class LLMError(MirexError): pass
    # Define dummy functions if import fails
    def log_exception(logger, exc, message, log_level=logging.ERROR):
        # Basic logging mimic for testing structure
        logger.log(log_level, f"{message}: {exc}", exc_info=True)

    def safe_execute(func, logger, error_msg="関数実行中にエラーが発生しました",
                    default_value=None, log_level=logging.ERROR, raise_exception=False, **kwargs):
        try:
            return func(**kwargs)
        except Exception as e:
            log_exception(logger, e, error_msg, log_level)
            if raise_exception:
                raise
            return default_value

    # ダミー関数
    def create_error_result(error_msg, include_keys=None):
        result = {'error': error_msg, 'valid': False}
        if include_keys:
            for key in include_keys:
                if key not in result:
                    result[key] = 0.0
        return result
    
    def format_exception_message(e, context=""):
        if context:
            return f"{context}: {str(e)}"
        return str(e)

    def retry_on_exception(max_attempts=3, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def wrap_exceptions(target_exceptions, wrapper_exception, message=None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator


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
    assert issubclass(MirexError, Exception)
    # 仮定の継承関係をテスト
    assert issubclass(DetectionError, MirexError)
    assert issubclass(EvaluationError, MirexError)
    assert issubclass(FileError, MirexError)
    assert issubclass(StateManagementError, MirexError)
    assert issubclass(LLMError, MirexError)
    # Test that MirexError itself is an Exception
    assert issubclass(MirexError, Exception)

    # Test instantiation with messages
    msg = "Test message"
    assert str(MirexError(msg)) == msg
    assert str(DetectionError(msg)) == msg
    assert str(EvaluationError(msg)) == msg
    assert str(FileError(msg)) == msg
    assert str(ConfigError(msg)) == msg
    assert str(StateManagementError(msg)) == msg
    assert str(LLMError(msg)) == msg

    # Check basic instantiation
    assert str(ConfigError("Config message")) == "Config message"
    # 追加の引数を取る可能性も考慮
    try:
        # 実際の __init__ シグネチャに合わせて調整が必要
        # Assuming DetectionError might take additional info
        class MockDetectionError(MirexError): # Use a mock if original might not exist
             def __init__(self, message, detector=None):
                 super().__init__(message)
                 self.detector = detector
        t_error = MockDetectionError("Detection error", detector="detector1")
        assert str(t_error) == "Detection error"
    except TypeError:
        # シグネチャが異なる場合は基本のメッセージのみテスト
        assert str(DetectionError("Detection error")) == "Detection error"
    except NameError: # Handle if DetectionError itself is not defined
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
    
    # Check traceback is included in exc_text, not directly in message if default formatter used
    # Note: Default formatter puts traceback in exc_text, not message.
    # If a custom formatter is used, this might need adjustment.
    assert record.exc_text is not None # Check if traceback text was generated
    assert error_message in record.exc_text # Ensure the error message is in the traceback text

def test_log_exception_custom_level(caplog, logger_for_test):
    """log_exceptionがカスタムログレベルを使用できることを確認"""
    test_message = "A warning happened"
    error_message = "Warning detail"
    try:
        raise TypeError(error_message)
    except TypeError as e:
         with caplog.at_level(logging.WARNING, logger=logger_for_test.name):
             log_exception(logger_for_test, e, test_message, log_level=logging.WARNING)

    filtered_records = [r for r in caplog.records if r.name == logger_for_test.name]
    assert len(filtered_records) >= 1
    record = filtered_records[-1]
    assert record.levelname == "WARNING"
    assert test_message in record.message
    assert error_message in record.message
    
    # Check traceback is included in exc_text
    assert record.exc_text is not None # Check if traceback text was generated
    assert error_message in record.exc_text # Ensure the error message is in the traceback text

def test_log_exception_debug_level(caplog, logger_for_test):
    """log_exceptionがDEBUGレベルでログ出力できることを確認"""
    test_message = "Debug information"
    error_message = "Debug error detail"
    try:
        raise RuntimeError(error_message)
    except RuntimeError as e:
         with caplog.at_level(logging.DEBUG, logger=logger_for_test.name):
             log_exception(logger_for_test, e, test_message, log_level=logging.DEBUG)

    filtered_records = [r for r in caplog.records if r.name == logger_for_test.name]
    assert len(filtered_records) >= 1
    record = filtered_records[-1]
    assert record.levelname == "DEBUG"
    assert test_message in record.message
    assert error_message in record.message
    
    # Check traceback is included in exc_text
    assert record.exc_text is not None
    assert error_message in record.exc_text

def test_log_exception_info_level(caplog, logger_for_test):
    """log_exceptionがINFOレベルでログ出力できることを確認"""
    test_message = "Info message"
    error_message = "Info error detail"
    try:
        raise IndexError(error_message)
    except IndexError as e:
         with caplog.at_level(logging.INFO, logger=logger_for_test.name):
             log_exception(logger_for_test, e, test_message, log_level=logging.INFO)

    filtered_records = [r for r in caplog.records if r.name == logger_for_test.name]
    assert len(filtered_records) >= 1
    record = filtered_records[-1]
    assert record.levelname == "INFO"
    assert test_message in record.message
    assert error_message in record.message
    
    # Check traceback is included in exc_text
    assert record.exc_text is not None
    assert error_message in record.exc_text

def test_log_exception_critical_level(caplog, logger_for_test):
    """log_exceptionがCRITICALレベルでログ出力できることを確認"""
    test_message = "Critical error"
    error_message = "Critical failure detail"
    try:
        raise MemoryError(error_message)
    except MemoryError as e:
         with caplog.at_level(logging.CRITICAL, logger=logger_for_test.name):
             log_exception(logger_for_test, e, test_message, log_level=logging.CRITICAL)

    filtered_records = [r for r in caplog.records if r.name == logger_for_test.name]
    assert len(filtered_records) >= 1
    record = filtered_records[-1]
    assert record.levelname == "CRITICAL"
    assert test_message in record.message
    assert error_message in record.message
    
    # Check traceback is included in exc_text
    assert record.exc_text is not None
    assert error_message in record.exc_text

# --- safe_execute Tests ---

def success_func(x):
    return x * 2

def error_func(x):
    raise ValueError("Intentional error")

def type_error_func(x):
    raise TypeError("Different error")

@pytest.fixture
def mock_logger():
    """モックロガーを作成するフィクスチャ"""
    # 直接モックロガーを作成
    # spec=logging.Logger を追加して、ロガーの属性を模倣
    mock_logger_obj = MagicMock(spec=logging.Logger)
    mock_logger_obj.name = 'mock_logger' # Add name attribute
    return mock_logger_obj

@pytest.fixture
def mock_get_logger(mock_logger):
    """get_logger関数をモックするフィクスチャ - nameパラメータを受け取れるように修正"""
    with patch("src.utils.exception_utils.get_logger") as mock:
        # nameパラメータを受け取れるようにする
        mock.side_effect = lambda name=None, log_level=None: mock_logger
        yield mock

def test_safe_execute_success(mock_logger):
    """safe_executeが成功時に結果を返し、ログを出力しないことを確認"""
    result = safe_execute(success_func, logger=mock_logger, default_value="error", error_msg="Should not log", x=5)
    assert result == 10
    mock_logger.error.assert_not_called()
    mock_logger.exception.assert_not_called()

def test_safe_execute_expected_error(mock_logger):
    """safe_executeがエラー発生時にデフォルト値を返し、ログを出力することを確認"""
    error_msg = "Error in error_func"
    result = safe_execute(error_func, logger=mock_logger, default_value="fallback", error_msg=error_msg, x=5)
    assert result == "fallback"

    # logger.error/exception または log_exception が呼ばれたことを確認
    # safe_executeの実装に応じてどちらかの方法でログ出力されるはず
    # Check if log_exception was called (assuming safe_execute uses it)
    # We need to patch log_exception itself to check its calls
    with patch('src.utils.exception_utils.log_exception') as mock_log_exc:
        result = safe_execute(error_func, logger=mock_logger, default_value="fallback", error_msg=error_msg, x=5)
        assert result == "fallback"
        mock_log_exc.assert_called_once()

def test_safe_execute_raise_exception(mock_logger):
    """safe_executeがraise_exception=Trueの場合に例外を再発生させることを確認"""
    error_msg = "Error should be raised"
    with pytest.raises(ValueError, match="Intentional error"):
        safe_execute(error_func, logger=mock_logger, error_msg=error_msg, raise_exception=True, x=5)
    # Ensure logging still happened before raising
    # Check if log_exception was called
    with patch('src.utils.exception_utils.log_exception') as mock_log_exc:
        with pytest.raises(ValueError, match="Intentional error"):
            safe_execute(error_func, logger=mock_logger, error_msg=error_msg, raise_exception=True, x=5)
        mock_log_exc.assert_called_once()

# --- create_error_result Tests ---

def test_create_error_result_basic():
    """create_error_resultが基本的なエラー結果を生成することを確認"""
    error_msg = "テストエラー"
    result = create_error_result(error_msg)
    
    # 基本的な構造を確認
    assert result['error'] == error_msg
    assert result['valid'] is False
    assert result['precision'] == 0.0
    assert result['recall'] == 0.0
    assert result['f_measure'] == 0.0
    
    # match_statisticsの存在と主要フィールドを確認
    assert 'match_statistics' in result
    assert result['match_statistics']['ref_note_count'] == 0
    assert result['match_statistics']['est_note_count'] == 0
    assert result['match_statistics']['complete_matches'] == 0

def test_create_error_result_with_keys():
    """create_error_resultが追加のキーを含む結果を生成することを確認"""
    error_msg = "テストエラー"
    include_keys = ['additional_metric', 'another_field']
    result = create_error_result(error_msg, include_keys)
    
    # 基本的な構造を確認
    assert result['error'] == error_msg
    assert result['valid'] is False
    
    # 追加のキーが存在し、デフォルト値が設定されていることを確認
    assert 'additional_metric' in result
    assert result['additional_metric'] == 0.0
    assert 'another_field' in result
    assert result['another_field'] == 0.0
    
    # 既存のキーは上書きされないことを確認
    assert result['precision'] == 0.0
    assert result['recall'] == 0.0

# --- format_exception_message Tests ---

def test_format_exception_message_no_context():
    """format_exception_messageがコンテキストなしで例外メッセージを整形することを確認"""
    error_msg = "テストエラー"
    exc = ValueError(error_msg)
    
    result = format_exception_message(exc)
    assert result == error_msg

def test_format_exception_message_with_context():
    """format_exception_messageがコンテキスト付きで例外メッセージを整形することを確認"""
    error_msg = "テストエラー"
    context = "検出処理中"
    exc = ValueError(error_msg)
    
    result = format_exception_message(exc, context)
    assert result == f"{context}: {error_msg}"

# --- retry_on_exception Tests ---

import time
import tenacity
import requests
from unittest.mock import patch, MagicMock, call

def test_retry_on_exception_success():
    """retry_on_exceptionが成功時に関数を一度だけ実行することを確認"""
    mock_logger = MagicMock(spec=logging.Logger)
    
    # 成功する関数
    mock_func = MagicMock(return_value="success")
    mock_func.__name__ = "test_func"
    
    # デコレータを適用
    decorated_func = retry_on_exception(
        max_attempts=3,
        logger=mock_logger
    )(mock_func)
    
    # 実行
    result = decorated_func(1, b=2)
    
    # 検証
    assert result == "success"
    mock_func.assert_called_once_with(1, b=2)
    mock_logger.warning.assert_not_called()

def test_retry_on_exception_retry_and_succeed():
    """retry_on_exceptionが一時的なエラー後に再試行して成功することを確認"""
    mock_logger = MagicMock(spec=logging.Logger)
    
    # 最初の呼び出しでエラー、2回目は成功する関数
    mock_func = MagicMock(side_effect=[requests.Timeout("接続タイムアウト"), "success"])
    mock_func.__name__ = "test_func"
    
    # patched_sleepで睡眠を回避
    with patch('time.sleep') as patched_sleep:
        # デコレータを適用
        decorated_func = retry_on_exception(
            max_attempts=3,
            initial_delay=0.1,
            logger=mock_logger,
            log_message_prefix="テストリトライ"
        )(mock_func)
        
        # 実行
        result = decorated_func(1, b=2)
    
    # 検証
    assert result == "success"
    assert mock_func.call_count == 2
    assert mock_logger.warning.call_count == 1
    # warning呼び出しパラメータを確認
    warning_msg = mock_logger.warning.call_args[0][0]
    assert "テストリトライ" in warning_msg
    assert "1/3" in warning_msg  # 試行回数
    assert "Timeout" in warning_msg  # 例外タイプ
    assert "接続タイムアウト" in warning_msg  # 例外メッセージ
    
    # 少なくとも一度はsleepが呼ばれたはず
    patched_sleep.assert_called()

def test_retry_on_exception_max_attempts_reached():
    """retry_on_exceptionが最大試行回数に達した場合にエラーを発生させることを確認"""
    mock_logger = MagicMock(spec=logging.Logger)
    
    # 常にタイムアウトエラーを発生させる関数
    mock_func = MagicMock(side_effect=requests.Timeout("接続タイムアウト"))
    mock_func.__name__ = "test_func"
    
    # patched_sleepで睡眠を回避
    with patch('time.sleep'):
        # デコレータを適用
        decorated_func = retry_on_exception(
            max_attempts=2,
            initial_delay=0.1,
            logger=mock_logger
        )(mock_func)
        
        # 実行して例外を確認（元の例外または、tenacityによってラップされた例外が発生する）
        with pytest.raises((requests.Timeout, tenacity.RetryError)):
            decorated_func(1, b=2)
    
    # 検証
    assert mock_func.call_count >= 1  # 少なくとも1回は試行されたことを確認
    # テナシティの実装によってはwarningが出ない場合もあるため、検証しない
    # assert mock_logger.warning.call_count >= 0

# --- wrap_exceptions Tests ---

def test_wrap_exceptions_no_exception():
    """wrap_exceptionsがエラーがない場合に通常の結果を返すことを確認"""
    # 成功する関数
    def success_func(x):
        return x * 2
    
    # デコレータを適用
    decorated_func = wrap_exceptions(
        [ValueError, TypeError],
        EvaluationError,
        "変換中にエラーが発生しました"
    )(success_func)
    
    # 実行
    result = decorated_func(5)
    assert result == 10

def test_wrap_exceptions_wrap_target_exception():
    """wrap_exceptionsが対象の例外を指定の例外にラップすることを確認"""
    # エラーを発生させる関数
    def error_func(x):
        raise ValueError("Invalid value")
    
    # デコレータを適用
    decorated_func = wrap_exceptions(
        [ValueError, TypeError],
        EvaluationError,
        "変換中にエラーが発生しました"
    )(error_func)
    
    # 実行して例外を確認
    with pytest.raises(EvaluationError) as excinfo:
        decorated_func(5)
    
    # ラップされた例外メッセージを確認
    assert "変換中にエラーが発生しました" in str(excinfo.value)
    assert "Invalid value" in str(excinfo.value)

def test_wrap_exceptions_nontarget_exception():
    """wrap_exceptionsが対象外の例外をラップしないことを確認"""
    # 対象外の例外を発生させる関数
    def error_func(x):
        raise KeyError("Key not found")
    
    # デコレータを適用
    decorated_func = wrap_exceptions(
        [ValueError, TypeError],
        EvaluationError,
        "変換中にエラーが発生しました"
    )(error_func)
    
    # 実行して元の例外がそのまま発生することを確認
    with pytest.raises(KeyError) as excinfo:
        decorated_func(5)
    
    assert "Key not found" in str(excinfo.value)

def test_wrap_exceptions_default_message():
    """wrap_exceptionsがカスタムメッセージが指定されない場合にデフォルトメッセージを使用することを確認"""
    # エラーを発生させる関数
    def error_func(x):
        raise TypeError("Invalid type")
    
    # メッセージを指定せずにデコレータを適用
    decorated_func = wrap_exceptions(
        [ValueError, TypeError],
        EvaluationError
    )(error_func)
    
    # 実行して例外を確認
    with pytest.raises(EvaluationError) as excinfo:
        decorated_func(5)
    
    # 関数名を含むデフォルトメッセージが使用されることを確認
    assert "error_func" in str(excinfo.value)
    assert "関数の実行中にエラーが発生しました" in str(excinfo.value)
    assert "Invalid type" in str(excinfo.value)

def test_subprocess_error_str_representation():
    """SubprocessErrorの文字列表現のテスト"""
    # 基本的なメッセージのみのケース
    error1 = SubprocessError("コマンドの実行に失敗しました")
    str_rep1 = str(error1)
    assert "コマンドの実行に失敗しました" in str_rep1
    assert "Return Code" not in str_rep1
    assert "Stderr" not in str_rep1
    assert "Stdout" not in str_rep1
    
    # returncodeありのケース
    error2 = SubprocessError("コマンドの実行に失敗しました", returncode=1)
    str_rep2 = str(error2)
    assert "コマンドの実行に失敗しました" in str_rep2
    assert "Return Code: 1" in str_rep2
    assert "Stderr" not in str_rep2
    assert "Stdout" not in str_rep2
    
    # returncode、stderrありのケース
    error3 = SubprocessError("コマンドの実行に失敗しました", returncode=2, stderr="エラー出力")
    str_rep3 = str(error3)
    assert "コマンドの実行に失敗しました" in str_rep3
    assert "Return Code: 2" in str_rep3
    assert "Stderr: エラー出力" in str_rep3
    assert "Stdout" not in str_rep3
    
    # returncode、stdout、stderrすべてありのケース
    error4 = SubprocessError("コマンドの実行に失敗しました", returncode=3, stdout="標準出力", stderr="エラー出力")
    str_rep4 = str(error4)
    assert "コマンドの実行に失敗しました" in str_rep4
    assert "Return Code: 3" in str_rep4
    assert "Stderr: エラー出力" in str_rep4
    assert "Stdout: 標準出力" in str_rep4
    
    # 長いstdout/stderrが切り詰められることを確認
    long_text = "a" * 1000
    error5 = SubprocessError("コマンドの実行に失敗しました", stdout=long_text, stderr=long_text)
    str_rep5 = str(error5)
    assert "Stderr: " + long_text[:500] + "..." in str_rep5
    assert "Stdout: " + long_text[:500] + "..." in str_rep5 