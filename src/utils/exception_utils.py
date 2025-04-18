"""
例外処理を統一するためのユーティリティモジュール

このモジュールは、アプリケーション全体で例外処理を統一するためのユーティリティ関数と
例外クラスを提供します。アプリケーション固有の例外を定義し、一貫した方法で
エラーを処理するための機能を提供します。

利用例:
```python
from src.utils.exception_utils import log_exception, safe_execute, DetectionError

try:
    # 処理
    if problem:
        raise DetectionError("検出処理に失敗しました")
except Exception as e:
    log_exception(logger, e, "処理中にエラーが発生")
```

または安全に関数を実行する場合:
```python
result = safe_execute(
    func=my_function,
    logger=logger,
    error_msg="関数実行エラー",
    default_value={},
    param1="value1"
)
```
"""

import datetime
import logging
import time
import traceback
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import requests
import tenacity

T = TypeVar("T")


# カスタム例外クラス定義
class MirexError(Exception):
    """基本となるMIREX評価環境の例外クラス"""

    pass


class DetectionError(MirexError):
    """音検出中に発生するエラー"""

    pass


class EvaluationError(MirexError):
    """評価処理中に発生するエラー"""

    pass


class FileError(MirexError):
    """ファイル操作中に発生するエラー"""

    pass


class ConfigError(MirexError):
    """設定関連のエラー"""

    pass


class GridSearchError(MirexError):
    """グリッドサーチ処理中に発生するエラー"""

    pass


class VisualizationError(MirexError):
    """可視化処理中に発生するエラー"""

    pass


class LLMError(MirexError):
    """LLM API呼び出しや応答処理に関するエラー"""

    pass


class StateManagementError(MirexError):
    """セッションやジョブの状態管理（DBアクセスなど）に関するエラー"""

    pass


class SynthesizerError(MirexError):
    """音声合成処理中に発生するエラー"""

    pass


class SubprocessError(MirexError):
    """サブプロセス実行に関するエラー"""

    def __init__(
        self,
        message: str,
        returncode: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self):
        msg = super().__str__()
        if self.returncode is not None:
            msg += f" (Return Code: {self.returncode})"
        if self.stderr:
            msg += f"\nStderr: {self.stderr[:500]}..."  # 長すぎる場合に切り詰め
        if self.stdout:
            msg += f"\nStdout: {self.stdout[:500]}..."  # 長すぎる場合に切り詰め
        return msg


def log_exception(
    logger: logging.Logger,
    e: Exception,
    message: str = "エラーが発生しました",
    log_level: int = logging.ERROR,
) -> str:
    """
    例外をログに記録し、フォーマットされたエラーメッセージを返します。

    Parameters
    ----------
    logger : logging.Logger
        ロガーオブジェクト
    e : Exception
        発生した例外
    message : str, optional
        エラーメッセージのプレフィックス, by default "エラーが発生しました"
    log_level : int, optional
        ログレベル, by default logging.ERROR

    Returns
    -------
    str
        フォーマットされたエラーメッセージ（トレースバック情報を含む）
    """
    error_msg = f"{message}: {str(e)}"
    stack_trace = traceback.format_exc()

    if log_level == logging.DEBUG:
        logger.debug(f"{error_msg}\\n{stack_trace}", exc_info=True)
    elif log_level == logging.INFO:
        logger.info(f"{error_msg}\\n{stack_trace}", exc_info=True)
    elif log_level == logging.WARNING:
        logger.warning(f"{error_msg}\\n{stack_trace}", exc_info=True)
    elif log_level == logging.ERROR:
        logger.error(f"{error_msg}\\n{stack_trace}", exc_info=True)
    elif log_level == logging.CRITICAL:
        logger.critical(f"{error_msg}\\n{stack_trace}", exc_info=True)

    return f"{error_msg}\\n{stack_trace}"


def safe_execute(
    func: Callable[..., T],
    logger: logging.Logger,
    error_msg: str = "関数実行中にエラーが発生しました",
    default_value: Optional[T] = None,
    log_level: int = logging.ERROR,
    raise_exception: bool = False,
    **kwargs,
) -> Union[T, Optional[T]]:
    """
    関数を安全に実行し、例外が発生した場合はログに記録して既定値を返します。

    Parameters
    ----------
    func : Callable
        実行する関数
    logger : logging.Logger
        ロガーオブジェクト
    error_msg : str, optional
        エラーメッセージのプレフィックス, by default "関数実行中にエラーが発生しました"
    default_value : Any, optional
        例外発生時に返す値, by default None
    log_level : int, optional
        ログレベル, by default logging.ERROR
    raise_exception : bool, optional
        例外を再発生させるかどうか, by default False
    **kwargs
        関数に渡す引数

    Returns
    -------
    Any
        関数の実行結果、または例外発生時はdefault_value

    Raises
    ------
    Exception
        raise_exception=Trueの場合、発生した例外を再発生させます
    """
    try:
        return func(**kwargs)
    except Exception as e:
        log_exception(logger, e, error_msg, log_level)
        if raise_exception:
            raise
        return default_value


def create_error_result(
    error_msg: str, include_keys: List[str] = None
) -> Dict[str, Any]:
    """
    エラー時のデフォルト評価結果を生成します。

    Parameters
    ----------
    error_msg : str
        エラーメッセージ
    include_keys : List[str], optional
        結果に含めるキー, by default None

    Returns
    -------
    Dict[str, Any]
        エラー時のデフォルト評価結果
    """
    # 基本的な評価指標
    metrics = {"precision": 0.0, "recall": 0.0, "f_measure": 0.0}

    # エラー結果の構造
    result = {
        "precision": 0.0,
        "recall": 0.0,
        "f_measure": 0.0,
        "match_statistics": {
            "ref_note_count": 0,
            "est_note_count": 0,
            "onset_matches": 0,
            "offset_matches": 0,
            "pitch_matches": 0,
            "complete_matches": 0,
            "onset_errors": [],
            "offset_errors": [],
            "pitch_errors": [],
            "unmatched_ref_notes": 0,
            "unmatched_est_notes": 0,
            "average_onset_error": 0.0,
            "average_offset_error": 0.0,
            "average_pitch_error": 0.0,
            "max_onset_error": 0.0,
            "max_offset_error": 0.0,
            "max_pitch_error": 0.0,
            "onset_match_rate": 0.0,
            "offset_match_rate": 0.0,
            "pitch_match_rate": 0.0,
            "complete_match_rate": 0.0,
        },
        "error": error_msg,
        "valid": False,
    }

    # 追加のキーがある場合は追加
    if include_keys:
        for key in include_keys:
            if key not in result:
                result[key] = 0.0

    return result


def format_exception_message(e: Exception, context: str = "") -> str:
    """
    例外メッセージを整形します。

    Parameters
    ----------
    e : Exception
        例外オブジェクト
    context : str, optional
        追加のコンテキスト情報, by default ""

    Returns
    -------
    str
        整形されたエラーメッセージ
    """
    if context:
        return f"{context}: {str(e)}"
    return str(e)


def retry_on_exception(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    retry_exceptions: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]] = (
        requests.exceptions.RequestException,
        requests.Timeout,
        LLMError,  # LLM関連の一時的なエラーも対象とする場合
        TimeoutError,  # tenacity.RetryError が TimeoutError をラップすることがあるため
    ),
    logger: Optional[logging.Logger] = None,
    log_message_prefix: str = "リトライ試行",
) -> Callable:
    """
    指定された例外が発生した場合に関数をリトライするデコレータ。

    指数バックオフと最大リトライ回数を設定できます。

    Parameters
    ----------
    max_attempts : int, optional
        最大リトライ回数, by default 3
    initial_delay : float, optional
        最初のリトライまでの待機時間（秒）, by default 1.0
    backoff_factor : float, optional
        リトライごとに待機時間を乗算する係数, by default 2.0
    max_delay : float, optional
        最大待機時間（秒）, by default 10.0
    retry_exceptions : Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]], optional
        リトライ対象とする例外クラス（またはタプル）, by default (requests.exceptions.RequestException, requests.Timeout, LLMError, TimeoutError)
    logger : Optional[logging.Logger], optional
        リトライ試行をログ記録するためのロガー, by default None
    log_message_prefix : str, optional
        リトライ時のログメッセージの接頭辞, by default "リトライ試行"

    Returns
    -------
    Callable
        デコレータ関数

    Raises
    ------
    tenacity.RetryError
        最大リトライ回数に達しても例外が解消されなかった場合
    """

    def before_sleep_log(retry_state: tenacity.RetryCallState):
        """リトライ前にログを出力するコールバック関数"""
        if logger:
            exception = retry_state.outcome.exception()
            logger.warning(
                f"{log_message_prefix} {retry_state.attempt_number}/{max_attempts} 回目。"
                f" 例外: {type(exception).__name__}: {exception}."
                f" 次のリトライまで {retry_state.next_action.sleep:.2f} 秒待機します。"
            )

    # リトライ設定
    retry_config = tenacity.retry(
        stop=tenacity.stop_after_attempt(max_attempts),
        wait=tenacity.wait_exponential(
            multiplier=initial_delay, exp_base=backoff_factor, max=max_delay
        ),
        retry=(
            tenacity.retry_if_exception_type(retry_exceptions)
            if retry_exceptions
            else tenacity.retry_if_exception_type(Exception)
        ),
        before_sleep=before_sleep_log if logger else None,
        reraise=True,  # tenacity.RetryErrorを発生させる
    )

    # デコレータを適用
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        return retry_config(func)

    return decorator


def wrap_exceptions(
    target_exceptions: List[type],
    wrapper_exception: type,
    message: Optional[str] = None,
) -> Callable:
    """
    特定の例外を別の例外でラップするデコレータを作成します。

    Parameters
    ----------
    target_exceptions : List[type]
        ラップする例外クラスのリスト
    wrapper_exception : type
        ラップに使用する例外クラス
    message : Optional[str], optional
        例外にラップする際に使用するカスタムメッセージ, by default None

    Returns
    -------
    Callable
        作成されたデコレータ関数

    Examples
    --------
    ```python
    @wrap_exceptions([ValueError, TypeError], DetectionError, "検出処理でエラーが発生しました")
    def process_audio(audio_data):
        # 処理
        pass
    ```
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except tuple(target_exceptions) as e:
                error_message = (
                    message
                    if message
                    else f"{func.__name__}関数の実行中にエラーが発生しました"
                )
                raise wrapper_exception(f"{error_message}: {str(e)}") from e

        return wrapper

    return decorator
