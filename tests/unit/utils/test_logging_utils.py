import logging
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.utils.logging_utils import setup_logger, get_logger


# ロガーのテスト前後でリセットするための関数
def reset_logger(name):
    """テスト用にロガーをリセットする"""
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)  # デフォルトレベルに戻す
    logger.propagate = False  # 親ロガーへの伝播を無効化


@pytest.fixture
def cleanup_loggers():
    """すべてのテスト前後でロガーの状態をクリーンアップするフィクスチャ"""
    # テスト前のクリーンアップ
    yield
    # テスト後のクリーンアップ（ロギング関連のテストではすべてのロガーをリセット）
    for log_name in list(logging.Logger.manager.loggerDict.keys()):
        if log_name.startswith("test_"):
            reset_logger(log_name)


def test_setup_logger_basic_config(cleanup_loggers):
    """基本的なロガー設定を確認する"""
    logger_name = "test_setup_logger"
    reset_logger(logger_name)  # テスト実行前にロガーをリセット
    logger = setup_logger(logger_name)

    assert logger.name == logger_name
    assert logger.level == logging.NOTSET  # ロガー自体のレベルはNOTSET
    assert logger.propagate is True

    # 実装ではhandlersが追加されるかどうかはハンドラがすでに存在するかに依存
    # hasHandlersがTrueを返す場合、新しいハンドラは追加されない
    assert logger.hasHandlers() == True

    # コンソールハンドラが追加されたことを確認
    console_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(console_handlers) == 1
    assert console_handlers[0].level == logging.INFO  # デフォルトレベル


def test_setup_logger_custom_level(cleanup_loggers):
    """カスタムログレベルがハンドラに設定されることを確認する"""
    logger_name = "test_setup_logger_custom"
    reset_logger(logger_name)  # 前のテストの影響を確実に排除
    logger = setup_logger(logger_name, level=logging.DEBUG)

    assert logger.level == logging.NOTSET

    # コンソールハンドラのレベルを確認
    console_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(console_handlers) == 1
    assert console_handlers[0].level == logging.DEBUG


def test_setup_logger_separate_levels(cleanup_loggers):
    """コンソールとファイルで異なるレベルを設定できることを確認する"""
    logger_name = "test_setup_logger_levels"
    reset_logger(logger_name)  # 前のテストの影響を確実に排除
    logger = setup_logger(
        logger_name, level=logging.INFO, console_level=logging.WARNING
    )

    # コンソールハンドラのレベルを確認
    console_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert (
        len(console_handlers) == 1
    )  # file_levelがあってもoutput_dirがない場合はファイルハンドラは追加されない
    assert console_handlers[0].level == logging.WARNING


def test_setup_logger_with_file_output(tmp_path: Path, cleanup_loggers):
    """ファイル出力が有効な場合にファイルハンドラが追加されることを確認する"""
    logger_name = "file_log"
    reset_logger(logger_name)  # 確実にリセット
    output_dir = tmp_path / "logs"

    logger = setup_logger(
        logger_name,
        output_dir=str(output_dir),
        level=logging.INFO,
        file_level=logging.DEBUG,
    )

    # ハンドラの数を確認（コンソールとファイル）
    handlers = logger.handlers
    assert any(
        isinstance(h, logging.FileHandler) for h in handlers
    ), "ファイルハンドラが追加されていません"

    # ファイルハンドラのレベルを確認
    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].level == logging.DEBUG

    # コンソールハンドラも確認
    console_handlers = [
        h
        for h in handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(console_handlers) == 1
    assert console_handlers[0].level == logging.INFO


def test_setup_logger_idempotency(cleanup_loggers):
    """ロガーが既に設定されている場合は、設定を変更せずに返すことを確認する"""
    logger_name = "test_setup_idempotent"
    reset_logger(logger_name)  # 確実にリセット

    # 最初に設定
    logger1 = setup_logger(logger_name, level=logging.WARNING)
    console_handlers1 = [
        h for h in logger1.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(console_handlers1) == 1
    assert console_handlers1[0].level == logging.WARNING

    # 初期ハンドラ数を記録
    initial_handler_count = len(logger1.handlers)

    # 再度設定（異なるレベル）
    logger2 = setup_logger(logger_name, level=logging.DEBUG)

    # 同じロガーが返されるはず
    assert logger1 is logger2

    # hasHandlersがTrueを返すので、ハンドラの数は変わらないはず
    assert len(logger2.handlers) == initial_handler_count

    # レベルも変更されないはず
    console_handlers2 = [
        h for h in logger2.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(console_handlers2) == 1
    assert console_handlers2[0].level == logging.WARNING  # 最初に設定したレベルのまま


def test_setup_logger_invalid_output_dir(caplog, cleanup_loggers):
    """無効な出力ディレクトリが指定された場合にエラーログが出ることを確認する"""
    logger_name = "test_invalid_dir"
    invalid_dir = "/non_existent_dir_for_sure/logs"  # 通常書き込めないパス
    reset_logger(logger_name)  # 確実にリセット

    with caplog.at_level(logging.ERROR):
        logger = setup_logger(logger_name, output_dir=invalid_dir)

    # コンソールハンドラだけは追加されるはず
    console_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(console_handlers) == 1

    # ファイルハンドラのエラーメッセージが出力されることを確認
    assert any("Failed to set up file handler" in msg for msg in caplog.messages)


def test_get_logger_new(cleanup_loggers):
    """新しいロガーを作成してハンドラが設定されることを確認する"""
    logger_name = "test_get_new"
    reset_logger(logger_name)  # 確実にリセット

    logger = get_logger(logger_name, log_level=logging.DEBUG)

    assert logger.name == logger_name
    assert logger.level == logging.DEBUG

    # ハンドラが設定されているか確認
    assert logger.hasHandlers()
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_get_logger_existing(cleanup_loggers):
    """既存のロガーを取得し、設定が変更されないことを確認する"""
    logger_name = "test_get_existing"
    reset_logger(logger_name)  # 確実にリセット

    # 最初に setup_logger で設定しておく
    initial_logger = setup_logger(logger_name, level=logging.WARNING)
    initial_handlers = list(initial_logger.handlers)

    # get_logger で取得
    logger = get_logger(logger_name, log_level=logging.DEBUG)  # 異なるログレベル

    # 同じロガーインスタンスが返される
    assert logger is initial_logger

    # レベルや構成は変更されていないはず
    console_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(console_handlers) == 1
    assert console_handlers[0].level == logging.WARNING  # setup_loggerで設定したレベル


def test_get_logger_existing_no_handlers(cleanup_loggers):
    """ハンドラのない既存のロガーに対して、ハンドラが追加されることを確認する"""
    logger_name = "test_get_empty"
    reset_logger(logger_name)  # 確実にリセット

    # 素のロガーを取得し、レベルだけ設定しておく
    empty_logger = logging.getLogger(logger_name)
    empty_logger.setLevel(logging.INFO)
    assert len(empty_logger.handlers) == 0  # ハンドラはない

    # get_logger で取得すると、ハンドラが追加される
    logger = get_logger(logger_name, log_level=logging.DEBUG)

    # 同じロガーインスタンスが返される
    assert logger is empty_logger

    # ハンドラが追加されている
    assert logger.hasHandlers()
    console_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(console_handlers) == 1
    assert console_handlers[0].level == logging.DEBUG  # get_loggerで指定したレベル
