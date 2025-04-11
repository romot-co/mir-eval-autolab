import logging
import os
from pathlib import Path
import pytest

from src.utils.logging_utils import setup_logger, get_logger

# 各テスト後にロガーの状態をリセットするためのヘルパー関数
def reset_logger(logger_name: str):
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    # Logger Dictionaryからも削除 (必須ではないが、よりクリーンにする場合)
    if logger_name in logging.Logger.manager.loggerDict:
         del logging.Logger.manager.loggerDict[logger_name]
    # propagateの状態もリセット（デフォルトに戻す）
    logger.propagate = True
    # レベルもリセット (NOTSET)
    logger.setLevel(logging.NOTSET)


@pytest.fixture(autouse=True)
def cleanup_loggers():
    """テスト実行後に、テストで生成された可能性のあるロガーをクリーンアップするフィクスチャ"""
    yield # テスト実行
    # テストで使用した可能性のあるロガー名をリストアップ
    test_logger_names = ["test_setup_logger", "test_get_logger", "another_logger", "file_log"]
    for name in test_logger_names:
        reset_logger(name)
    # ルートロガーのハンドラもクリア（他のテストへの影響を防ぐため）
    # root_logger = logging.getLogger()
    # for handler in root_logger.handlers[:]:
    #     root_logger.removeHandler(handler)
    #     handler.close()


# --- setup_logger Tests ---

def test_setup_logger_basic_config():
    """基本的なロガー設定を確認する"""
    logger_name = "test_setup_logger"
    reset_logger(logger_name) # 事前にクリア
    logger = setup_logger(logger_name)

    assert logger.name == logger_name
    assert logger.level == logging.NOTSET # ロガー自体のレベルはNOTSET
    assert logger.propagate is True
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].level == logging.INFO # デフォルトハンドラレベル
    assert isinstance(logger.handlers[0].formatter, logging.Formatter)

    # 後片付け
    reset_logger(logger_name)

def test_setup_logger_custom_level():
    """カスタムログレベルがハンドラに設定されることを確認する"""
    logger_name = "test_setup_logger"
    reset_logger(logger_name)
    logger = setup_logger(logger_name, level=logging.DEBUG)

    assert logger.handlers[0].level == logging.DEBUG
    reset_logger(logger_name)

def test_setup_logger_separate_levels():
    """コンソールとファイルで異なるレベルを設定できることを確認する"""
    logger_name = "test_setup_logger"
    reset_logger(logger_name)
    logger = setup_logger(logger_name, level=logging.INFO, console_level=logging.WARNING, file_level=logging.DEBUG)

    assert len(logger.handlers) == 1 # file_levelがあってもoutput_dirがない場合はファイルハンドラは追加されない
    assert logger.handlers[0].level == logging.WARNING # console_levelが優先される
    reset_logger(logger_name)

def test_setup_logger_with_file_output(tmp_path: Path):
    """ファイル出力が有効な場合にファイルハンドラが追加されることを確認する"""
    logger_name = "file_log"
    output_dir = tmp_path / "logs"
    reset_logger(logger_name)

    logger = setup_logger(logger_name, output_dir=str(output_dir), level=logging.INFO, file_level=logging.DEBUG)

    assert len(logger.handlers) == 2
    console_handler = next(h for h in logger.handlers if isinstance(h, logging.StreamHandler))
    file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))

    assert console_handler.level == logging.INFO
    assert file_handler.level == logging.DEBUG
    assert file_handler.baseFilename.startswith(str(output_dir / logger_name))
    assert os.path.exists(output_dir) # ディレクトリが作成されているか

    # ログファイルに書き込めるか簡単なテスト
    test_message = "This is a test message."
    logger.debug(test_message) # file_handler のレベルは DEBUG なので書き込まれるはず
    logger.info("Info message") # console/file 両方に書き込まれるはず

    file_handler.flush()
    file_handler.close() # 明示的に閉じる

    # ログファイルの内容を確認 (ファイルが1つだけのはず)
    log_files = list(output_dir.glob(f"{logger_name}*.log"))
    assert len(log_files) == 1
    log_content = log_files[0].read_text(encoding='utf-8')
    assert test_message in log_content
    assert "Info message" in log_content

    reset_logger(logger_name)


def test_setup_logger_idempotency():
    """ロガーが既に設定されている場合は、設定を変更せずに返すことを確認する"""
    logger_name = "test_setup_logger"
    reset_logger(logger_name)

    # 最初に設定
    logger1 = setup_logger(logger_name, level=logging.WARNING)
    initial_handlers = logger1.handlers[:] # ハンドラのリストをコピー
    initial_level = logger1.handlers[0].level

    # 再度呼び出す (異なるレベルを指定してみる)
    logger2 = setup_logger(logger_name, level=logging.DEBUG)

    assert logger2 is logger1 # 同じロガーインスタンスが返る
    assert logger2.handlers == initial_handlers # ハンドラは変更されていない
    assert logger2.handlers[0].level == initial_level # ハンドラのレベルも変更されていない

    reset_logger(logger_name)

def test_setup_logger_invalid_output_dir(caplog):
    """無効な出力ディレクトリが指定された場合にエラーログが出ることを確認する"""
    logger_name = "test_setup_logger"
    invalid_dir = "/non_existent_dir_for_sure/logs" # 通常書き込めないパス
    reset_logger(logger_name)

    with caplog.at_level(logging.ERROR):
         logger = setup_logger(logger_name, output_dir=invalid_dir)

    assert len(logger.handlers) == 1 # コンソールハンドラのみ
    assert any(f"Failed to set up file handler for logger '{logger_name}'" in record.message for record in caplog.records)

    reset_logger(logger_name)


# --- get_logger Tests ---

def test_get_logger_new():
    """新しいロガーを取得できることを確認する"""
    logger_name = "test_get_logger"
    reset_logger(logger_name)
    logger = get_logger(logger_name, log_level=logging.DEBUG)

    assert logger.name == logger_name
    assert logger.level == logging.DEBUG # get_logger はロガー自体のレベルを設定する
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].level == logging.DEBUG
    assert isinstance(logger.handlers[0].formatter, logging.Formatter)
    # get_logger は propagate を変更しないはず (デフォルトは True)
    assert logger.propagate is True

    reset_logger(logger_name)

def test_get_logger_existing():
    """既存のロガーを取得し、設定が変更されないことを確認する"""
    logger_name = "test_get_logger"
    reset_logger(logger_name)

    # 最初に setup_logger で設定しておく
    initial_logger = setup_logger(logger_name, level=logging.WARNING)
    initial_handlers = initial_logger.handlers[:]
    # setup_logger はロガー自体のレベルを設定しないので NOTSET
    initial_logger_level = initial_logger.level

    # get_logger を呼び出す
    retrieved_logger = get_logger(logger_name, log_level=logging.INFO) # 異なるレベルを指定

    assert retrieved_logger is initial_logger
    # get_logger は既存ハンドラがあると何もしないので、レベルやハンドラは変わらないはず
    assert retrieved_logger.level == initial_logger_level # NOTSET のままのはず
    assert retrieved_logger.handlers == initial_handlers
    assert retrieved_logger.handlers[0].level == logging.WARNING # setup_loggerで設定されたレベルのまま

    reset_logger(logger_name)

def test_get_logger_existing_no_handlers():
    """ハンドラがない既存ロガーを取得した場合に設定が行われることを確認する"""
    logger_name = "another_logger"
    reset_logger(logger_name)

    # ハンドラなしのロガーを事前に取得
    existing_logger = logging.getLogger(logger_name)
    assert not existing_logger.hasHandlers()
    existing_logger.setLevel(logging.CRITICAL) # 事前にレベルだけ設定しておく

    # get_logger を呼び出す
    retrieved_logger = get_logger(logger_name, log_level=logging.ERROR)

    assert retrieved_logger is existing_logger
    # ハンドラがなかったので、get_logger が設定を行う
    assert retrieved_logger.level == logging.ERROR # get_logger で指定したレベルに設定される
    assert len(retrieved_logger.handlers) == 1
    assert isinstance(retrieved_logger.handlers[0], logging.StreamHandler)
    assert retrieved_logger.handlers[0].level == logging.ERROR

    reset_logger(logger_name) 