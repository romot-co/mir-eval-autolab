import pytest
import logging


# ロギング関連の問題を修正するためのフィクスチャ
@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """テスト実行中にロギングの問題が発生しないようにする"""
    # ロギングのルートハンドラをクリーンな状態にする
    root_logger = logging.getLogger()

    # ハンドラがすでに設定されている場合は、それを保持
    handlers = root_logger.handlers.copy()

    # テスト実行前の設定を行う（必要に応じて）
    yield

    # テスト終了後、ハンドラを復元する
    root_logger.handlers = handlers


# モッキングでの問題を防ぐためにロギング関数をモックする
@pytest.fixture
def mock_logger(monkeypatch):
    """ロギング関数をモックしてテストの問題を防ぐ"""
    mock_log = logging.getLogger("mock_logger")
    mock_log.setLevel(logging.DEBUG)

    # テスト用のNullHandlerを追加
    mock_log.addHandler(logging.NullHandler())

    # logging.getLoggerをモックして、常に同じロガーインスタンスを返すようにする
    def mock_get_logger(*args, **kwargs):
        return mock_log

    monkeypatch.setattr(logging, "getLogger", mock_get_logger)

    return mock_log
