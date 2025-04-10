"""
path_utils.py のディレクトリ取得関数に対するテスト
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils import path_utils
from src.utils.exception_utils import FileError, ConfigError

# --- ディレクトリ取得関数のテスト ---

@pytest.fixture
def reset_env_vars(monkeypatch):
    """環境変数をリセット"""
    vars_to_unset = [
        'MIREX_OUTPUT_DIR',
        'MIREX_GRID_SEARCH_DIR',
        'MIREX_VERSIONS_DIR',
        'MIREX_SESSION_DIR',
        'MCP_STATE_DIR',
        'MIREX_DATASETS_DIR',
        'MIREX_AUDIO_DIR',
        'MIREX_LABEL_DIR'
    ]
    for var in vars_to_unset:
        monkeypatch.delenv(var, raising=False)
    return vars_to_unset

@pytest.fixture
def mock_paths_for_test(tmp_path, monkeypatch, reset_env_vars):
    """テスト用のパス設定をまとめたフィクスチャー"""
    # プロジェクトルートとワークスペースの設定
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    
    # データセット関連ディレクトリ
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir()
    
    synth_dir = datasets_dir / "synthesized"
    synth_dir.mkdir()
    
    audio_dir = synth_dir / "audio"
    audio_dir.mkdir()
    
    label_dir = synth_dir / "labels"
    label_dir.mkdir()
    
    # モック設定
    # キャッシュの初期化
    monkeypatch.setattr(path_utils, "_project_root", None)
    
    # 関数のモック
    monkeypatch.setattr(path_utils, "get_project_root", lambda: project_root)
    monkeypatch.setattr(path_utils, "get_workspace_dir", lambda: workspace_dir)
    monkeypatch.setattr(path_utils, "get_datasets_dir", lambda: datasets_dir)
    monkeypatch.setattr(path_utils, "get_synthesized_datasets_dir", lambda: synth_dir)
    
    return {
        "project_root": project_root,
        "workspace_dir": workspace_dir,
        "datasets_dir": datasets_dir,
        "synth_dir": synth_dir,
        "audio_dir": audio_dir,
        "label_dir": label_dir
    }

def test_get_evaluation_results_dir_default(mock_paths_for_test):
    """get_evaluation_results_dir 関数のデフォルト動作テスト"""
    # 直接関数をモックしたのでパッチで内部的なget_workspace_dirの呼び出しをオーバーライド
    with patch("src.utils.path_utils.get_workspace_dir", return_value=mock_paths_for_test["workspace_dir"]):
        results_dir = path_utils.get_evaluation_results_dir()
        expected_dir = mock_paths_for_test["workspace_dir"] / "evaluation_results"
        assert results_dir == expected_dir
        assert results_dir.exists()  # ensure_dirでディレクトリが作成される

def test_get_evaluation_results_dir_env_var(mock_paths_for_test, monkeypatch):
    """環境変数 MIREX_OUTPUT_DIR が設定されている場合の get_evaluation_results_dir のテスト"""
    custom_dir = mock_paths_for_test["workspace_dir"] / "custom_results"
    custom_dir.mkdir()
    monkeypatch.setenv("MIREX_OUTPUT_DIR", str(custom_dir))
    
    results_dir = path_utils.get_evaluation_results_dir()
    assert results_dir == custom_dir

def test_get_grid_search_results_dir_default(mock_paths_for_test):
    """get_grid_search_results_dir 関数のデフォルト動作テスト"""
    with patch("src.utils.path_utils.get_workspace_dir", return_value=mock_paths_for_test["workspace_dir"]):
        results_dir = path_utils.get_grid_search_results_dir()
        expected_dir = mock_paths_for_test["workspace_dir"] / "grid_search_results"
        assert results_dir == expected_dir
        assert results_dir.exists()  # ensure_dirでディレクトリが作成される

def test_get_grid_search_results_dir_env_var(mock_paths_for_test, monkeypatch):
    """環境変数 MIREX_GRID_SEARCH_DIR が設定されている場合のテスト"""
    custom_dir = mock_paths_for_test["workspace_dir"] / "custom_grid_search"
    custom_dir.mkdir()
    monkeypatch.setenv("MIREX_GRID_SEARCH_DIR", str(custom_dir))
    
    results_dir = path_utils.get_grid_search_results_dir()
    assert results_dir == custom_dir

def test_get_improved_versions_dir_default(mock_paths_for_test):
    """get_improved_versions_dir 関数のデフォルト動作テスト"""
    with patch("src.utils.path_utils.get_workspace_dir", return_value=mock_paths_for_test["workspace_dir"]):
        versions_dir = path_utils.get_improved_versions_dir()
        expected_dir = mock_paths_for_test["workspace_dir"] / "improved_versions"
        assert versions_dir == expected_dir
        assert versions_dir.exists()  # ensure_dirでディレクトリが作成される

def test_get_improved_versions_dir_env_var(mock_paths_for_test, monkeypatch):
    """環境変数 MIREX_VERSIONS_DIR が設定されている場合のテスト"""
    custom_dir = mock_paths_for_test["workspace_dir"] / "custom_versions"
    custom_dir.mkdir()
    monkeypatch.setenv("MIREX_VERSIONS_DIR", str(custom_dir))
    
    versions_dir = path_utils.get_improved_versions_dir()
    assert versions_dir == custom_dir

def test_get_session_dir_default(mock_paths_for_test):
    """get_session_dir 関数のデフォルト動作テスト"""
    session_id = "test_session_123"
    
    with patch("src.utils.path_utils.get_workspace_dir", return_value=mock_paths_for_test["workspace_dir"]):
        session_dir = path_utils.get_session_dir(session_id)
        expected_dir = mock_paths_for_test["workspace_dir"] / "sessions" / session_id
        assert session_dir == expected_dir
        assert session_dir.exists()  # ensure_dirでディレクトリが作成される

def test_get_session_dir_invalid_id():
    """不正なセッションID で get_session_dir を呼び出した場合のテスト"""
    with pytest.raises(ValueError, match="セッションIDに使用できない文字が含まれています"):
        path_utils.get_session_dir("invalid/session")

def test_get_session_dir_env_var(mock_paths_for_test, monkeypatch):
    """環境変数 MIREX_SESSION_DIR が設定されている場合のテスト"""
    session_base = mock_paths_for_test["workspace_dir"] / "custom_sessions"
    session_base.mkdir()
    monkeypatch.setenv("MIREX_SESSION_DIR", str(session_base))
    
    session_id = "test_session_456"
    session_dir = path_utils.get_session_dir(session_id)
    expected_dir = session_base / session_id
    assert session_dir == expected_dir
    assert session_dir.exists()  # ensure_dirでディレクトリが作成される

def test_get_state_dir_default(mock_paths_for_test):
    """get_state_dir 関数のデフォルト動作テスト"""
    with patch("src.utils.path_utils.get_workspace_dir", return_value=mock_paths_for_test["workspace_dir"]):
        state_dir = path_utils.get_state_dir()
        expected_dir = mock_paths_for_test["workspace_dir"] / "improvement_states"
        assert state_dir == expected_dir
        assert state_dir.exists()  # ensure_dirでディレクトリが作成される

def test_get_state_dir_env_var(mock_paths_for_test, monkeypatch):
    """環境変数 MCP_STATE_DIR が設定されている場合のテスト"""
    custom_dir = mock_paths_for_test["workspace_dir"] / "custom_states"
    custom_dir.mkdir()
    monkeypatch.setenv("MCP_STATE_DIR", str(custom_dir))
    
    state_dir = path_utils.get_state_dir()
    assert state_dir == custom_dir

def test_get_db_dir_default(mock_paths_for_test):
    """get_db_dir 関数のデフォルト動作テスト"""
    with patch("src.utils.path_utils.get_workspace_dir", return_value=mock_paths_for_test["workspace_dir"]):
        db_dir = path_utils.get_db_dir()
        expected_dir = mock_paths_for_test["workspace_dir"] / "db"
        assert db_dir == expected_dir
        assert db_dir.exists()  # ensure_dirでディレクトリが作成される

def test_get_audio_dir_default(mock_paths_for_test):
    """get_audio_dir 関数のデフォルト動作テスト"""
    # モック関数を直接上書き
    with patch("src.utils.path_utils.get_synthesized_datasets_dir", return_value=mock_paths_for_test["synth_dir"]):
        audio_dir = path_utils.get_audio_dir()
        expected_dir = mock_paths_for_test["synth_dir"] / "audio"
        assert audio_dir == expected_dir

def test_get_audio_dir_env_var(mock_paths_for_test, monkeypatch):
    """環境変数 MIREX_AUDIO_DIR が設定されている場合のテスト"""
    custom_dir = mock_paths_for_test["workspace_dir"] / "custom_audio"
    custom_dir.mkdir()
    monkeypatch.setenv("MIREX_AUDIO_DIR", str(custom_dir))
    
    audio_dir = path_utils.get_audio_dir()
    assert audio_dir == custom_dir

def test_get_audio_dir_not_exists(mock_paths_for_test):
    """存在しない音声ディレクトリのテスト"""
    # 存在しないディレクトリを作成
    non_existent_dir = mock_paths_for_test["project_root"] / "non_existent" / "synthesized"
    
    with patch("src.utils.path_utils.get_synthesized_datasets_dir", return_value=non_existent_dir):
        with patch("src.utils.path_utils.logger") as mock_logger:
            audio_dir = path_utils.get_audio_dir()
            expected_dir = non_existent_dir / "audio"
            assert audio_dir == expected_dir
            mock_logger.warning.assert_called_once()

def test_get_label_dir_default(mock_paths_for_test):
    """get_label_dir 関数のデフォルト動作テスト"""
    with patch("src.utils.path_utils.get_synthesized_datasets_dir", return_value=mock_paths_for_test["synth_dir"]):
        label_dir = path_utils.get_label_dir()
        expected_dir = mock_paths_for_test["synth_dir"] / "labels"
        assert label_dir == expected_dir

def test_get_label_dir_env_var(mock_paths_for_test, monkeypatch):
    """環境変数 MIREX_LABEL_DIR が設定されている場合のテスト"""
    custom_dir = mock_paths_for_test["workspace_dir"] / "custom_labels"
    custom_dir.mkdir()
    monkeypatch.setenv("MIREX_LABEL_DIR", str(custom_dir))
    
    label_dir = path_utils.get_label_dir()
    assert label_dir == custom_dir

def test_get_label_dir_not_exists(mock_paths_for_test):
    """存在しないラベルディレクトリのテスト"""
    # 存在しないディレクトリを作成
    non_existent_dir = mock_paths_for_test["project_root"] / "non_existent" / "synthesized"
    
    with patch("src.utils.path_utils.get_synthesized_datasets_dir", return_value=non_existent_dir):
        with patch("src.utils.path_utils.logger") as mock_logger:
            label_dir = path_utils.get_label_dir()
            expected_dir = non_existent_dir / "labels"
            assert label_dir == expected_dir
            mock_logger.warning.assert_called_once()

def test_get_detectors_src_dir_default(mock_paths_for_test):
    """get_detectors_src_dir 関数のデフォルト動作テスト"""
    # srcディレクトリを作成
    src_dir = mock_paths_for_test["project_root"] / "src"
    src_dir.mkdir(exist_ok=True)
    detectors_dir = src_dir / "detectors"
    detectors_dir.mkdir(exist_ok=True)
    
    with patch("src.utils.path_utils.get_project_root", return_value=mock_paths_for_test["project_root"]):
        result = path_utils.get_detectors_src_dir()
        assert result == detectors_dir 