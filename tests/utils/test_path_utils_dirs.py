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
    
    return {
        "project_root": project_root,
        "workspace_dir": workspace_dir,
        "datasets_dir": datasets_dir,
        "synth_dir": synth_dir,
        "audio_dir": audio_dir,
        "label_dir": label_dir,
        "config": {}
    }

def test_get_db_dir_default(mock_paths_for_test):
    """get_db_dir 関数のデフォルト動作テスト"""
    with patch("src.utils.path_utils.get_workspace_dir", return_value=mock_paths_for_test["workspace_dir"]):
        db_dir = path_utils.get_db_dir(config=mock_paths_for_test["config"])
        expected_dir = mock_paths_for_test["workspace_dir"] / "db"
        assert db_dir == expected_dir
        assert db_dir.exists()  # ensure_dirでディレクトリが作成される

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