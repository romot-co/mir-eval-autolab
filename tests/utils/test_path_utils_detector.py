"""
path_utils.py の検出器パス関連関数のテスト
"""

import pytest
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from src.utils import path_utils
from src.utils.exception_utils import FileError, ConfigError

# --- 検出器パス関連の関数のテスト ---

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
def setup_detector_dirs(tmp_path, monkeypatch, reset_env_vars):
    """検出器関連ディレクトリの構築"""
    # プロジェクトルート構造
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    
    # src/detectors ディレクトリ
    src_dir = project_root / "src"
    src_dir.mkdir()
    detectors_dir = src_dir / "detectors"
    detectors_dir.mkdir()
    
    # ベース検出器ファイル
    test_detector_file = detectors_dir / "TestDetector.py"
    test_detector_file.write_text("# Test detector base file")
    
    lower_detector_file = detectors_dir / "lowerdetector.py"
    lower_detector_file.write_text("# Lower case detector file")
    
    # ワークスペースディレクトリ
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    
    # improved_versions ディレクトリ
    improved_dir = workspace_dir / "improved_versions"
    improved_dir.mkdir()
    
    # 改善版検出器ファイル
    improved_file1 = improved_dir / "TestDetector_v1.py"
    improved_file1.write_text("# Improved version 1")
    
    improved_file2 = improved_dir / "TestDetector_v2.py"
    improved_file2.write_text("# Improved version 2")
    
    # キャッシュをクリア
    monkeypatch.setattr(path_utils, "_project_root", None)
    
    # 関数を完全に置き換えるパッチ
    monkeypatch.setattr(path_utils, "get_project_root", lambda: project_root)
    monkeypatch.setattr(path_utils, "get_workspace_dir", lambda: workspace_dir)
    monkeypatch.setattr(path_utils, "get_improved_versions_dir", lambda: improved_dir)
    monkeypatch.setattr(path_utils, "get_detectors_src_dir", lambda: detectors_dir)
    
    return {
        "project_root": project_root,
        "src_detectors": detectors_dir,
        "improved_dir": improved_dir,
        "base_detector": test_detector_file,
        "lower_detector": lower_detector_file,
        "improved_v1": improved_file1,
        "improved_v2": improved_file2
    }

def test_get_detector_path_base(setup_detector_dirs):
    """ベース検出器ファイルを取得するテスト"""
    # 改善版を指定しない場合はベース検出器を返す
    # 最新の改善版が検出されないようにタイムスタンプを調整
    # テスト時は外部の影響を受けないよう、関数呼び出し内部を個別にパッチ
    with patch('pathlib.Path.glob') as mock_glob:
        mock_glob.return_value = []  # 改善版ファイルが見つからないケース
        
        detector_path = path_utils.get_detector_path("TestDetector")
        assert detector_path == setup_detector_dirs["base_detector"]

def test_get_detector_path_specific_version(setup_detector_dirs):
    """特定バージョンの検出器ファイルを取得するテスト"""
    detector_path = path_utils.get_detector_path("TestDetector", version="v1")
    assert detector_path == setup_detector_dirs["improved_v1"]

def test_get_detector_path_latest_version(setup_detector_dirs):
    """最新バージョンの検出器ファイルを取得するテスト"""
    # v2のタイムスタンプを最新にする
    now = time.time()
    os.utime(setup_detector_dirs["improved_v1"], (now - 100, now - 100))
    os.utime(setup_detector_dirs["improved_v2"], (now, now))
    
    with patch('pathlib.Path.glob') as mock_glob:
        # globの戻り値として両方のファイルを返す
        mock_glob.return_value = [
            setup_detector_dirs["improved_v1"],
            setup_detector_dirs["improved_v2"]
        ]
        
        # 最新の改善版(v2)が取得できるはず
        detector_path = path_utils.get_detector_path("TestDetector")
        assert detector_path == setup_detector_dirs["improved_v2"]

def test_get_detector_path_lower_case(setup_detector_dirs):
    """小文字のファイル名でも検出できることを確認するテスト"""
    with patch('pathlib.Path.glob') as mock_glob:
        mock_glob.return_value = []  # 改善版ファイルが見つからないケース
        
        detector_path = path_utils.get_detector_path("lowerdetector")
        assert detector_path == setup_detector_dirs["lower_detector"]

def test_get_detector_path_invalid_detector_name():
    """無効な検出器名に対するテスト"""
    with pytest.raises(ValueError, match="検出器名に使用できない文字が含まれています"):
        path_utils.get_detector_path("invalid/detector")

def test_get_detector_path_invalid_version():
    """無効なバージョン名に対するテスト"""
    with pytest.raises(ValueError, match="バージョン名に使用できない文字が含まれています"):
        path_utils.get_detector_path("TestDetector", version="v1/invalid")

def test_get_detector_path_version_not_found(setup_detector_dirs):
    """存在しないバージョンを指定した場合のテスト"""
    with pytest.raises(FileNotFoundError, match="指定されたバージョンの検出器が見つかりません"):
        path_utils.get_detector_path("TestDetector", version="v999")

def test_get_detector_path_detector_not_found(setup_detector_dirs):
    """存在しない検出器を指定した場合のテスト"""
    with patch('pathlib.Path.glob') as mock_glob:
        mock_glob.return_value = []  # 改善版ファイルが見つからないケース
        
        with patch('pathlib.Path.is_file') as mock_is_file:
            mock_is_file.return_value = False  # ファイルが存在しないケース
            
            with pytest.raises(FileNotFoundError, match="検出器ファイルが見つかりませんでした"):
                path_utils.get_detector_path("NonExistentDetector")

def test_get_detector_path_improved_dir_error_handling(setup_detector_dirs):
    """改善版ディレクトリの読み取りエラーハンドリングテスト"""
    # globではファイルが見つかるが、個別ファイルのstatでエラーが発生するシナリオ
    with patch('src.utils.path_utils.get_improved_versions_dir') as mock_improved_dir:
        mock_improved_dir.return_value = Path(setup_detector_dirs["project_root"]) / "workspace" / "improved_versions"
        
        # is_dir()はTrueを返すようにパッチ
        with patch.object(Path, 'is_dir', return_value=True):
            with patch('pathlib.Path.glob') as mock_glob:
                # 検出器ファイルを返すようモック
                mock_glob.return_value = [setup_detector_dirs["improved_v1"]]
                
                # globで見つかった個別ファイルに対するstatメソッドのみがエラーを返すようにする
                # pathlib.Pathのstatメソッドをモック化する関数を定義
                original_stat = Path.stat
                
                def mock_stat_method(self):
                    # ファイルパスが改善バージョンのファイルであればエラーを発生させる
                    if str(self) == str(setup_detector_dirs["improved_v1"]):
                        raise OSError("特定ファイルの読み取りエラー")
                    # それ以外は元のstat()メソッドを呼び出す
                    return original_stat(self)
                
                # statメソッドをパッチ
                with patch.object(Path, 'stat', new=mock_stat_method):
                    with patch("src.utils.path_utils.logger") as mock_logger:
                        # エラーが発生しても、最後にベースの検出器を取得できれば成功
                        with patch('src.utils.path_utils.get_detectors_src_dir') as mock_detectors_src:
                            mock_detectors_src.return_value = Path(setup_detector_dirs["project_root"]) / "src" / "detectors"
                            with patch('pathlib.Path.is_file', return_value=True):
                                detector_path = path_utils.get_detector_path("TestDetector")
                                
                                # ベースのパスが返されることを確認
                                assert detector_path
                                assert detector_path.name == "TestDetector.py"
                                # 警告ログが出力されることを確認
                                mock_logger.warning.assert_called_once()
                                assert "改善版ファイルの最終更新日時取得エラー" in mock_logger.warning.call_args[0][0]

def test_get_detector_path_no_improved_dir(setup_detector_dirs):
    """改善版ディレクトリが存在しない場合のテスト"""
    # 改善版ディレクトリが存在しないことをシミュレート
    non_existent_dir = Path(setup_detector_dirs["project_root"]) / "non_existent"
    
    with patch('src.utils.path_utils.get_improved_versions_dir') as mock_get_improved_dir:
        mock_get_improved_dir.return_value = non_existent_dir
    
        # is_dirをパッチして、特定のパスに対してFalseを返すように設定
        original_is_dir = Path.is_dir
        
        def mocked_is_dir(self):
            if str(self) == str(non_existent_dir):
                return False
            return original_is_dir(self)
        
        with patch.object(Path, 'is_dir', new=mocked_is_dir):
            # ベースの検出器を取得できれば成功
            with patch('src.utils.path_utils.get_detectors_src_dir') as mock_detectors_src:
                mock_detectors_src.return_value = Path(setup_detector_dirs["project_root"]) / "src" / "detectors"
                with patch('pathlib.Path.is_file', return_value=True):
                    detector_path = path_utils.get_detector_path("TestDetector")
                    
                    # ベースのパスが返されることを確認
                    assert detector_path
                    assert detector_path.name == "TestDetector.py" 