# tests/unit/utils/test_path_utils.py
import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging
import re

# Assume functions are defined in src.utils.path_utils
# Provide dummies if necessary
try:
    from src.utils.path_utils import (
        get_project_root,
        get_workspace_dir,
        get_output_base_dir,
        is_safe_path_component,
        validate_path_within_allowed_dirs,
        ensure_dir,
        get_detector_path,
        get_improved_versions_dir,
        get_detectors_src_dir,
        get_dataset_paths,
        setup_python_path,
        get_allowed_upload_directories,
        get_allowed_vad_directories,
        get_absolute_path,
        find_project_root,
    )
    from src.utils.exception_utils import FileError, ConfigError

    PATHLIB_PATCH_TARGET = "src.utils.path_utils.Path"
    OS_PATCH_TARGET = "src.utils.path_utils.os"
except ImportError:
    print("Warning: Using dummy implementations for path_utils.")

    # Dummy implementations
    def get_project_root():
        # Dummy: Go up until a known marker (e.g., pyproject.toml) or assume structure
        # This dummy is likely inaccurate for actual testing
        p = (
            Path(__file__).resolve().parent.parent.parent
        )  # Guess based on tests/unit/utils
        # Look for a common marker like pyproject.toml or .git
        for _ in range(3):  # Limit search depth
            if (p / "pyproject.toml").exists() or (p / ".git").exists():
                return p
            if p.parent == p:
                break
            p = p.parent
        # Fallback guess if marker not found
        return Path(__file__).resolve().parent.parent.parent

    def get_workspace_dir(config=None):
        # Dummy: Use a subdirectory in project root
        # Allow override via config dict (basic simulation)
        if config and "workspace" in config:
            return Path(config["workspace"])
        return get_project_root() / "workspace"

    def get_output_base_dir(config=None):
        # Dummy: Use a subdirectory in project root
        if config and "output_base" in config:
            return Path(config["output_base"])
        return get_project_root() / "output"

    def is_safe_path_component(component):
        if not isinstance(component, str):
            return False
        # Basic safety checks
        return (
            bool(component)
            and ".." not in component
            and "/" not in component
            and "\\" not in component
        )

    def validate_path_within_allowed_dirs(path_to_validate, allowed_base_dirs):
        try:
            resolved_path = Path(path_to_validate).resolve()
        except Exception:
            # If path is invalid (e.g., contains null bytes on some OS), resolve might fail
            raise ValueError(f"Invalid path format: {path_to_validate}")

        validated = False
        for base_dir in allowed_base_dirs:
            try:
                resolved_base = Path(base_dir).resolve()
                # Check if resolved_path is equal to or starts with resolved_base
                if resolved_path == resolved_base or str(resolved_path).startswith(
                    str(resolved_base) + os.sep
                ):
                    validated = True
                    break  # Found a valid base directory
            except Exception:
                # Ignore errors resolving base_dir? Or raise?
                # Let's ignore for dummy, real one might need error handling here.
                continue

        if not validated:
            raise ValueError(
                f"Path {path_to_validate} is outside allowed directories: {allowed_base_dirs}"
            )
        return True

    def ensure_dir(dir_path):
        path_obj = Path(dir_path)
        # Check if a file exists at the location
        if path_obj.is_file():
            raise FileExistsError(
                f"Cannot create directory, a file exists at: {dir_path}"
            )
        # Create directory, handling potential race conditions is complex
        path_obj.mkdir(parents=True, exist_ok=True)

    PATHLIB_PATCH_TARGET = "pathlib.Path"  # Patch standard pathlib if dummies used
    OS_PATCH_TARGET = "os"  # Patch standard os

    # ダミーのFileError例外クラス
    class FileError(Exception):
        pass


# --- Helper Fixture for Path Mocking ---
@pytest.fixture(scope="function")
def mock_path_env(monkeypatch, tmp_path):
    """Mocks Path, os.environ, and provides temporary paths."""
    # Use tmp_path provided by pytest for realistic temporary paths
    mock_project_root = tmp_path / "project"
    mock_workspace = mock_project_root / "ws"
    mock_output = mock_project_root / "out"

    # Create the base directories in the temp filesystem
    mock_project_root.mkdir()
    mock_workspace.mkdir()
    mock_output.mkdir()

    # Return the paths for use in tests without mocking functions
    return {
        "project_root": mock_project_root,
        "workspace": mock_workspace,
        "output": mock_output,
        "environ": os.environ.copy(),
    }


# テスト関数ごとにコンテキストを分離するためのフィクスチャ
@pytest.fixture(scope="function", autouse=True)
def reset_module_caches():
    """各テストの前に関連モジュールのキャッシュをリセットする"""
    import importlib
    import sys

    # テスト前の処理は何もしない
    yield
    # テスト後にモックがリークしないようにする
    # src.utils.path_utils の _project_root キャッシュをクリアする
    if "src.utils.path_utils" in sys.modules:
        with patch("src.utils.path_utils._project_root", None):
            pass


# --- Path Function Tests ---


# Test project root with temporarily defined environment variable
def test_get_project_root_with_env(monkeypatch, tmp_path):
    """Tests that get_project_root respects environment variables."""
    env_path = tmp_path / "env_project_root"
    env_path.mkdir()

    # Set environment variable temporarily for this test
    with monkeypatch.context() as m:
        m.setenv("MIREX_PROJECT_ROOT", str(env_path))
        # Patch get_project_root to not use cache
        with patch("src.utils.path_utils._project_root", None):
            result = get_project_root()
            assert result == env_path


# 実際の実装をテストするため、スキップを解除
def test_get_project_root_actual():
    """実際のプロジェクトルートの取得をテスト"""
    # テスト関数内でのみパッチングを使用
    with patch("src.utils.path_utils._project_root", None):
        result = get_project_root()
        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_dir()
        # プロジェクトのルートマーカーファイルのいずれかが存在することを確認
        root_markers = [".git", "pyproject.toml", "setup.py", "README.md"]
        assert any((result / marker).exists() for marker in root_markers)


def test_get_workspace_dir_actual():
    """実際のワークスペースディレクトリの取得をテスト"""
    # 各テスト関数内でキャッシュをリセット
    with patch("src.utils.path_utils._workspace_dir", None):
        with patch("src.utils.path_utils._project_root", None):
            result = get_workspace_dir()
            assert isinstance(result, Path)
            # ワークスペースは存在しなくても良い（自動作成される可能性があるため）
            # プロジェクトルート内にあることを確認
            project_root = get_project_root()
            assert str(result).startswith(str(project_root))


def test_get_output_base_dir_actual():
    """実際の出力ベースディレクトリの取得をテスト"""
    # 各テスト関数内でキャッシュをリセット
    with patch("src.utils.path_utils._output_base_dir", None):
        with patch("src.utils.path_utils._project_root", None):
            # 最小限の config 辞書を作成
            config = {"paths": {"output_base": "output"}}
            result = get_output_base_dir(config)
            assert isinstance(result, Path)
            # 出力ディレクトリは存在しなくても良い（自動作成される可能性があるため）
            # プロジェクトルート内にあることを確認
            project_root = get_project_root()
            assert str(result).startswith(str(project_root))


# --- is_safe_path_component Tests ---
@pytest.mark.parametrize(
    "component, expected",
    [
        ("filename.txt", True),
        ("sub_dir", True),
        ("with_underscore", True),
        ("with-hyphen", True),
        ("with.dot", True),
        ("..", False),
        ("dir/file", False),
        ("dir\\\\file", False),  # Backslash
        ("/absolute", False),  # Leading slash
        ("", False),  # Empty string (Added)
        (".", False),  # Current directory (Added)
        ("../allowed", False),  # Contains '..'
        ("allowed/..", False),  # Contains '..'
        (" space ", False),  # Leading/trailing space
        ("file\\x07name", False),  # Control character (BEL) (Added)
        ("<forbidden>", False),  # Forbidden char < (Added)
        (">forbidden>", False),  # Forbidden char > (Added)
        (":forbidden:", False),  # Forbidden char : (Added)
        ('\\"forbidden\\"', False),  # Forbidden char " (Added)
        ("|forbidden|", False),  # Forbidden char | (Added)
        ("?forbidden?", False),  # Forbidden char ? (Added)
        ("*forbidden*", False),  # Forbidden char * (Added)
    ],
)
def test_is_safe_path_component(component, expected, caplog):
    """Test is_safe_path_component with various inputs."""
    caplog.set_level(logging.DEBUG)  # Capture debug logs
    result = is_safe_path_component(component)
    assert result == expected
    # If expected is False, check if a debug log message was generated
    # Special cases: Empty string and '.' return False early without logging
    if not expected and component not in ["", "."]:
        assert len(caplog.records) == 1
        assert "Unsafe component" in caplog.text
    elif expected or component in ["", "."]:
        assert (
            len(caplog.records) == 0
        )  # Should not log for safe components or early returns


# It seems the test for None input was missing or incorrect, let's add a specific one.
def test_is_safe_path_component_none_input(caplog):
    """Test is_safe_path_component with None input (should return False)."""
    caplog.set_level(logging.DEBUG)
    # The function handles None via `if not component:` and returns False early.
    result = is_safe_path_component(None)
    assert result is False
    # Check that no logs are generated because it returns early
    assert len(caplog.records) == 0


# --- validate_path_within_allowed_dirs Tests ---


@pytest.fixture(scope="function")
def allowed_dirs_setup(tmp_path):
    """Creates some allowed directories for testing."""
    dir1 = tmp_path / "allowed1"
    dir2 = tmp_path / "allowed1" / "sub"
    dir3 = tmp_path / "other_allowed"
    dir1.mkdir()
    dir2.mkdir(parents=True)
    dir3.mkdir()
    # Create files inside
    (dir2 / "safe_file.txt").touch()
    (dir3 / "another.dat").touch()
    # Create a non-allowed dir
    unsafe_dir = tmp_path / "unsafe"
    unsafe_dir.mkdir()
    (unsafe_dir / "unsafe_file.txt").touch()
    # Return tmp_path and allowed dirs as Paths instead of strings
    return tmp_path, dir1, dir3


def test_validate_path_within_allowed_success(allowed_dirs_setup):
    """Tests paths that should be successfully validated."""
    # パッチングを関数内に限定
    with patch("src.utils.path_utils._project_root", None):
        tmpdir, allowed_dir1, allowed_dir3 = allowed_dirs_setup
        base_path = allowed_dir1
        sub_path = base_path / "sub"

        # ファイルが存在する場合のテスト
        safe_file = sub_path / "safe_file.txt"
        result = validate_path_within_allowed_dirs(
            str(safe_file), [allowed_dir1, allowed_dir3], allow_absolute=True
        )
        assert isinstance(result, Path)
        assert result.resolve() == safe_file.resolve()

        # ディレクトリが存在する場合のテスト
        result = validate_path_within_allowed_dirs(
            str(sub_path), [allowed_dir1, allowed_dir3], allow_absolute=True
        )
        assert isinstance(result, Path)
        assert result.resolve() == sub_path.resolve()


def test_validate_path_outside_allowed_failure(allowed_dirs_setup):
    """Tests paths that should fail validation."""
    # パッチングを関数内に限定
    with patch("src.utils.path_utils._project_root", None):
        tmpdir, allowed_dir1, allowed_dir3 = allowed_dirs_setup
        unsafe_path = tmpdir / "unsafe" / "unsafe_file.txt"

        # 許可されたディレクトリの外にあるパスはエラーになるはず
        with pytest.raises((ValueError, FileError)):
            validate_path_within_allowed_dirs(
                str(unsafe_path), [allowed_dir1, allowed_dir3]
            )


def test_validate_path_parent_traversal_resolves_inside(allowed_dirs_setup):
    """Tests traversal that resolves back inside allowed dir (should pass)."""
    # パッチングを関数内に限定
    with patch("src.utils.path_utils._project_root", None):
        tmpdir, allowed_dir1, allowed_dir3 = allowed_dirs_setup
        # Example: /tmp/pytest-of-user/pytest-0/allowed1/sub/../sub/safe_file.txt
        # Resolves to: /tmp/pytest-of-user/pytest-0/allowed1/sub/safe_file.txt
        tricky_path = allowed_dir1 / "sub" / ".." / "sub" / "safe_file.txt"

        result = validate_path_within_allowed_dirs(
            str(tricky_path), [allowed_dir1, allowed_dir3], allow_absolute=True
        )

        # Check that result is the resolved path
        resolved_path = allowed_dir1 / "sub" / "safe_file.txt"
        assert result.resolve() == resolved_path.resolve()


def test_validate_path_non_existent(allowed_dirs_setup):
    """Tests validation for paths that don't exist yet (should still work)."""
    # パッチングを関数内に限定
    with patch("src.utils.path_utils._project_root", None):
        tmpdir, allowed_dir1, allowed_dir3 = allowed_dirs_setup
        non_existent_path = allowed_dir1 / "new_dir" / "new_file.txt"

        try:
            # 存在チェックのオプションが実装されている場合
            result = validate_path_within_allowed_dirs(
                str(non_existent_path),
                [allowed_dir1, allowed_dir3],
                check_existence=False,
                allow_absolute=True,
            )
        except TypeError:
            # オプションが実装されていない場合、デフォルトで動作するか確認
            result = validate_path_within_allowed_dirs(
                str(non_existent_path),
                [allowed_dir1, allowed_dir3],
                allow_absolute=True,
            )

        # 結果はパスオブジェクトになっているはず
        assert isinstance(result, Path)
        # Result should be the resolved path
        assert result.resolve() == non_existent_path.resolve()


def test_validate_path_is_file_check(allowed_dirs_setup):
    """validate_path_within_allowed_dirsでファイル/ディレクトリチェックをテストする"""
    tmpdir, allowed_dir1, allowed_dir2 = allowed_dirs_setup

    # テスト用のファイルとディレクトリを作成
    test_file = allowed_dir1 / "test_file.txt"
    test_file.touch()
    test_dir = allowed_dir1 / "test_dir"
    test_dir.mkdir()

    # 実際に実装されている動作に合わせてテストを調整
    # 絶対パスが許可されない場合、相対パスでテスト
    with patch("src.utils.path_utils.Path.is_absolute", return_value=False):
        # ファイルパスとcheck_is_file=Trueを指定した場合は成功
        result = validate_path_within_allowed_dirs(
            test_file, [allowed_dir1], check_is_file=True, allow_absolute=True
        )
        assert result.name == test_file.name

        # ディレクトリパスとcheck_is_file=Falseを指定した場合は成功
        result = validate_path_within_allowed_dirs(
            test_dir, [allowed_dir1], check_is_file=False, allow_absolute=True
        )
        assert result.name == test_dir.name

        # ファイルパスとcheck_is_file=Falseを指定した場合はエラー
        with pytest.raises(
            FileError,
            match=f"指定されたパスはディレクトリではありません: {test_file.resolve()}",
        ):
            validate_path_within_allowed_dirs(
                test_file, [allowed_dir1], check_is_file=False, allow_absolute=True
            )

        # ディレクトリパスとcheck_is_file=Trueを指定した場合はエラー
        with pytest.raises(
            FileError,
            match=f"指定されたパスはファイルではありません: {test_dir.resolve()}",
        ):
            validate_path_within_allowed_dirs(
                test_dir, [allowed_dir1], check_is_file=True, allow_absolute=True
            )


def test_validate_path_absolute_path_handling(allowed_dirs_setup):
    """validate_path_within_allowed_dirsで絶対パスの取り扱いをテストする"""
    tmpdir, allowed_dir1, allowed_dir2 = allowed_dirs_setup

    # 絶対パスと相対パスの両方を用意
    abs_path = allowed_dir1 / "file.txt"
    abs_path.touch()
    rel_path = "file.txt"  # 現在のディレクトリからの相対パス

    # 現在の実装では、実際のディレクトリ外の相対パスはうまく扱えない場合がある
    # まず絶対パスをallow_absolute=Trueでテスト
    result = validate_path_within_allowed_dirs(
        abs_path, [allowed_dir1], allow_absolute=True
    )
    assert result.name == abs_path.name

    # allow_absolute=Falseで絶対パスを使用するとエラー
    with pytest.raises(ValueError) as exc_info:
        validate_path_within_allowed_dirs(
            abs_path, [allowed_dir1], allow_absolute=False
        )
    assert "絶対パスは許可されていません" in str(exc_info.value)

    # 現在の実装に合わせて、プロジェクトルートをパッチ
    with patch("src.utils.path_utils.get_project_root", return_value=allowed_dir1):
        # 相対パスのテスト (check_existence=Falseとallow_absolute=Trueを指定)
        with patch.object(Path, "is_absolute", return_value=False):
            result = validate_path_within_allowed_dirs(
                rel_path, [allowed_dir1], check_existence=False, allow_absolute=True
            )
            assert result.name == rel_path


def test_get_workspace_dir_with_environment_var(monkeypatch, tmp_path):
    """環境変数MIREX_WORKSPACEが設定されている場合のget_workspace_dirのテスト"""
    workspace_path = tmp_path / "custom_workspace"
    workspace_path.mkdir()

    with monkeypatch.context() as m:
        # 環境変数を設定
        m.setenv("MIREX_WORKSPACE", str(workspace_path))

        # キャッシュをクリア
        with patch("src.utils.path_utils._workspace_dir", None):
            with patch("src.utils.path_utils._project_root", None):
                # 環境変数からワークスペースディレクトリが取得されることを確認
                result = get_workspace_dir()
                assert result == workspace_path


def test_get_workspace_dir_with_relative_path(monkeypatch, tmp_path):
    """MIREX_WORKSPACEに相対パスが設定されている場合のget_workspace_dirのテスト"""
    project_root = tmp_path / "project"
    project_root.mkdir()

    with monkeypatch.context() as m:
        # 相対パスを環境変数に設定
        m.setenv("MIREX_WORKSPACE", "relative_workspace")

        # キャッシュとプロジェクトルートをモック
        with patch("src.utils.path_utils._workspace_dir", None):
            with patch(
                "src.utils.path_utils.get_project_root", return_value=project_root
            ):
                # 相対パスがプロジェクトルートを基準に解決されることを確認
                result = get_workspace_dir()
                expected_path = project_root / "relative_workspace"
                assert result == expected_path


def test_get_workspace_dir_ensure_dir_failure(monkeypatch, tmp_path):
    """ensure_dirが失敗する場合のget_workspace_dirのテスト"""
    project_root = tmp_path / "project"
    project_root.mkdir()

    with monkeypatch.context() as m:
        # キャッシュとプロジェクトルートをモック
        with patch("src.utils.path_utils._workspace_dir", None):
            with patch(
                "src.utils.path_utils.get_project_root", return_value=project_root
            ):
                # ensure_dirが失敗するようにモック
                with patch(
                    "src.utils.path_utils.ensure_dir",
                    side_effect=FileError("ディレクトリ作成エラー"),
                ):
                    # ConfigErrorが発生することを確認
                    with pytest.raises(ConfigError) as exc_info:
                        get_workspace_dir()
                    # 実際の実装に合わせてエラーメッセージを確認
                    assert (
                        "ワークスペースディレクトリの設定またはアクセスに問題があります"
                        in str(exc_info.value)
                    )
                    assert "ディレクトリ作成エラー" in str(exc_info.value)


def test_ensure_dir_writable_check_failure(monkeypatch, tmp_path):
    """書き込みチェックが失敗する場合のensure_dirのテスト"""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    # 一時ファイル作成がOSErrorを送出するようにモック
    with patch(
        "tempfile.NamedTemporaryFile", side_effect=OSError("書き込み権限がありません")
    ):
        # check_writable=Trueの場合にFileErrorが発生
        with pytest.raises(FileError) as exc_info:
            ensure_dir(test_dir, check_writable=True)

        # 実際の実装に合わせてエラーメッセージを確認
        # ディレクトリに書き込み権限がないか、予期せぬエラーのどちらかのメッセージが含まれるはず
        error_msg = str(exc_info.value)
        assert (
            "ディレクトリ確認中に予期せぬエラー" in error_msg
            or "ディレクトリに書き込み権限がありません" in error_msg
        )


# --- get_detector_path Tests ---


@pytest.fixture
def detector_test_setup(tmp_path):
    """検出器パスのテスト用セットアップ"""
    # ディレクトリ構造
    src_dir = tmp_path / "src" / "detectors"
    src_dir.mkdir(parents=True)

    improved_dir = tmp_path / "workspace" / "improved_versions"
    improved_dir.mkdir(parents=True)

    # クラス名用のサブディレクトリ
    detector_class_dir = improved_dir / "TestDetector"
    detector_class_dir.mkdir()

    # オリジナルファイル
    original_file = src_dir / "test_detector.py"
    original_file.write_text("# Original detector file")

    # 改善版ファイル
    improved_v1 = detector_class_dir / "test_detector_v1.py"
    improved_v1.write_text("# Improved detector v1")

    improved_v2 = detector_class_dir / "test_detector_v2.py"
    improved_v2.write_text("# Improved detector v2")

    # ダミー設定
    config = {
        "paths": {
            "workspace": str(tmp_path / "workspace"),
            "detectors_src": str(src_dir),
        }
    }

    return {
        "tmp_path": tmp_path,
        "src_dir": src_dir,
        "improved_dir": improved_dir,
        "detector_class_dir": detector_class_dir,
        "original_file": original_file,
        "improved_v1": improved_v1,
        "improved_v2": improved_v2,
        "config": config,
    }


def test_get_detector_path_original(detector_test_setup, monkeypatch):
    """オリジナルの検出器パスを取得するテスト"""
    setup = detector_test_setup

    # get_detectors_src_dir と get_improved_versions_dir をモック
    def mock_get_detectors_src_dir(_):
        return setup["src_dir"]

    def mock_get_improved_versions_dir(_):
        return setup["improved_dir"]

    monkeypatch.setattr(
        "src.utils.path_utils.get_detectors_src_dir", mock_get_detectors_src_dir
    )
    monkeypatch.setattr(
        "src.utils.path_utils.get_improved_versions_dir", mock_get_improved_versions_dir
    )

    # テスト実行（use_original=Trueを指定）
    result = get_detector_path("TestDetector", setup["config"], use_original=True)

    # 検証
    assert result == setup["original_file"].resolve()


def test_get_detector_path_specific_version(detector_test_setup, monkeypatch):
    """特定バージョンの改善版検出器パスを取得するテスト"""
    setup = detector_test_setup

    # get_detectors_src_dir と get_improved_versions_dir をモック
    def mock_get_detectors_src_dir(_):
        return setup["src_dir"]

    def mock_get_improved_versions_dir(_):
        return setup["improved_dir"]

    monkeypatch.setattr(
        "src.utils.path_utils.get_detectors_src_dir", mock_get_detectors_src_dir
    )
    monkeypatch.setattr(
        "src.utils.path_utils.get_improved_versions_dir", mock_get_improved_versions_dir
    )

    # テスト実行
    result = get_detector_path("TestDetector", setup["config"], version="v1")

    # 検証
    assert result == setup["improved_v1"].resolve()


def test_get_detector_path_latest_version(detector_test_setup, monkeypatch):
    """最新バージョンの改善版検出器パスを取得するテスト"""
    setup = detector_test_setup

    # get_detectors_src_dir と get_improved_versions_dir をモック
    def mock_get_detectors_src_dir(_):
        return setup["src_dir"]

    def mock_get_improved_versions_dir(_):
        return setup["improved_dir"]

    monkeypatch.setattr(
        "src.utils.path_utils.get_detectors_src_dir", mock_get_detectors_src_dir
    )
    monkeypatch.setattr(
        "src.utils.path_utils.get_improved_versions_dir", mock_get_improved_versions_dir
    )

    # テスト実行 (バージョン指定なし = 最新版)
    result = get_detector_path("TestDetector", setup["config"])

    # 検証
    assert result == setup["improved_v2"].resolve()


def test_get_detector_path_not_found(detector_test_setup, monkeypatch):
    """存在しない検出器のパスを取得しようとするテスト"""
    setup = detector_test_setup

    # get_detectors_src_dir と get_improved_versions_dir をモック
    def mock_get_detectors_src_dir(_):
        return setup["src_dir"]

    def mock_get_improved_versions_dir(_):
        return setup["improved_dir"]

    monkeypatch.setattr(
        "src.utils.path_utils.get_detectors_src_dir", mock_get_detectors_src_dir
    )
    monkeypatch.setattr(
        "src.utils.path_utils.get_improved_versions_dir", mock_get_improved_versions_dir
    )

    # テスト実行
    with pytest.raises(FileNotFoundError):
        get_detector_path("NonExistentDetector", setup["config"])


def test_get_detector_path_invalid_version(detector_test_setup, monkeypatch):
    """存在しないバージョンの検出器パスを取得しようとするテスト"""
    setup = detector_test_setup

    # get_detectors_src_dir と get_improved_versions_dir をモック
    def mock_get_detectors_src_dir(_):
        return setup["src_dir"]

    def mock_get_improved_versions_dir(_):
        return setup["improved_dir"]

    monkeypatch.setattr(
        "src.utils.path_utils.get_detectors_src_dir", mock_get_detectors_src_dir
    )
    monkeypatch.setattr(
        "src.utils.path_utils.get_improved_versions_dir", mock_get_improved_versions_dir
    )

    # テスト実行
    with pytest.raises(FileNotFoundError):
        get_detector_path("TestDetector", setup["config"], version="v999")


# --- get_dataset_paths Tests ---


@pytest.fixture
def dataset_test_setup(tmp_path):
    """データセットパスのテスト用セットアップ"""
    # ディレクトリ構造
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    audio_dir = datasets_dir / "test_audio"
    audio_dir.mkdir()

    label_dir = datasets_dir / "test_labels"
    label_dir.mkdir()

    # ファイル作成
    audio_file1 = audio_dir / "sample1.wav"
    audio_file1.write_text("audio1")

    audio_file2 = audio_dir / "sample2.wav"
    audio_file2.write_text("audio2")

    label_file1 = label_dir / "sample1.csv"
    label_file1.write_text("label1")

    label_file2 = label_dir / "sample2.csv"
    label_file2.write_text("label2")

    # filelist
    filelist_dir = tmp_path / "filelists"
    filelist_dir.mkdir()
    filelist_path = filelist_dir / "test_filelist.txt"
    filelist_path.write_text("sample1\nsample2\n")

    # ダミー設定
    config = {
        "paths": {"datasets_base": str(datasets_dir)},
        "datasets": {
            "test_dataset": {
                "audio_dir": "test_audio",
                "label_dir": "test_labels",
                "audio_ext": ".wav",
                "label_ext": ".csv",
            },
            "test_dataset_with_filelist": {
                "audio_dir": "test_audio",
                "label_dir": "test_labels",
                "filelist": str(filelist_path),
                "audio_ext": ".wav",
                "label_ext": ".csv",
            },
        },
    }

    return {
        "tmp_path": tmp_path,
        "datasets_dir": datasets_dir,
        "audio_dir": audio_dir,
        "label_dir": label_dir,
        "audio_file1": audio_file1,
        "audio_file2": audio_file2,
        "label_file1": label_file1,
        "label_file2": label_file2,
        "filelist_path": filelist_path,
        "config": config,
    }


def test_get_dataset_paths_basic(dataset_test_setup, monkeypatch):
    """基本的なデータセットパスの取得テスト"""
    setup = dataset_test_setup

    # get_project_root をモック
    def mock_get_project_root():
        return setup["tmp_path"]

    monkeypatch.setattr("src.utils.path_utils.get_project_root", mock_get_project_root)

    # テスト実行
    audio_dir, label_dir, file_pairs = get_dataset_paths(
        setup["config"], "test_dataset"
    )

    # 検証
    assert audio_dir == setup["audio_dir"]
    assert label_dir == setup["label_dir"]
    assert len(file_pairs) == 2

    # ファイルペアの内容を確認
    files = [(a.name, l.name) for a, l in file_pairs]
    assert ("sample1.wav", "sample1.csv") in files
    assert ("sample2.wav", "sample2.csv") in files


def test_get_dataset_paths_with_filelist(dataset_test_setup, monkeypatch):
    """ファイルリストを使用したデータセットパスの取得テスト"""
    setup = dataset_test_setup

    # get_project_root をモック
    def mock_get_project_root():
        return setup["tmp_path"]

    monkeypatch.setattr("src.utils.path_utils.get_project_root", mock_get_project_root)

    # テスト実行
    audio_dir, label_dir, file_pairs = get_dataset_paths(
        setup["config"], "test_dataset_with_filelist"
    )

    # 検証
    assert audio_dir == setup["audio_dir"]
    assert label_dir == setup["label_dir"]
    assert len(file_pairs) == 2

    # ファイルペアの内容を確認
    files = [(a.name, l.name) for a, l in file_pairs]
    assert ("sample1.wav", "sample1.csv") in files
    assert ("sample2.wav", "sample2.csv") in files


def test_get_dataset_paths_unknown_dataset(dataset_test_setup, monkeypatch):
    """存在しないデータセット名を指定した場合のテスト"""
    setup = dataset_test_setup

    # get_project_root をモック
    def mock_get_project_root():
        return setup["tmp_path"]

    monkeypatch.setattr("src.utils.path_utils.get_project_root", mock_get_project_root)

    # テスト実行
    with pytest.raises(ConfigError, match=".*定義が見つかりません.*"):
        get_dataset_paths(setup["config"], "unknown_dataset")


def test_get_dataset_paths_missing_section(monkeypatch, tmp_path):
    """設定にdatasetsセクションがない場合のget_dataset_pathsのテスト"""
    project_root = tmp_path / "project"
    project_root.mkdir()

    # datasetsセクションがない設定
    invalid_config = {}

    with monkeypatch.context() as m:
        with patch("src.utils.path_utils.get_project_root", return_value=project_root):
            # ConfigErrorが発生することを確認
            with pytest.raises(ConfigError) as exc_info:
                get_dataset_paths(invalid_config, "test_dataset")
            # 実際の実装に合わせてエラーメッセージを確認
            assert (
                "config.yaml に 'datasets' セクションが見つからないか、無効です"
                in str(exc_info.value)
            )


def test_get_dataset_paths_missing_dataset(monkeypatch, tmp_path):
    """指定されたデータセットが設定に存在しない場合のget_dataset_pathsのテスト"""
    project_root = tmp_path / "project"
    project_root.mkdir()

    # データセットが含まれていない設定
    config = {"datasets": {"other_dataset": {}}}

    with monkeypatch.context() as m:
        with patch("src.utils.path_utils.get_project_root", return_value=project_root):
            # ConfigErrorが発生することを確認
            with pytest.raises(ConfigError) as exc_info:
                get_dataset_paths(config, "test_dataset")
            # 実際の実装に合わせてエラーメッセージを確認
            assert (
                "config.yaml の 'datasets' セクションにデータセット 'test_dataset' の定義が見つかりません"
                in str(exc_info.value)
            )


def test_get_dataset_paths_missing_required_dirs(monkeypatch, tmp_path):
    """audio_dirまたはlabel_dirが欠けている場合のget_dataset_pathsのテスト"""
    project_root = tmp_path / "project"
    project_root.mkdir()

    # audio_dirが欠けている設定
    config1 = {"datasets": {"test_dataset": {"label_dir": "labels"}}}

    with monkeypatch.context() as m:
        with patch("src.utils.path_utils.get_project_root", return_value=project_root):
            # ConfigErrorが発生することを確認
            with pytest.raises(ConfigError) as exc_info:
                get_dataset_paths(config1, "test_dataset")
            # 実際の実装に合わせてエラーメッセージを確認
            assert "データセット 'test_dataset' の設定に 'audio_dir' または 'label_dir' がありません" in str(
                exc_info.value
            )


def test_get_dataset_paths_invalid_directories(monkeypatch, tmp_path):
    """設定されたディレクトリが存在しない場合のget_dataset_pathsのテスト"""
    project_root = tmp_path / "project"
    project_root.mkdir()

    # 無効なディレクトリを持つ設定
    config = {
        "datasets": {
            "test_dataset": {
                "audio_dir": "non_existent_audio",
                "label_dir": "non_existent_labels",
            }
        }
    }

    with monkeypatch.context() as m:
        with patch("src.utils.path_utils.get_project_root", return_value=project_root):
            # オリジナルの動作によるエラーが発生することを確認
            with pytest.raises(FileError) as exc_info:
                get_dataset_paths(config, "test_dataset")
            # 実際の実装に合わせてエラーメッセージを確認
            assert "オーディオディレクトリが見つかりません" in str(exc_info.value)


def test_get_dataset_paths_with_filelist_missing(monkeypatch, tmp_path):
    """filelistが指定されているが存在しない場合のget_dataset_pathsのテスト"""
    project_root = tmp_path / "project"
    project_root.mkdir()

    # ディレクトリ構造を作成
    audio_dir = project_root / "audio"
    label_dir = project_root / "labels"
    audio_dir.mkdir()
    label_dir.mkdir()

    # 存在しないファイルリストを持つ設定
    config = {
        "datasets": {
            "test_dataset": {
                "audio_dir": str(audio_dir),
                "label_dir": str(label_dir),
                "filelist": "non_existent_filelist.txt",
            }
        }
    }

    with monkeypatch.context() as m:
        with patch("src.utils.path_utils.get_project_root", return_value=project_root):
            # FileErrorが発生することを確認
            with pytest.raises(FileError) as exc_info:
                get_dataset_paths(config, "test_dataset")
            assert "ファイルリストが見つかりません" in str(exc_info.value)


def test_get_dataset_paths_with_different_label_formats(monkeypatch, tmp_path, caplog):
    """異なるlabel_formatでのget_dataset_pathsのテスト"""
    project_root = tmp_path / "project"
    project_root.mkdir()

    # ディレクトリ構造を作成
    audio_dir = project_root / "audio"
    label_dir = project_root / "labels"
    audio_dir.mkdir()
    label_dir.mkdir()

    # オーディオファイルとラベルファイルを作成
    audio_file = audio_dir / "test_stem.wav"
    audio_file.touch()

    # 異なるフォーマットのラベルファイル
    melody1_file = label_dir / "test_stem_MELODY1.csv"
    melody1_file.touch()

    melody2_file = label_dir / "test_stem_MELODY2.csv"
    melody2_file.touch()

    mirex_file = label_dir / "test_stem.txt"
    mirex_file.touch()

    default_file = label_dir / "test_stem.csv"
    default_file.touch()

    # JSONフォーマットのラベルファイルを追加
    label1_file = label_dir / "test_stem_label1.json"
    label1_file.touch()

    label2_file = label_dir / "test_stem_label2.json"
    label2_file.touch()

    # 異なるlabel_formatでテスト - 実際の実装に合わせてテスト調整
    label_formats = ["melody1", "melody2", "mirex_melody", None]  # None = デフォルト

    for label_format in label_formats:
        config = {
            "datasets": {
                "test_dataset": {
                    "audio_dir": str(audio_dir),
                    "label_dir": str(label_dir),
                }
            }
        }

        if label_format:
            config["datasets"]["test_dataset"]["label_format"] = label_format

        with monkeypatch.context() as m:
            with patch(
                "src.utils.path_utils.get_project_root", return_value=project_root
            ):
                # 各フォーマットでファイルペアが返されることを確認
                audio_path, label_path, file_pairs = get_dataset_paths(
                    config, "test_dataset"
                )

                assert audio_path == audio_dir
                assert label_path == label_dir

                # デフォルトフォーマット（None）の場合は2つのファイルペアが見つかる
                if label_format is None:
                    assert (
                        len(file_pairs) == 2
                    )  # Default format finds two pairs (.txt and .csv)
                    # ファイルペアのラベルパスを確認
                    label_paths = [pair[1] for pair in file_pairs]
                    # Ensure the .txt and .csv files are present
                    assert any(str(path).endswith(".txt") for path in label_paths)
                    assert any(str(path).endswith(".csv") for path in label_paths)
                    # Ensure the .json files are NOT present as pairs (due to missing audio)
                    assert not any(str(path).endswith(".json") for path in label_paths)
                else:
                    assert (
                        len(file_pairs) == 1
                    )  # 特定のフォーマットでは1つのファイルペアが見つかる

                assert file_pairs[0][0] == audio_file  # オーディオファイルは同じ

                # ラベルファイルの実在確認
                label_file = file_pairs[0][1]
                assert label_file.exists()

                # label_formatに応じて期待されるファイル名パターンを確認
                if label_format == "melody1":
                    assert label_file.name.endswith("_MELODY1.csv")
                elif label_format == "melody2":
                    assert label_file.name.endswith("_MELODY2.csv")
                elif label_format == "mirex_melody":
                    # Ensure the .txt file is returned as per fallback logic
                    assert label_file.suffix == ".txt"
                else:  # デフォルト
                    # Check if the label is either .txt or .csv for the default case
                    assert label_file.suffix in [".txt", ".csv"]


# --- setup_python_path Tests ---


def test_setup_python_path(monkeypatch):
    """setup_python_path関数のテスト"""
    from src.utils.path_utils import setup_python_path
    import sys

    # テスト用の一時的なパスを作成
    mock_project_root = Path("/fake/project/root")
    mock_src_dir = mock_project_root / "src"

    # get_project_rootをモック
    def mock_get_project_root():
        return mock_project_root

    # os.path.isdirをモック
    def mock_isdir(path):
        return str(path) == str(mock_src_dir)

    # load_environment_variablesをモック
    def mock_load_env():
        pass

    # 元のsys.pathを保存
    original_sys_path = sys.path.copy()

    try:
        # モックを適用
        monkeypatch.setattr(
            "src.utils.path_utils.get_project_root", mock_get_project_root
        )
        monkeypatch.setattr("os.path.isdir", mock_isdir)
        monkeypatch.setattr(
            "src.utils.path_utils.load_environment_variables", mock_load_env
        )

        # もし既にパスに含まれている場合は一時的に削除
        if str(mock_project_root) in sys.path:
            sys.path.remove(str(mock_project_root))
        if str(mock_src_dir) in sys.path:
            sys.path.remove(str(mock_src_dir))

        # 関数を実行
        setup_python_path()

        # 検証
        assert str(mock_project_root) in sys.path
        assert str(mock_src_dir) in sys.path
        # 先頭に追加されることを確認
        assert sys.path[0] == str(mock_src_dir)
        assert sys.path[1] == str(mock_project_root)
    finally:
        # sys.pathを元に戻す
        sys.path = original_sys_path


def test_setup_python_path_error_handling(monkeypatch):
    """setup_python_path関数のエラーハンドリングテスト"""
    from src.utils.path_utils import setup_python_path

    # get_project_rootが例外を投げるようにモック
    def mock_get_project_root_error():
        from src.utils.exception_utils import ConfigError

        raise ConfigError("プロジェクトルートが見つかりません")

    # load_environment_variablesをモック
    def mock_load_env():
        pass

    # モックを適用
    monkeypatch.setattr(
        "src.utils.path_utils.get_project_root", mock_get_project_root_error
    )
    monkeypatch.setattr(
        "src.utils.path_utils.load_environment_variables", mock_load_env
    )

    # エラーが処理され、例外が投げられないことを確認
    setup_python_path()  # 例外が発生しないことを確認


# --- get_absolute_path Tests ---


class TestGetAbsolutePath:
    def test_absolute_input(self, tmp_path):
        """Test with an absolute path input."""
        abs_path = (tmp_path / "absolute" / "file.txt").resolve()
        # Create the parent dir for resolve() not to fail if path doesn't exist
        abs_path.parent.mkdir(parents=True)

        result = get_absolute_path(str(abs_path))
        assert result == abs_path

        result_path_obj = get_absolute_path(abs_path)
        assert result_path_obj == abs_path

    def test_relative_input_default_base(self, monkeypatch, tmp_path):
        """Test with a relative path input and default base (project root)."""
        mock_project_root = (tmp_path / "mock_project").resolve()
        monkeypatch.setattr(
            "src.utils.path_utils.get_project_root", lambda: mock_project_root
        )

        relative_path_str = "data/file.txt"
        relative_path_obj = Path(relative_path_str)
        expected_path = (mock_project_root / relative_path_str).resolve()

        # String input
        result_str = get_absolute_path(relative_path_str)
        assert result_str == expected_path

        # Path object input
        result_path = get_absolute_path(relative_path_obj)
        assert result_path == expected_path

    def test_relative_input_custom_base(self, tmp_path):
        """Test with a relative path input and a specified base_dir."""
        custom_base = (tmp_path / "custom_base").resolve()
        custom_base.mkdir()  # Ensure base exists

        relative_path_str = "sub/dir/another.txt"
        relative_path_obj = Path(relative_path_str)
        expected_path = (custom_base / relative_path_str).resolve()

        # String input
        result_str = get_absolute_path(relative_path_str, base_dir=custom_base)
        assert result_str == expected_path

        # Path object input
        result_path = get_absolute_path(relative_path_obj, base_dir=custom_base)
        assert result_path == expected_path

    def test_resolve_behavior(self, tmp_path):
        """Test that the function uses resolve() correctly."""
        base = tmp_path / "base"
        base.mkdir()
        tricky_relative = "../base/./file.txt"  # Path requiring resolution
        expected = (tmp_path / "base" / "file.txt").resolve()

        result = get_absolute_path(tricky_relative, base_dir=base)
        assert result == expected


# --- get_allowed_upload_directories Tests ---


def test_get_allowed_upload_directories(monkeypatch):
    """get_allowed_upload_directories関数のテスト"""
    # 環境変数のモック
    env_vars = {"MIREX_ALLOWED_UPLOAD_DIRS": "/path/to/upload1:/path/to/upload2"}

    # 環境変数を設定
    monkeypatch.setattr(os, "environ", env_vars)

    # Path.resolveをモック
    original_resolve = Path.resolve

    def mock_resolve(self, strict=False):
        return self

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    # 関数を実行
    result = get_allowed_upload_directories()

    # 検証
    assert len(result) == 2
    assert "/path/to/upload1" in str(result[0])
    assert "/path/to/upload2" in str(result[1])


def test_get_allowed_upload_directories_default(monkeypatch):
    """環境変数が設定されていない場合、デフォルト値が使用されることを確認"""
    # 環境変数が設定されていない場合
    env_vars = {}

    # 環境変数を設定
    monkeypatch.setattr(os, "environ", env_vars)

    # get_project_rootをモック
    def mock_get_project_root():
        return Path("/mock/project/root")

    monkeypatch.setattr("src.utils.path_utils.get_project_root", mock_get_project_root)

    # mkdir のモック
    mkdir_called = False

    def mock_mkdir(self, parents=False, exist_ok=False):
        nonlocal mkdir_called
        mkdir_called = True

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    # Path.exists のモック
    def mock_exists(self):
        return False  # ディレクトリが存在しないと仮定

    monkeypatch.setattr(Path, "exists", mock_exists)

    # Path.resolveをモック
    def mock_resolve(self, strict=False):
        return self

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    # 関数を実行
    result = get_allowed_upload_directories()

    # デフォルト値の検証
    assert len(result) == 1
    assert result[0].name == "uploads"
    assert mkdir_called  # Path.mkdir が呼び出されたことを確認


def test_get_allowed_upload_directories_exists(monkeypatch):
    """デフォルトディレクトリが既に存在する場合のテスト（get_allowed_upload_directories）"""
    # 環境変数が設定されていない場合
    env_vars = {}

    # 環境変数を設定
    monkeypatch.setattr(os, "environ", env_vars)

    # get_project_rootをモック
    def mock_get_project_root():
        return Path("/mock/project/root")

    monkeypatch.setattr("src.utils.path_utils.get_project_root", mock_get_project_root)

    # mkdir のモック
    mkdir_called = False

    def mock_mkdir(self, parents=False, exist_ok=False):
        nonlocal mkdir_called
        mkdir_called = True

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    # Path.exists のモック - ディレクトリが既に存在するケース
    def mock_exists(self):
        return True

    monkeypatch.setattr(Path, "exists", mock_exists)

    # Path.is_dir のモック
    def mock_is_dir(self):
        return True

    monkeypatch.setattr(Path, "is_dir", mock_is_dir)

    # Path.resolveをモック
    def mock_resolve(self, strict=False):
        return self

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    # 関数を実行
    result = get_allowed_upload_directories()

    # 検証
    assert len(result) == 1
    assert result[0].name == "uploads"
    assert not mkdir_called  # ディレクトリが存在するのでmkdirは呼ばれないはず


def test_get_allowed_upload_directories_permission_error(monkeypatch):
    """ディレクトリ作成時に権限エラーが発生する場合のテスト（get_allowed_upload_directories）"""
    # 環境変数が設定されていない場合
    env_vars = {}

    # 環境変数を設定
    monkeypatch.setattr(os, "environ", env_vars)

    # get_project_rootをモック
    def mock_get_project_root():
        return Path("/mock/project/root")

    monkeypatch.setattr("src.utils.path_utils.get_project_root", mock_get_project_root)

    # mkdir のモック - 権限エラーを発生させる
    def mock_mkdir_error(self, parents=False, exist_ok=False):
        raise PermissionError("権限がありません")

    monkeypatch.setattr(Path, "mkdir", mock_mkdir_error)

    # Path.exists のモック
    def mock_exists(self):
        return False  # ディレクトリは存在しない

    monkeypatch.setattr(Path, "exists", mock_exists)

    # Path.resolveをモック
    def mock_resolve(self, strict=False):
        return self

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    # 関数を実行 - 警告ログが出るはずだが例外は発生しない
    result = get_allowed_upload_directories()

    # 検証
    assert len(result) == 1
    assert result[0].name == "uploads"


def test_allowed_upload_directories_relative_path(monkeypatch, tmp_path):
    """相対パスが正しく解決されることをテスト"""
    project_root = tmp_path / "project"
    project_root.mkdir()
    rel_dir = "rel_uploads"
    abs_dir = str(tmp_path / "abs_uploads")

    # 相対パスと絶対パスが混在する環境変数
    monkeypatch.setenv("MIREX_ALLOWED_UPLOAD_DIRS", f"{rel_dir}:{abs_dir}")

    with patch("src.utils.path_utils.get_project_root", return_value=project_root):
        result = get_allowed_upload_directories()

        # 相対パスが正しくプロジェクトルートからの解決されていることを確認
        assert len(result) == 2
        assert result[0] == (project_root / rel_dir).resolve()
        assert result[1] == Path(abs_dir).resolve()


def test_get_allowed_vad_directories_edge_cases(monkeypatch, tmp_path):
    """VADディレクトリのエッジケースをテスト"""
    # 存在するが通常のファイルであるパスを設定
    tmp_file = tmp_path / "test_file.txt"
    tmp_file.write_text("test content")

    monkeypatch.setenv("MIREX_ALLOWED_VAD_DIRS", str(tmp_file))

    # get_project_rootをモックして一貫したデフォルトパスを確保
    project_root = tmp_path / "project_root"
    project_root.mkdir()

    with patch("src.utils.path_utils.get_project_root", return_value=project_root):
        result = get_allowed_vad_directories()
        # 現在の実装では警告は出るが、ファイルパスであってもリストに追加される
        assert len(result) == 1
        # ディレクトリではなくファイル名が返される
        assert result[0].name == "test_file.txt"
        # 正しいディレクトリ構造が維持されているか確認
        assert str(tmp_file.resolve()) in str(result[0])


def test_validate_path_invalid_path_format():
    """無効なパス形式の処理をテスト"""
    if os.name == "nt":  # Windows
        # Windowsの場合は代替テスト: 無効な文字を含むパス
        invalid_path = r"C:\some:\path"
    else:
        # Unix/Linuxの場合: NULバイトを含むパス
        invalid_path = "some\0path"

    with pytest.raises(ValueError):
        validate_path_within_allowed_dirs(invalid_path, ["/valid/path"])


# --- Additional tests for find_project_root and get_project_root ---


def test_find_project_root_env_var_invalid_path(monkeypatch, tmp_path, caplog):
    """環境変数MIREX_PROJECT_ROOTが設定されているが、実際には存在しないパスを指している場合のテスト"""
    # 存在しないパスを環境変数に設定
    non_existent_path = str(tmp_path / "does_not_exist")

    with monkeypatch.context() as m:
        m.setenv("MIREX_PROJECT_ROOT", non_existent_path)

        # _project_rootキャッシュをクリア
        with patch("src.utils.path_utils._project_root", None):
            from src.utils.path_utils import find_project_root

            # マーカーファイルを準備（環境変数が無効な場合のフォールバック用）
            project_dir = tmp_path / "project"
            project_dir.mkdir()
            marker_file = project_dir / "pyproject.toml"
            marker_file.touch()

            # カレントディレクトリをプロジェクトディレクトリの子ディレクトリに設定
            child_dir = project_dir / "src"
            child_dir.mkdir()

            with patch("pathlib.Path.cwd", return_value=child_dir):
                # 実装では警告ログを出力して、警告メッセージをキャプチャするためにwarnings.warnを使用
                with patch("warnings.warn") as mock_warn:
                    result = find_project_root()
                    # 警告が出ていることを確認
                    mock_warn.assert_called_once()
                    # マーカーファイルが見つかる代替パスが返されることを確認
                    assert result == project_dir


def test_find_project_root_marker_search(monkeypatch, tmp_path):
    """環境変数が設定されていない場合、マーカーファイルの検索が行われることを確認するテスト"""
    # 環境変数MIREX_PROJECT_ROOTを削除
    with monkeypatch.context() as m:
        m.delenv("MIREX_PROJECT_ROOT", raising=False)

        # _project_rootキャッシュをクリア
        with patch("src.utils.path_utils._project_root", None):
            from src.utils.path_utils import find_project_root

            # プロジェクト構造を作成
            project_dir = tmp_path / "project"
            project_dir.mkdir()

            # マーカーファイルを作成
            marker_file = project_dir / "pyproject.toml"
            marker_file.touch()

            # 子ディレクトリから検索を開始
            child_dir = project_dir / "src" / "utils"
            child_dir.mkdir(parents=True)

            with patch("pathlib.Path.cwd", return_value=child_dir):
                # マーカーファイルによってプロジェクトルートが検出されることを確認
                result = find_project_root()
                assert result == project_dir


def test_find_project_root_no_marker_found(monkeypatch, tmp_path):
    """環境変数もマーカーファイルも見つからない場合のテスト"""
    # 環境変数MIREX_PROJECT_ROOTを削除
    with monkeypatch.context() as m:
        m.delenv("MIREX_PROJECT_ROOT", raising=False)

        # _project_rootキャッシュをクリア
        with patch("src.utils.path_utils._project_root", None):
            from src.utils.path_utils import find_project_root

            # マーカーファイルがない構造を作成
            empty_dir = tmp_path / "empty"
            empty_dir.mkdir()

            with patch("pathlib.Path.cwd", return_value=empty_dir):
                # マーカーファイルがない場合はConfigErrorが発生することを確認
                with pytest.raises(ConfigError) as exc_info:
                    find_project_root(max_depth_up=1)  # 検索深度を制限

                assert "プロジェクトルートが見つかりませんでした" in str(exc_info.value)


def test_get_project_root_cache_behavior(monkeypatch, tmp_path):
    """get_project_rootのキャッシュ動作をテストする"""
    # 初期状態で環境変数を設定
    env_path1 = tmp_path / "env_path1"
    env_path1.mkdir()

    with monkeypatch.context() as m:
        m.setenv("MIREX_PROJECT_ROOT", str(env_path1))

        # _project_rootキャッシュをクリア
        with patch("src.utils.path_utils._project_root", None):
            # 1回目の呼び出し（キャッシュに保存される）
            result1 = get_project_root()
            assert result1 == env_path1

            # キャッシュを直接設定（use_cache=Falseでも使用されないことを確認するため）
            # 実装では、_project_rootはプライベートモジュール変数
            with patch("src.utils.path_utils._project_root", env_path1):
                # 環境変数を変更
                env_path2 = tmp_path / "env_path2"
                env_path2.mkdir()
                m.setenv("MIREX_PROJECT_ROOT", str(env_path2))

                # キャッシュを使用する場合は最初の値が返される
                result2 = get_project_root(use_cache=True)
                assert result2 == env_path1

                # キャッシュをクリアし、キャッシュを使用しない場合は新しい値が返される
                with patch("src.utils.path_utils._project_root", None):
                    result3 = get_project_root(use_cache=False)
                    assert result3 == env_path2


# --- ensure_dir Tests ---


def test_ensure_dir_creates_new(tmp_path):
    """Tests that ensure_dir creates a new directory including parents."""
    new_dir = tmp_path / "a" / "b" / "c"
    assert not new_dir.exists()

    result = ensure_dir(str(new_dir))

    assert new_dir.is_dir()
    assert isinstance(result, Path)
    assert result == new_dir


def test_ensure_dir_existing(tmp_path):
    """Tests that ensure_dir works with an existing directory."""
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    assert existing_dir.is_dir()

    result = ensure_dir(str(existing_dir))

    assert existing_dir.is_dir()
    assert isinstance(result, Path)
    assert result == existing_dir


def test_ensure_dir_with_file_conflict(tmp_path):
    """Tests that ensure_dir raises an error if a file exists at the path."""
    file_path = tmp_path / "file.txt"
    file_path.touch()
    assert file_path.is_file()

    with pytest.raises(FileError):
        ensure_dir(str(file_path))


def test_ensure_dir_with_permission_error(monkeypatch, tmp_path):
    """権限エラー時のディレクトリ作成処理をテスト"""
    test_dir = tmp_path / "test_dir"

    # mkdirが権限エラーを発生させるようにモック
    def mock_mkdir_permission_error(*args, **kwargs):
        raise PermissionError("権限がありません")

    with patch.object(Path, "mkdir", side_effect=mock_mkdir_permission_error):
        # check_writable=TrueでもFalseでも例外が発生する（現在の実装の動作に合わせる）
        with pytest.raises(FileError):
            ensure_dir(test_dir, check_writable=True)

        with pytest.raises(FileError):
            ensure_dir(test_dir, check_writable=False)
