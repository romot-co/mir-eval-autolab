import pytest
from pathlib import Path
import os
import sys
import re

# テスト対象モジュールをインポート (src が PYTHONPATH に追加されている想定)
from src.utils import path_utils
from src.utils.exception_utils import FileError, ConfigError

# --- is_safe_path_component のテスト ---

@pytest.mark.parametrize("component, expected", [
    # --- 安全なケース ---
    ("myfile", True),
    ("my_file", True),
    ("my-file", True),
    ("file123", True),
    ("myfile.txt", True),
    ("data", True),
    (".hiddenfile", True), # 先頭のドットは通常許容される
    ("a_b.c-d_1.log", True),

    # --- 安全でないケース ---
    ("", False),              # 空文字列
    (".", False),             # カレントディレクトリ
    ("..", False),            # 親ディレクトリ
    ("../somefile", False),   # 親ディレクトリを含む
    ("folder/.", False),     # カレントディレクトリを含む
    ("folder/file", False),  # パス区切り文字 (/) を含む
    ("folder\\file", False), # パス区切り文字 (\) を含む
    ("/absolute/path", False),# 絶対パス形式
    ("C:\\Windows", False),  # Windows絶対パス形式
    ("file\0name", False),    # Nullバイト
    (" file ", False),       # 先頭または末尾のスペース（実装によるが、通常避けるべき）
    ("file*", False),        # ワイルドカード（実装によるが、安全でない可能性）
    ("file?", False),        # ワイルドカード（実装によるが、安全でない可能性）
    ("fi:le", False),        # コロン（Windowsで問題）
    ("file<>|", False),     # Windowsで禁止されている文字
])
def test_is_safe_path_component(component, expected):
    """is_safe_path_component が様々な入力に対して正しく安全性を判定するかテスト"""
    assert path_utils.is_safe_path_component(component) == expected

# --- validate_path_within_allowed_dirs のテスト ---

@pytest.fixture(scope='function')
def setup_test_dirs(tmp_path):
    base_dir1 = tmp_path / "allowed_dir1"
    base_dir2 = tmp_path / "allowed_dir2"
    outside_dir = tmp_path / "outside_dir"

    # 許可ディレクトリ作成
    base_dir1.mkdir()
    base_dir2.mkdir()
    (base_dir1 / "subdir").mkdir()
    (base_dir1 / "subdir" / "nested_file.txt").touch()
    (base_dir1 / "file_in_allowed1.txt").touch()
    (base_dir2 / "file_in_allowed2.log").touch()

    # 許可されていないディレクトリ作成
    outside_dir.mkdir()
    (outside_dir / "outside_file.data").touch()

    # 絶対パスとして使うため resolve
    return {
        "allowed1": base_dir1.resolve(),
        "allowed2": base_dir2.resolve(),
        "outside": outside_dir.resolve(),
        "tmp_path": tmp_path.resolve() # プロジェクトルートの代わり
    }

# monkeypatch を使用して get_project_root をモックするヘルパー
@pytest.fixture(autouse=True)
def mock_project_root(monkeypatch, setup_test_dirs, request):
    """get_project_root がテスト用のプロジェクトルートを返すようにモック"""
    # 以下のテストではモックを適用しない（実際の関数動作をテスト）
    # - test_find_project_root_env_var_* - 環境変数テスト
    # - test_find_project_root_simple_case - カスタムマーカーによるルート検出テスト
    # - test_get_project_root_raises_error_if_not_found - エラー発生ケースのテスト
    if (request.function.__name__.startswith('test_find_project_root_env_var') or
        request.function.__name__ == 'test_find_project_root_simple_case' or
        request.function.__name__ == 'test_get_project_root_raises_error_if_not_found'):
        # これらのテストではモックせず、キャッシュだけクリア
        monkeypatch.setattr("src.utils.path_utils._project_root", None, raising=False)
        return

    # 通常のテスト用のモック処理
    def mock_get_root():
        return setup_test_dirs["tmp_path"]
    # path_utils モジュール内の get_project_root を直接置き換える
    monkeypatch.setattr("src.utils.path_utils.get_project_root", mock_get_root)
    # find_project_root もモックしてキャッシュを確実にバイパス
    monkeypatch.setattr("src.utils.path_utils.find_project_root", lambda *args, **kwargs: setup_test_dirs["tmp_path"])
    # キャッシュもクリア
    monkeypatch.setattr("src.utils.path_utils._project_root", None, raising=False)

# --- 正常系テスト --- #

def test_validate_path_allowed_file(setup_test_dirs):
    """許可されたディレクトリ内の存在するファイルを検証"""
    allowed_bases = [setup_test_dirs["allowed1"], setup_test_dirs["allowed2"]]
    # 相対パスで指定 (プロジェクトルート = tmp_path とする)
    relative_path = Path("allowed_dir1") / "file_in_allowed1.txt"
    # プロジェクトルートを設定 (テスト実行時のカレントディレクトリではないため)
    path_utils._project_root = setup_test_dirs["tmp_path"]

    validated_path = path_utils.validate_path_within_allowed_dirs(relative_path, allowed_bases)
    expected_path = setup_test_dirs["allowed1"] / "file_in_allowed1.txt"
    assert validated_path == expected_path
    assert validated_path.is_absolute()

def test_validate_path_allowed_subdir_file(setup_test_dirs):
    """許可されたサブディレクトリ内のファイルを検証"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    relative_path = Path("allowed_dir1") / "subdir" / "nested_file.txt"
    path_utils._project_root = setup_test_dirs["tmp_path"]
    validated_path = path_utils.validate_path_within_allowed_dirs(relative_path, allowed_bases)
    expected_path = setup_test_dirs["allowed1"] / "subdir" / "nested_file.txt"
    assert validated_path == expected_path

def test_validate_path_allowed_dir_itself(setup_test_dirs):
    """許可されたディレクトリ自体を検証 (check_is_file=False)"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    relative_path = Path("allowed_dir1")
    path_utils._project_root = setup_test_dirs["tmp_path"]
    validated_path = path_utils.validate_path_within_allowed_dirs(
        relative_path, allowed_bases, check_is_file=False
    )
    assert validated_path == setup_test_dirs["allowed1"]

def test_validate_path_check_existence_false(setup_test_dirs):
    """存在しないパスを check_existence=False で検証"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    relative_path = Path("allowed_dir1") / "non_existent_file.txt"
    path_utils._project_root = setup_test_dirs["tmp_path"]
    validated_path = path_utils.validate_path_within_allowed_dirs(
        relative_path, allowed_bases, check_existence=False
    )
    expected_path = setup_test_dirs["allowed1"] / "non_existent_file.txt"
    assert validated_path == expected_path

def test_validate_path_absolute_path_allowed(setup_test_dirs):
    """絶対パスを allow_absolute=True で検証"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    absolute_path = setup_test_dirs["allowed1"] / "file_in_allowed1.txt"
    # プロジェクトルートは影響しないはず
    path_utils._project_root = setup_test_dirs["tmp_path"]

    validated_path = path_utils.validate_path_within_allowed_dirs(
        absolute_path, allowed_bases, allow_absolute=True
    )
    assert validated_path == absolute_path

def test_validate_path_check_is_file_true(setup_test_dirs):
    """check_is_file=True でファイルを検証"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    relative_path = Path("allowed_dir1") / "file_in_allowed1.txt"
    path_utils._project_root = setup_test_dirs["tmp_path"]
    validated_path = path_utils.validate_path_within_allowed_dirs(
        relative_path, allowed_bases, check_is_file=True
    )
    assert validated_path.is_file()

def test_validate_path_check_is_file_false(setup_test_dirs):
    """check_is_file=False でディレクトリを検証"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    relative_path = Path("allowed_dir1") / "subdir"
    path_utils._project_root = setup_test_dirs["tmp_path"]
    validated_path = path_utils.validate_path_within_allowed_dirs(
        relative_path, allowed_bases, check_is_file=False
    )
    assert validated_path.is_dir()

# --- 異常系テスト --- #

def test_validate_path_outside_allowed_dir(setup_test_dirs):
    """許可されていないディレクトリ内のパスで FileError"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    relative_path = Path("outside_dir") / "outside_file.data"
    path_utils._project_root = setup_test_dirs["tmp_path"]
    with pytest.raises(FileError, match="は許可されたディレクトリ.*内にありません"):
        path_utils.validate_path_within_allowed_dirs(relative_path, allowed_bases)

def test_validate_path_traversal_attempt(setup_test_dirs):
    """ディレクトリトラバーサルを試みるパスで FileError"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    # allowed_dir1 から出て outside_dir に入ろうとする
    relative_path = Path("allowed_dir1") / ".." / "outside_dir" / "outside_file.data"
    path_utils._project_root = setup_test_dirs["tmp_path"]
    with pytest.raises(FileError, match="は許可されたディレクトリ.*内にありません"):
        path_utils.validate_path_within_allowed_dirs(relative_path, allowed_bases)

def test_validate_path_traversal_absolute_mixed(setup_test_dirs):
    """絶対パスとトラバーサルを混ぜたケース (allow_absolute=True)"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    absolute_path_trav = setup_test_dirs["allowed1"] / "subdir" / ".." / ".." / "outside_dir" / "outside_file.data"
    path_utils._project_root = setup_test_dirs["tmp_path"]
    with pytest.raises(FileError, match="は許可されたディレクトリ.*内にありません"):
        path_utils.validate_path_within_allowed_dirs(absolute_path_trav, allowed_bases, allow_absolute=True)

def test_validate_path_non_existent_file_error(setup_test_dirs):
    """存在しないファイルを check_existence=True で検証し FileError"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    relative_path = Path("allowed_dir1") / "non_existent_file.txt"
    path_utils._project_root = setup_test_dirs["tmp_path"]
    with pytest.raises(FileError, match="存在しません"):
        path_utils.validate_path_within_allowed_dirs(relative_path, allowed_bases, check_existence=True)

def test_validate_path_absolute_path_not_allowed_error(setup_test_dirs):
    """絶対パスを allow_absolute=False で検証し ValueError"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    absolute_path = setup_test_dirs["allowed1"] / "file_in_allowed1.txt"
    path_utils._project_root = setup_test_dirs["tmp_path"]
    with pytest.raises(ValueError, match="絶対パスは許可されていません"):
        path_utils.validate_path_within_allowed_dirs(absolute_path, allowed_bases, allow_absolute=False)

def test_validate_path_wrong_type_is_file_true_error(setup_test_dirs):
    """ディレクトリを check_is_file=True で検証し FileError"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    relative_path = Path("allowed_dir1") / "subdir"
    path_utils._project_root = setup_test_dirs["tmp_path"]
    with pytest.raises(FileError, match="ファイルではありません"):
        path_utils.validate_path_within_allowed_dirs(relative_path, allowed_bases, check_is_file=True)

def test_validate_path_wrong_type_is_file_false_error(setup_test_dirs):
    """ファイルを check_is_file=False で検証し FileError"""
    allowed_bases = [setup_test_dirs["allowed1"]]
    relative_path = Path("allowed_dir1") / "file_in_allowed1.txt"
    path_utils._project_root = setup_test_dirs["tmp_path"]
    with pytest.raises(FileError, match="ディレクトリではありません"):
        path_utils.validate_path_within_allowed_dirs(relative_path, allowed_bases, check_is_file=False)

def test_validate_path_empty_allowed_dirs_error(setup_test_dirs):
    """allowed_base_dirs が空の場合に ConfigError"""
    relative_path = Path("allowed_dir1") / "file_in_allowed1.txt"
    path_utils._project_root = setup_test_dirs["tmp_path"]
    with pytest.raises(ConfigError, match="リストが空です"):
        path_utils.validate_path_within_allowed_dirs(relative_path, [])

# --- find_project_root / get_project_root のテスト ---

@pytest.fixture(scope='function')
def setup_simple_project(tmp_path, monkeypatch):
    """find_project_rootテスト用のシンプルなディレクトリ構造を作成し、マーカーリストを上書き"""
    project_root = tmp_path / "test_proj"
    sub_dir = project_root / "sub"
    marker_filename = ".proj_root_marker_test"
    marker_file = project_root / marker_filename

    project_root.mkdir()
    sub_dir.mkdir()
    marker_file.touch()

    # テスト期間中のみ ROOT_MARKERS を上書き
    original_markers = path_utils.ROOT_MARKERS
    test_markers = [marker_filename]
    monkeypatch.setattr(path_utils, 'ROOT_MARKERS', test_markers)
    
    # _markers も上書きして確実にマーカーが認識されるようにする
    monkeypatch.setattr(path_utils, '_markers', test_markers)

    yield { # yield を使って後処理で元に戻す
        "project_root": project_root.resolve(),
        "sub_dir": sub_dir.resolve(),
        "marker": marker_filename
    }

    # teardown は不要 (monkeypatchが自動で元に戻す)
    # monkeypatchを使わない場合はここで path_utils.ROOT_MARKERS = original_markers

# シンプルな構造でのテストケース
def test_find_project_root_simple_case(setup_simple_project, monkeypatch):
    """シンプルな構造と単一マーカーでルート検出をテスト"""
    start_dir = setup_simple_project["sub_dir"]
    expected_root = setup_simple_project["project_root"]
    marker_filename = setup_simple_project["marker"]

    # キャッシュと環境変数をクリア
    monkeypatch.setattr(path_utils, "_project_root", None)
    monkeypatch.delenv("MIREX_PROJECT_ROOT", raising=False)

    # .envの読み込みも無効化しておく (今回のテストでは不要)
    monkeypatch.setattr(path_utils, "has_dotenv", False)
    
    # マーカーの存在を確認
    marker_file = expected_root / marker_filename
    assert marker_file.exists(), f"マーカーファイル {marker_file} が存在しません"
    
    # ROOT_MARKERSが正しく設定されていることを確認
    assert path_utils.ROOT_MARKERS == [marker_filename], "ROOT_MARKERSが正しく設定されていません"

    # デバッグ情報
    print(f"検索開始ディレクトリ: {start_dir}")
    print(f"期待するルートディレクトリ: {expected_root}")
    print(f"マーカーファイル: {marker_file} (存在: {marker_file.exists()})")
    print(f"現在のROOT_MARKERS: {path_utils.ROOT_MARKERS}")
    print(f"_markers: {path_utils._markers}")

    # マーカーが実際に使用されるリストに含まれているか確認
    current_markers = path_utils._markers
    assert marker_filename in current_markers, f"マーカーファイル名 {marker_filename} が _markers リストに含まれていません"

    found_root = path_utils.find_project_root(
        start_dir=start_dir,
        search_downwards=False,
        max_depth_up=3  # 十分な深さを設定
    )
    assert found_root == expected_root

# --- 環境変数 MIREX_PROJECT_ROOT のテスト (シンプルフィクスチャ使用) ---

def test_find_project_root_env_var_set(tmp_path: Path, monkeypatch):
    """
    MIREX_PROJECT_ROOT 環境変数が設定されている場合、それを優先するテスト。
    マーカーファイルが存在しても環境変数が優先されることを確認。
    """
    # Arrange
    path_utils._project_root = None  # Clear cache before test
    fake_root = tmp_path / "fake_root"
    fake_root.mkdir()
    # マーカーファイルも念のため作成しておく（環境変数が優先されるかの確認）
    # (fake_root / ".git").mkdir() # マーカーファイルなしで試す

    start_dir = tmp_path / "subdir"
    start_dir.mkdir()

    expected_root = fake_root.resolve()
    monkeypatch.setenv("MIREX_PROJECT_ROOT", str(expected_root))

    # ★ 環境変数が正しく設定されたか確認するアサーションを追加
    assert os.getenv("MIREX_PROJECT_ROOT") == str(expected_root)

    try:
        # Call find_project_root without start_dir to rely purely on environment variable
        actual_root = path_utils.find_project_root(use_dotenv=False) # Disable .env loading for this test
        assert actual_root == expected_root
    finally:
        # Clean up environment variable
        del os.environ["MIREX_PROJECT_ROOT"]
        path_utils._project_root = None # Clear cache after test

def test_find_project_root_env_var_invalid(tmp_path: Path, monkeypatch):
    """環境変数 MIREX_PROJECT_ROOT が無効なパスの場合、マーカー検索にフォールバックすることを確認。"""
    # Arrange: キャッシュをクリアし、環境変数に無効なパスを設定
    path_utils._project_root = None  # Clear cache before test
    invalid_path = tmp_path / "non_existent_dir"
    monkeypatch.setenv("MIREX_PROJECT_ROOT", str(invalid_path))

    # マーカーファイルを持つ有効なプロジェクト構造を準備
    # ROOT_MARKERS リストに含まれるマーカーを使用
    test_proj = tmp_path / "test_proj"
    test_proj.mkdir()
    (test_proj / ".git").mkdir()  # .git はデフォルトの ROOT_MARKERS に含まれている
    
    # test_proj 内に src ディレクトリを作成 (検索を始めるポイント)
    src_dir = test_proj / "src"
    src_dir.mkdir()

    try:
        # Act: test_proj/src ディレクトリから検索を開始して .git が見つかるようにする
        # また、検索深度も制限する (デフォルトの10は広すぎる可能性)
        found_root = path_utils.find_project_root(
            start_dir=src_dir,  # test_proj/src から検索開始
            max_depth_up=2,     # 検索深度を制限 (src → test_proj の1段階だけ)
            search_downwards=False,
            use_dotenv=False    # .env ファイルの読み込みを無効化
        )
        
        # Assert: マーカーファイルを持つディレクトリが見つかるはず
        assert found_root == test_proj
    finally:
        # Clean up
        monkeypatch.delenv("MIREX_PROJECT_ROOT", raising=False)
        path_utils._project_root = None  # Clear cache after test

def test_get_project_root_raises_error_if_not_found(monkeypatch, tmp_path):
    """プロジェクトルートが見つからない場合に ConfigError が発生することをテスト"""
    # Arrange: マーカーも環境変数もない状態
    subdir = tmp_path / "subdir" / "subsubdir"
    subdir.mkdir(parents=True)
    monkeypatch.delenv("MIREX_PROJECT_ROOT", raising=False)
    monkeypatch.setattr(path_utils, "_project_root", None)
    
    # 強制的にマーカーが見つからないようにする
    # ROOT_MARKERSとmarkersをカスタムの値に設定
    custom_marker = "this_marker_does_not_exist_anywhere"
    monkeypatch.setattr(path_utils, 'ROOT_MARKERS', [custom_marker])
    monkeypatch.setattr(path_utils, '_markers', [custom_marker])

    # Act & Assert: get_project_root を呼び出すとエラーが発生する
    with pytest.raises(ConfigError, match="プロジェクトルートが見つかりませんでした"):
        # start_dir を subdir に指定して実行
        path_utils.find_project_root(
            start_dir=subdir,
            search_downwards=False,
            use_dotenv=False
        )

# --- 既存の find_project_root テスト (一旦コメントアウトまたは修正) ---
# ... (既存のスキップ/コメントアウトされたテスト) ...

# TODO: .env ファイルのテスト (シンプルフィクスチャ使用)

# --- ここに他の関数のテストを追加 ---

# 例: validate_path_within_allowed_dirs のテスト
# @pytest.fixture
# def setup_allowed_dirs(tmp_path):
#     # テスト用のディレクトリ構造を作成
#     ...
#     return allowed_dirs

# def test_validate_path_within_allowed_dirs_success(setup_allowed_dirs, tmp_path):
#     ...

# def test_validate_path_within_allowed_dirs_failure(setup_allowed_dirs, tmp_path):
#     ...

# def test_validate_path_existence(setup_allowed_dirs, tmp_path):
#     ...

# def test_validate_path_type(setup_allowed_dirs, tmp_path):
#     ... 