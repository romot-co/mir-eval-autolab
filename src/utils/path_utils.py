#!/usr/bin/env python3
"""
パス操作ユーティリティモジュール

このモジュールは、プロジェクト内でのパス操作に関するユーティリティ関数を提供します。
絶対パスの解決、プロジェクトルートの検出、ディレクトリの作成、パスの検証などの機能を含みます。
"""

import os
import sys
import re
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any
import warnings
from dotenv import load_dotenv, find_dotenv

# .envファイルのサポートを追加（利用可能な場合）
try:
    import dotenv
    has_dotenv = True
except ImportError:
    has_dotenv = False

# カスタム例外をインポート (相対パスから絶対パスに変更する可能性を考慮)
# 例: from src.utils.exception_utils import FileError, ConfigError
# 現状の相対パスのままにする場合は以下:
try:
    from .exception_utils import FileError, ConfigError
except ImportError:
    # 直接実行時などのフォールバック
    class FileError(OSError): pass
    class ConfigError(ValueError): pass

logger = logging.getLogger(__name__)

# プロジェクトルートを特定するマーカーファイル/ディレクトリ
# 複数のファイル/ディレクトリで構成されるプロジェクトに対応
ROOT_MARKERS = [
    'pyproject.toml', # Pythonプロジェクトの標準的なマーカー
    'mcp_server.py',
    '.git',           # Gitリポジトリのルート
    'src/detectors',  # プロジェクト固有の構造
    'src/evaluation', # プロジェクト固有の構造
    'config.yaml',    # プロジェクト固有の設定ファイル
]

# キャッシュしたプロジェクトルートパス
_project_root: Optional[Path] = None
_markers = ROOT_MARKERS + ["pyproject.toml", ".git"]
has_dotenv = find_dotenv() is not None

def load_environment_variables(dotenv_path: Optional[Union[str, Path]] = None, *, use_project_root_env: bool = True) -> bool:
    """
    .envファイルから環境変数を読み込む。

    指定されたパス、またはプロジェクトルートの .env を読み込む。

    Parameters
    ----------
    dotenv_path : Optional[Union[str, Path]], optional
        読み込む .env ファイルのパス。Noneの場合、プロジェクトルートの .env を探す。
    use_project_root_env : bool, optional
        dotenv_pathが指定されていない場合に、プロジェクトルートの.envを探して読み込むか, by default True

    Returns
    -------
    bool
        環境変数が読み込まれた場合は True、そうでない場合は False。
    """
    if not has_dotenv:
        logger.debug("dotenv パッケージが見つからないため、.env ファイルの読み込みをスキップします。")
        return False

    load_path: Optional[Path] = None
    if dotenv_path:
        load_path = Path(dotenv_path).resolve()
        if not load_path.exists():
            logger.warning(f"指定された .env パスが見つかりません: {load_path}")
            load_path = None # 見つからなければNoneに戻す
    elif use_project_root_env:
        try:
            # プロジェクトルートを取得 (キャッシュを利用)
            root = get_project_root(use_cache=True) # 通常はキャッシュされたものを使う
            default_env_path = root / '.env'
            if default_env_path.exists():
                load_path = default_env_path
            else:
                 # .env.example があれば情報ログを出す
                 example_path = root / '.env.example'
                 if example_path.exists():
                     logger.info(f".env ファイルが見つかりません ({default_env_path})。.env.example をコピーして設定できます。")
                # else: # .env も .env.example もない場合は特にログ不要
                #     logger.debug(f"プロジェクトルートに .env も .env.example も見つかりません: {root}")
        except ConfigError:
            # プロジェクトルートが見つからない場合は、.env 読み込みも失敗
            logger.warning("プロジェクトルートが未確定のため、プロジェクトルートの .env ファイルの読み込みをスキップします。")
        except Exception as e:
             logger.error(f"プロジェクトルート下の .env 検索中に予期せぬエラー: {e}")

    if load_path and load_path.is_file():
        try:
            loaded = dotenv.load_dotenv(dotenv_path=load_path, override=True) # override=True で既存の環境変数を上書き
            if loaded:
                logger.info(f"{load_path} から環境変数を読み込みました。")
                return True
            else:
                # .envファイルは存在したが空だった場合など
                logger.debug(f"{load_path} は存在しますが、環境変数は読み込まれませんでした。")
                return False
        except Exception as e:
            logger.error(f"{load_path} の読み込み中にエラーが発生しました: {e}", exc_info=True)
            return False
    else:
        if dotenv_path: # path指定があった場合のみ、見つからなかったログを出す
             logger.debug(f"指定された .env パスが見つからなかったか、ファイルではありません: {load_path}")
        # else: # プロジェクトルートに .env がないのは通常なのでログ不要
        #     logger.debug("読み込むべき .env ファイルが見つかりませんでした。")
        return False

def find_project_root(
    start_dir: Optional[Union[str, Path]] = None,
    markers: Optional[List[str]] = None,
    use_environment: bool = True,
    use_dotenv: bool = True,
    search_upwards: bool = True,
    search_downwards: bool = False,
    max_depth_up: int = 10,
    max_depth_down: int = 3,
) -> Optional[Path]:
    """
    Find the project root directory by searching for specific marker files or using environment variables.

    Priority order:
    1. Cached result (`_project_root`).
    2. MIREX_PROJECT_ROOT environment variable (if `use_environment` is True).
    3. Search upwards from `start_dir` for markers (if `search_upwards` is True).
    4. Search downwards from `start_dir` for markers (if `search_downwards` is True).
    5. Return `start_dir` if no root is found and no search performed.
    6. Return None if searches were performed but no root found.

    Args:
        start_dir: The directory to start searching from. Defaults to the current working directory.
        markers: A list of marker filenames to search for. Defaults to `_markers`.
        use_environment: Whether to check the MIREX_PROJECT_ROOT environment variable.
        use_dotenv: Whether to load .env file if python-dotenv is installed and .env exists.
                   Requires python-dotenv to be installed.
        search_upwards: Whether to search upwards from `start_dir`.
        search_downwards: Whether to search downwards from `start_dir`.
        max_depth_up: Maximum number of parent directories to search upwards.
        max_depth_down: Maximum depth to search downwards.

    Returns:
        The absolute Path to the project root directory, or None if not found.
    """
    global _project_root
    global has_dotenv # Use the globally checked status

    if _project_root is not None:
        return _project_root

    if start_dir is None:
        start_dir = Path.cwd()
    else:
        start_dir = Path(start_dir).resolve()

    if not start_dir.is_dir():
        warnings.warn(f"Start directory '{start_dir}' is not a valid directory. Using CWD.")
        start_dir = Path.cwd()

    current_markers = markers if markers is not None else _markers

    # 1. Check environment variable MIREX_PROJECT_ROOT
    if use_environment:
        if use_dotenv and has_dotenv:
            load_dotenv(override=False) # Load .env but don't override existing env vars

        project_root_env_var = os.getenv("MIREX_PROJECT_ROOT")
        if project_root_env_var:
            try:
                root_path = Path(project_root_env_var).resolve()
                if root_path.is_dir():
                    _project_root = root_path
                    return _project_root
                else:
                    warnings.warn(
                        f"MIREX_PROJECT_ROOT ('{project_root_env_var}') resolves to "
                        f"'{root_path}', which is not a valid directory. Ignoring."
                    )
            except Exception as e:
                warnings.warn(
                    f"Error resolving MIREX_PROJECT_ROOT ('{project_root_env_var}'): {e}. Ignoring."
                )

    # 2. Search upwards
    root_found = None
    current_dir = start_dir

    for level in range(max_depth_up + 1): # start_dir自体もチェックするため +1
        for marker in current_markers:
            marker_path = current_dir / marker
            if marker_path.exists():
                logger.info(f"プロジェクトルートを検出しました (マーカー '{marker}' 発見): {current_dir}")
                root_found = current_dir
                break

        # 親ディレクトリへ移動
        parent_dir = current_dir.parent
        if parent_dir == current_dir:
            # ファイルシステムのルートまで到達
            break
        current_dir = parent_dir

    if root_found:
        logger.info(f"プロジェクトルートを検出しました (マーカー検索結果): {root_found}")
        _project_root = root_found
        return _project_root

    # 見つからなかった場合
    msg = f"プロジェクトルートが見つかりませんでした (環境変数MIREX_PROJECT_ROOTもマーカー検索も失敗)。検索開始点: {start_dir}"
    logger.critical(msg)
    raise ConfigError(msg)

def get_project_root(use_cache: bool = True) -> Path:
    """
    プロジェクトルートディレクトリを取得する。キャッシュを利用可能。

    Parameters
    ----------
    use_cache : bool, optional
        グローバルキャッシュを利用するかどうか, by default True

    Returns
    -------
    Path
        プロジェクトルートディレクトリのパス

    Raises
    ------
    ConfigError
        プロジェクトルートが見つからない場合
    """
    global _project_root
    if use_cache and _project_root is not None:
        return _project_root

    # find_project_root でルートを検索 (エラーはここで発生する可能性)
    try:
        calculated_root = find_project_root()
        if use_cache:
            _project_root = calculated_root # キャッシュを更新
        return calculated_root
    except ConfigError as e:
        # エラーをそのまま再発生
        raise e
    except Exception as e:
         # 予期せぬエラー
         logger.critical(f"プロジェクトルート取得中に予期せぬエラー: {e}", exc_info=True)
         raise ConfigError(f"プロジェクトルート取得中に予期せぬエラー: {e}") from e


def get_absolute_path(relative_or_absolute_path: Union[str, Path],
                      base_dir: Optional[Path] = None) -> Path:
    """
    与えられたパスを絶対パスに変換する。
    相対パスの場合、指定されたベースディレクトリ（デフォルトはプロジェクトルート）からの相対パスとして解決する。

    Parameters
    ----------
    relative_or_absolute_path : Union[str, Path]
        変換するパス（相対または絶対）。
    base_dir : Optional[Path], optional
        相対パスの基準となるディレクトリ。Noneの場合、プロジェクトルートを使用する, by default None

    Returns
    -------
    Path
        解決された絶対パス。
    """
    path_obj = Path(relative_or_absolute_path)

    if path_obj.is_absolute():
        # 既に絶対パスの場合はそのまま返す
        return path_obj.resolve()
    else:
        # 相対パスの場合
        if base_dir is None:
            base_dir = get_project_root() # プロジェクトルートを基準にする
        return (base_dir / path_obj).resolve()

def ensure_dir(dir_path: Union[str, Path], *, check_writable: bool = True) -> Path:
    """
    指定されたパスのディレクトリが存在することを確認し、存在しない場合は作成する。
    オプションで書き込み可能かどうかもチェックする。

    Parameters
    ----------
    dir_path : Union[str, Path]
        確認または作成するディレクトリのパス。
    check_writable : bool, optional
        ディレクトリが書き込み可能かどうかのチェックを行うか, by default True

    Returns
    -------
    Path
        確認または作成されたディレクトリの絶対パス。

    Raises
    ------
    FileError
        ディレクトリの作成に失敗した場合、または書き込み不可の場合。
    TypeError
        dir_path が文字列またはPathオブジェクトでない場合。
    """
    if not isinstance(dir_path, (str, Path)):
        raise TypeError(f"dir_path must be a string or Path object, got {type(dir_path)}")

    try:
        path = Path(dir_path).resolve()
        # ディレクトリが存在しない場合は作成
        if not path.exists():
            logger.info(f"ディレクトリが存在しないため作成します: {path}")
            path.mkdir(parents=True, exist_ok=True)
        # ディレクトリでない場合はエラー
        elif not path.is_dir():
            raise FileError(f"指定されたパスはディレクトリではありません: {path}")

        # 書き込み可能チェック (オプション)
        if check_writable:
            # 一時ファイルを作成してテスト
            try:
                with tempfile.NamedTemporaryFile(prefix='write_test_', dir=str(path)) as tf:
                    logger.debug(f"ディレクトリへの書き込みテスト成功: {path}")
            except OSError as e:
                logger.error(f"ディレクトリへの書き込みテストに失敗しました: {path} - {e}", exc_info=True)
                raise FileError(f"ディレクトリに書き込み権限がありません: {path}") from e

        return path
    except OSError as e:
        logger.error(f"ディレクトリの確認または作成中にエラーが発生しました: {dir_path} - {e}", exc_info=True)
        raise FileError(f"ディレクトリの確認または作成に失敗しました: {dir_path}") from e
    except Exception as e:
        logger.error(f"ディレクトリ確認中に予期せぬエラー: {dir_path} - {e}", exc_info=True)
        raise FileError(f"ディレクトリ確認中に予期せぬエラーが発生しました: {dir_path}") from e

# --- Workspace Path --- #
_workspace_dir: Optional[Path] = None

def get_workspace_dir(config: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> Path:
    """
    MCP サーバーのワークスペースディレクトリを取得する。

    優先順位:
    1. キャッシュされたパス (_workspace_dir) があればそれを返す (use_cache=Trueの場合)。
    2. 環境変数 MIREX_WORKSPACE が設定されていれば、そのパスを使用する。
    3. デフォルトのワークスペースパス (プロジェクトルート/.mcp_server_data) を使用する。

    取得したパスのディレクトリが存在することを確認し、必要であれば作成する。
    また、そのディレクトリが書き込み可能であることを確認する。

    Parameters
    ----------
    config : Optional[Dict[str, Any]], optional
        アプリケーション設定辞書。現在は直接使用されないが、将来のために残す。
    use_cache : bool, optional
        グローバルキャッシュを利用するかどうか, by default True

    Returns
    -------
    Path
        ワークスペースディレクトリの絶対パス。

    Raises
    ------
    ConfigError
        環境変数で指定されたパスが無効な場合、またはプロジェクトルートが見つからない場合。
    FileError
        ディレクトリの作成または書き込みテストに失敗した場合。
    """
    global _workspace_dir
    if use_cache and _workspace_dir is not None:
        return _workspace_dir

    workspace_path_str: Optional[str] = os.environ.get('MIREX_WORKSPACE')
    project_root = get_project_root() # プロジェクトルートは必須

    workspace_dir: Path
    if workspace_path_str:
        logger.info(f"環境変数 MIREX_WORKSPACE からワークスペースパスを取得します: {workspace_path_str}")
        path_obj = Path(workspace_path_str)
        # 環境変数で指定されたパスが絶対パスか相対パスかによって処理を分ける
        if path_obj.is_absolute():
            workspace_dir = path_obj
        else:
            # 相対パスの場合はプロジェクトルート基準で解決
            workspace_dir = (project_root / path_obj).resolve()
            logger.debug(f"相対パスをプロジェクトルート基準で解決しました: {workspace_dir}")
    else:
        # デフォルトのパス (プロジェクトルート直下の隠しディレクトリ)
        default_workspace_name = ".mcp_server_data"
        workspace_dir = (project_root / default_workspace_name).resolve()
        logger.info(f"デフォルトのワークスペースパスを使用します: {workspace_dir}")

    # ディレクトリの存在確認、作成、書き込みテスト
    try:
        resolved_workspace_dir = ensure_dir(workspace_dir, check_writable=True)
    except (FileError, TypeError) as e:
        # ensure_dir からのエラーを ConfigError にラップして再送出 (設定に起因する問題として扱う)
        raise ConfigError(f"ワークスペースディレクトリの設定またはアクセスに問題があります: {workspace_dir}. 詳細: {e}") from e

    if use_cache:
        _workspace_dir = resolved_workspace_dir

    return resolved_workspace_dir

# --- Output Base Path --- #
_output_base_dir: Optional[Path] = None

def get_output_base_dir(config: Dict[str, Any], use_cache: bool = True) -> Path:
    """
    全ての実行結果が出力されるベースディレクトリ (`output_base`) を取得する。

    優先順位:
    1. キャッシュされたパス (_output_base_dir) があればそれを返す (use_cache=Trueの場合)。
    2. 環境変数 MIREX_OUTPUT_BASE が設定されていれば、そのパスを使用する。
    3. `config['paths']['output_base']` で指定されたパスを使用する。
    4. デフォルトのパス (プロジェクトルート/output) を使用する。

    取得したパスのディレクトリが存在することを確認し、必要であれば作成する。
    また、そのディレクトリが書き込み可能であることを確認する。

    Parameters
    ----------
    config : Dict[str, Any]
        アプリケーション設定辞書。`paths.output_base` を参照するために必須。
    use_cache : bool, optional
        グローバルキャッシュを利用するかどうか, by default True

    Returns
    -------
    Path
        出力ベースディレクトリの絶対パス。

    Raises
    ------
    ConfigError
        設定や環境変数で指定されたパスが無効な場合、またはプロジェクトルートが見つからない場合。
    FileError
        ディレクトリの作成または書き込みテストに失敗した場合。
    """
    global _output_base_dir
    if use_cache and _output_base_dir is not None:
        return _output_base_dir

    output_path_str: Optional[str] = os.environ.get('MIREX_OUTPUT_BASE')
    source = "環境変数 MIREX_OUTPUT_BASE"

    if not output_path_str:
        output_path_str = config.get('paths', {}).get('output_base')
        source = "設定ファイル (paths.output_base)"
        if not output_path_str:
            output_path_str = 'output' # デフォルト値
            source = "デフォルト値 ('output')"

    logger.info(f"{source} から出力ベースパスを取得します: {output_path_str}")

    project_root = get_project_root() # プロジェクトルートは必須
    path_obj = Path(output_path_str)

    output_dir: Path
    if path_obj.is_absolute():
        output_dir = path_obj
    else:
        # 相対パスの場合はプロジェクトルート基準で解決
        output_dir = (project_root / path_obj).resolve()
        logger.debug(f"相対パスをプロジェクトルート基準で解決しました: {output_dir}")

    # ディレクトリの存在確認、作成、書き込みテスト
    try:
        resolved_output_dir = ensure_dir(output_dir, check_writable=True)
    except (FileError, TypeError) as e:
        raise ConfigError(f"出力ベースディレクトリの設定またはアクセスに問題があります: {output_dir}. 詳細: {e}") from e

    if use_cache:
        _output_base_dir = resolved_output_dir

    return resolved_output_dir

# --- Workspace Subdirectories --- #

def get_improved_versions_dir(config: Dict[str, Any]) -> Path:
    """
    AIによって生成された改善版コードを保存するディレクトリ (`improved_versions`) のパスを取得する。
    このディレクトリはワークスペース内に配置される。

    Parameters
    ----------
    config : Dict[str, Any]
        アプリケーション設定辞書 (`paths.improved_versions` を参照)。

    Returns
    -------
    Path
        改善版コードディレクトリの絶対パス。

    Raises
    ------
    ConfigError
        設定やワークスペースディレクトリに問題がある場合。
    FileError
        ディレクトリの作成または書き込みテストに失敗した場合。
    """
    versions_subdir_name = config.get('paths', {}).get('improved_versions', 'improved_versions')
    if not isinstance(versions_subdir_name, str) or not is_safe_path_component(versions_subdir_name):
         raise ConfigError(f"設定内の 'paths.improved_versions' が不正です: {versions_subdir_name}")

    try:
        workspace = get_workspace_dir(config) # ワークスペースパスを取得 (キャッシュ利用可能)
        versions_dir = workspace / versions_subdir_name
        # このディレクトリも存在確認と書き込みテストを行う
        return ensure_dir(versions_dir, check_writable=True)
    except (ConfigError, FileError, TypeError) as e:
        # エラーをラップして再送出
        raise ConfigError(f"改善バージョンディレクトリの設定またはアクセスに問題があります: {versions_subdir_name}. 詳細: {e}") from e

def get_db_dir(config: Dict[str, Any]) -> Path:
    """
    データベースファイル (mcp_server_state.db など) を格納するディレクトリ (`db`) のパスを取得する。
    このディレクトリはワークスペース内に配置される。

    Parameters
    ----------
    config : Dict[str, Any]
        アプリケーション設定辞書 (`paths.db` を参照)。

    Returns
    -------
    Path
        データベースディレクトリの絶対パス。

    Raises
    ------
    ConfigError
        設定やワークスペースディレクトリに問題がある場合。
    FileError
        ディレクトリの作成または書き込みテストに失敗した場合。
    """
    db_subdir_name = config.get('paths', {}).get('db', 'db')
    if not isinstance(db_subdir_name, str) or not is_safe_path_component(db_subdir_name):
         raise ConfigError(f"設定内の 'paths.db' が不正です: {db_subdir_name}")

    try:
        workspace = get_workspace_dir(config) # ワークスペースパスを取得 (キャッシュ利用可能)
        db_dir = workspace / db_subdir_name
        # このディレクトリも存在確認と書き込みテストを行う
        return ensure_dir(db_dir, check_writable=True)
    except (ConfigError, FileError, TypeError) as e:
        raise ConfigError(f"DBディレクトリの設定またはアクセスに問題があります: {db_subdir_name}. 詳細: {e}") from e

# --- Source Code Paths --- #

def get_detectors_src_dir(config: Optional[Dict[str, Any]] = None) -> Path:
    """検出器のソースコードが格納されているディレクトリ (`detectors_src`) のパスを取得する。"""
    # config があればそこから、なければデフォルト値
    src_dir_name = 'src/detectors' # デフォルトはプロジェクトルートからの相対パス
    if config:
        src_dir_name = config.get('paths', {}).get('detectors_src', src_dir_name)

    project_root = get_project_root()
    detectors_dir = project_root / src_dir_name
    if not detectors_dir.is_dir():
         # ここでは ensure_dir しない方が良い (ソースディレクトリが存在しないのは設定ミス)
         raise ConfigError(f"検出器ソースディレクトリが見つかりません: {detectors_dir}")
    return detectors_dir.resolve() # 絶対パスを返すように修正

def get_detector_path(detector_name: str, config: Dict[str, Any], version: Optional[str] = None) -> Path:
    """
    指定された検出器の Python ファイルパスを取得する。
    改善版が存在する場合はそちらを優先する。

    Parameters
    ----------
    detector_name : str
        検出器のクラス名またはファイル名 (例: 'PZSTDDetector' または 'pzstd_detector.py')。
    config : Dict[str, Any]
        アプリケーション設定辞書。`get_improved_versions_dir` と `get_detectors_src_dir` を呼び出すために必要。
    version : Optional[str], optional
        取得したい改善版のバージョン (例: 'v3')。None の場合は最新の改善版またはオリジナルを探す。

    Returns
    -------
    Path
        検出器ファイルの絶対パス。

    Raises
    ------
    FileNotFoundError
        指定された検出器ファイルが見つからない場合。
    ConfigError
        パス設定に問題がある場合。
    ValueError
        detector_name や version が不正な形式の場合。
    """
    if not detector_name or not isinstance(detector_name, str):
        raise ValueError("detector_name must be a non-empty string.")
    if version is not None and (not isinstance(version, str) or not version.startswith('v') or not version[1:].isdigit()):
         raise ValueError("version must be None or a string like 'v1', 'v2', etc.")

    # detector_name からファイル名を推定 (クラス名 -> スネークケース.py)
    if detector_name.endswith('.py'):
        filename = detector_name
        class_name_part = detector_name[:-3] # .py を除去
    else:
        # クラス名をスネークケースに変換 (より堅牢な方法を検討)
        # 例: 大文字の前にアンダースコアを挿入 (先頭と連続大文字を除く)
        s1 = re.sub(r"(?<!^)(?=[A-Z])", "_", detector_name).lower()
        filename = f"{s1}.py"
        class_name_part = detector_name # 元のクラス名

    logger.debug(f"検出器 '{detector_name}' のファイル名を推定: {filename}")

    try:
        # 改善版ディレクトリを取得 (ワークスペース内)
        improved_versions_dir = get_improved_versions_dir(config)
        detector_versions_dir = improved_versions_dir / class_name_part # クラス名ベースのサブディレクトリ

        target_path: Optional[Path] = None

        # 改善版ディレクトリを確認
        if detector_versions_dir.is_dir():
            if version:
                # 特定バージョンの改善版を探す
                version_filename = f"{filename.rsplit('.', 1)[0]}_{version}.py"
                potential_path = detector_versions_dir / version_filename
                if potential_path.is_file():
                    target_path = potential_path
                    logger.info(f"指定されたバージョンの改善版を使用します ({version}): {target_path}")
                else:
                    # ★ 修正: 特定バージョンが見つからない場合はエラー
                    logger.error(f"指定されたバージョン '{version}' の改善版ファイルが見つかりません: {potential_path}")
                    # 修正: f-string を正しく閉じる
                    raise FileNotFoundError(
                        f"指定されたバージョン '{version}' の検出器ファイルが見つかりません。\n"
                        f"検索パス: {potential_path}"
                    )
            else:
                # 最新バージョンの改善版を探す
                latest_version_num = -1
                latest_file_path = None
                version_pattern = re.compile(rf"^{re.escape(filename.rsplit('.', 1)[0])}_v(\d+)\.py$")
                try:
                    for item in detector_versions_dir.iterdir():
                        if item.is_file():
                            match = version_pattern.match(item.name)
                            if match:
                                v_num = int(match.group(1))
                                if v_num > latest_version_num:
                                    latest_version_num = v_num
                                    latest_file_path = item
                except OSError as e:
                    logger.warning(f"改善版ディレクトリの読み取り中にエラー ({detector_versions_dir}): {e}. オリジナルを探します。")
                    # エラーが発生した場合、最新版は見つからなかったとして次に進む

                if latest_file_path:
                    target_path = latest_file_path
                    logger.info(f"最新バージョンの改善版 (v{latest_version_num}) を使用します: {target_path}")

        # 改善版が見つかった場合はそれを返す
        if target_path and target_path.is_file():
            return target_path.resolve()
        elif version: # target_path が None だが version が指定されていた場合 (上記のエラー発生ケース)
             # このパスには到達しないはず (上で FileNotFoundError が raise されるため)
             # 念のためエラーを追加
             raise FileNotFoundError(f"指定されたバージョン '{version}' の検出器ファイルが見つかりませんでした (予期せぬ状態)。")

        # 特定バージョン指定がなく、改善版が見つからなかった場合、または改善版ディレクトリが存在しない場合
        # オリジナルの検出器を探す
        detectors_src_dir = get_detectors_src_dir(config)
        original_path = detectors_src_dir / filename
        if original_path.is_file():
            logger.info(f"オリジナルの検出器ファイルを使用します: {original_path}")
            return original_path.resolve()
        else:
            # オリジナルも見つからない場合
            # ★ 修正: エラーメッセージをより詳細に
            searched_paths = f"改善版検索パス: {detector_versions_dir}\nオリジナル検索パス: {original_path}"
            if version:
                 # 特定バージョンを探して見つからなかった場合 (このパスには到達しないはずだが念のため)
                 msg = f"指定されたバージョン '{version}' の検出器ファイルも、オリジナルのファイルも見つかりませんでした。"
            elif detector_versions_dir.is_dir() and latest_version_num != -1:
                 # 最新版を探したがオリジナルも見つからなかった (通常ありえない？)
                 msg = f"最新の改善版 (v{latest_version_num}) もオリジナルの検出器ファイルも見つかりませんでした。"
            elif detector_versions_dir.is_dir():
                 # 改善版ディレクトリはあるが、バージョンファイルが見つからず、オリジナルもない
                 msg = f"改善版ディレクトリにバージョンファイルが見つからず、オリジナルの検出器ファイルも見つかりませんでした。"
            else:
                 # 改善版ディレクトリがなく、オリジナルもない
                 msg = f"改善版ディレクトリもオリジナルの検出器ファイルも見つかりませんでした。"

            raise FileNotFoundError(f"{msg}\n{searched_paths}")

    except (ConfigError, FileError, ValueError) as e:
         # 設定やパスアクセスに関するエラー
         # ★ 修正: FileError は ValueError ではなく ConfigError にラップする
         if isinstance(e, FileError):
              raise ConfigError(f"検出器パスの取得中にファイルアクセスエラーが発生しました: {e}") from e
         elif isinstance(e, ValueError):
              raise ValueError(f"検出器パスの取得中に無効な値が指定されました: {e}") from e # ValueErrorはそのまま
         else: # ConfigError
              raise e # ConfigErrorはそのまま
    except Exception as e:
        logger.error(f"検出器パス取得中に予期せぬエラー: {e}", exc_info=True)
        raise FileError(f"検出器パスの取得中に予期せぬエラーが発生しました: {detector_name}, version={version}. 詳細: {e}") from e

# --- Data Paths --- #

def get_dataset_paths(config: Dict[str, Any], dataset_name: str) -> Tuple[Path, Path, str, str]:
    """
    指定されたデータセットの音声ディレクトリ、参照ディレクトリ、およびパターンを取得する。

    `config['datasets'][dataset_name]` の設定と `config['paths']['datasets_base']` を参照する。
    パスは絶対パスに解決される。

    Parameters
    ----------
    config : Dict[str, Any]
        アプリケーション設定辞書。
    dataset_name : str
        `config['datasets']` 内のキー。

    Returns
    -------
    Tuple[Path, Path, str, str]
        (audio_dir_path, ref_dir_path, audio_pattern, ref_pattern) のタプル。

    Raises
    ------
    ConfigError
        設定が見つからないか、不正な場合。
    FileError
        ディレクトリが存在しない場合。
    """
    datasets_config = config.get('datasets')
    if not datasets_config or dataset_name not in datasets_config:
        raise ConfigError(f"設定ファイルにデータセット '{dataset_name}' の定義が見つかりません。")

    ds_config = datasets_config[dataset_name]
    required_keys = ['audio_dir', 'label_dir', 'audio_pattern', 'label_pattern']
    if not all(key in ds_config for key in required_keys):
        missing = [key for key in required_keys if key not in ds_config]
        raise ConfigError(f"データセット '{dataset_name}' の設定に不足しているキーがあります: {missing}")

    # データセット固有のベースパスがあればそれを使う、なければ全体のベースパス
    dataset_base_override = ds_config.get('datasets_base')
    if dataset_base_override:
         datasets_base_str = dataset_base_override
         source = f"データセット '{dataset_name}' 固有の datasets_base"
    else:
         datasets_base_str = config.get('paths', {}).get('datasets_base', 'datasets')
         source = "全体の paths.datasets_base"

    logger.debug(f"{source} からデータセットベースパスを取得: {datasets_base_str}")

    base_dir = Path(datasets_base_str)
    project_root = get_project_root()
    if not base_dir.is_absolute():
        base_dir = (project_root / base_dir).resolve()

    if not base_dir.is_dir():
         # ここで ensure_dir しない方が良い (データセットディレクトリは存在している前提)
         raise ConfigError(f"データセットのベースディレクトリが見つかりません ({source}): {base_dir}")

    # 各ディレクトリパスを解決
    try:
        audio_dir_rel = Path(ds_config['audio_dir'])
        audio_dir = (base_dir / audio_dir_rel).resolve() if not audio_dir_rel.is_absolute() else audio_dir_rel.resolve()
        if not audio_dir.is_dir():
            raise FileError(f"Audio directory not found for dataset '{dataset_name}': {audio_dir}")

        label_dir_rel = Path(ds_config['label_dir']) # label_dir を使用
        label_dir = (base_dir / label_dir_rel).resolve() if not label_dir_rel.is_absolute() else label_dir_rel.resolve()
        if not label_dir.is_dir():
             raise FileError(f"Label directory not found for dataset '{dataset_name}': {label_dir}")

        # パターンを取得
        audio_pattern = ds_config['audio_pattern']
        label_pattern = ds_config['label_pattern'] # label_pattern を使用

        if not isinstance(audio_pattern, str) or not isinstance(label_pattern, str):
             raise ConfigError(f"データセット '{dataset_name}' の audio_pattern または label_pattern が文字列ではありません。")

        return audio_dir, label_dir, audio_pattern, label_pattern

    except FileError as e:
        # ディレクトリが存在しないエラーはそのまま送出
        raise e
    except Exception as e:
        logger.error(f"データセット '{dataset_name}' のパス解決中にエラー: {e}", exc_info=True)
        raise ConfigError(f"データセット '{dataset_name}' のパス設定の処理中にエラーが発生しました。設定を確認してください。詳細: {e}") from e

# --- Output Directory Construction --- #

def get_output_dir(base_dir: Path, *subdirs: str, unique_suffix: Optional[str] = None) -> Path:
    """
    指定されたベースディレクトリの下に、サブディレクトリと一意なサフィックスを持つ出力パスを構築する。
    注意: この関数はパスを構築するだけで、ディレクトリの作成は行わない。
         ディレクトリ作成は ensure_dir を呼び出し元で使用する。

    Parameters
    ----------
    base_dir : Path
        出力のベースとなるディレクトリ (例: get_output_base_dir() の戻り値)。絶対パスである必要あり。
    *subdirs : str
        ベースディレクトリの下に作成するサブディレクトリ名のシーケンス。
    unique_suffix : Optional[str], optional
        パスの末尾に追加する一意な文字列 (例: ジョブID, セッションID+タイムスタンプ)。
        Noneの場合、サフィックスは追加されない。

    Returns
    -------
    Path
        構築された出力ディレクトリの絶対パス。

    Raises
    ------
    ValueError
        ベースディレクトリが絶対パスでない、またはサブディレクトリ名やサフィックスに不正な文字が含まれている場合。
    TypeError
        base_dir が Path オブジェクトでない場合。
    """
    if not isinstance(base_dir, Path):
        raise TypeError("base_dir must be a Path object.")
    if not base_dir.is_absolute():
        raise ValueError("base_dir must be an absolute path.")

    current_path = base_dir

    # サブディレクトリを安全に追加
    for subdir in subdirs:
        if not is_safe_path_component(subdir):
            raise ValueError(f"サブディレクトリ名に不正な文字が含まれています: '{subdir}'")
        current_path /= subdir

    # 一意なサフィックスを追加
    if unique_suffix:
        if not is_safe_path_component(unique_suffix):
             raise ValueError(f"一意なサフィックスに不正な文字が含まれています: '{unique_suffix}'")
        current_path = current_path / unique_suffix

    # ensure_created=False なのでディレクトリ作成は行わない
    # if ensure_created:
    #     try:
    #         ensure_dir(current_path, check_writable=True) # 作成し、書き込み可能か確認
    #     except (FileError, TypeError) as e:
    #         # ensure_dir からのエラーをそのまま送出
    #         raise FileError(f"出力ディレクトリの作成またはアクセスに失敗しました: {current_path}. 詳細: {e}") from e

    return current_path.resolve()

# --- Setup & Utilities --- #

def setup_python_path() -> None:
    """
    PYTHONPATHを設定してプロジェクトモジュールをインポートできるようにする。
    プロジェクトルートと src ディレクトリを sys.path の先頭に追加する。
    """
    try:
        project_root = str(get_project_root())
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            logger.debug(f"PYTHONPATHにプロジェクトルートを追加: {project_root}")

        # srcディレクトリがある場合、それもパスに追加
        src_dir = os.path.join(project_root, 'src')
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)
            logger.debug(f"PYTHONPATHにsrcディレクトリを追加: {src_dir}")

        # 環境変数の読み込みも行う（get_project_root内で実行されるが念のため）
        load_environment_variables()

    except ConfigError:
         logger.warning("プロジェクトルートが見つからないため、PYTHONPATHの設定が不完全かもしれません。")
    except Exception as e:
         logger.error(f"PYTHONPATH設定中に予期せぬエラー: {e}", exc_info=True)

# --- Deprecated Path Functions (Remove after refactoring) --- #

# def get_evaluation_results_dir() -> Path:
#     """(Deprecated) Get the directory for evaluation results."""
#     # ...

# def get_grid_search_results_dir() -> Path:
#     """(Deprecated) Get the directory for grid search results."""
#     # ...

# def get_session_dir(session_id: str) -> Path:
#     """(Deprecated) Get the directory for a specific session."""
#     # ...

# def get_state_dir() -> Path:
#     """(Deprecated) Get the directory for state files."""
#     # ...

# --- Module Initialization --- #
setup_python_path()
# load_environment_variables() # Optionally load here, or let consuming code call it.

# --- 安全なパスコンポーネント検証関数 (再掲) --- #
def is_safe_path_component(component: str) -> bool:
    r"""ファイル名やディレクトリ名として安全なコンポーネントかチェックする。

    パス区切り文字 (`/`, `\`)、親ディレクトリ参照 (`..`)、カレントディレクトリ (`.`)、
    ヌルバイト、その他の制御文字、前後の空白、特定の禁止文字 (`<>:\"|?*`)
    を含まないことを確認する。
    空文字列も許可しない。

    Parameters
    ----------
    component : str
        チェックするパスコンポーネント (ファイル名やディレクトリ名)

    Returns
    -------
    bool
        安全な場合は True、そうでない場合は False
    """
    if not component: # 空文字列は不可
        return False

    # カレントディレクトリ参照 (".") も不可とする
    if component == ".":
        return False

    # パス区切り文字、親ディレクトリ参照、ヌルバイトをチェック
    # ".." は完全一致でチェックする
    if '/' in component or '\\' in component or component == '..' or '\0' in component:
        logger.debug(f"Unsafe component '{component}': Contains path separators, '..', or null byte.")
        return False

    # 制御文字 (ASCII 0-31) をチェック
    if re.search(r'[\x00-\x1F]', component):
        logger.debug(f"Unsafe component '{component}': Contains control characters.")
        return False

    # 前後の空白を不可とする
    if component.strip() != component:
        logger.debug(f"Unsafe component '{component}': Contains leading/trailing whitespace.")
        return False

    # Windows で許されない文字 (<>:\"|?*) のチェック
    if re.search(r'[<>:"|?*]', component):
        logger.debug(f"Unsafe component '{component}': Contains forbidden characters.")
        return False

    # オプション: Windows 予約名チェック (必要であれば有効化)
    # if os.name == 'nt' and component.upper() in ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', ...]:
    #     logger.debug(f"Unsafe component '{component}': Is a reserved name on Windows.")
    #     return False

    # ここまで到達すれば安全
    return True

# --- パス検証関数 (再掲) --- #
def validate_path_within_allowed_dirs(
    path_to_check: Union[str, Path],
    allowed_base_dirs: List[Union[str, Path]],
    check_existence: bool = True,
    check_is_file: Optional[bool] = None,
    allow_absolute: bool = False
) -> Path:
    """指定されたパスが、許可されたベースディレクトリのいずれかの内部にあり、
    オプションで存在確認やファイル/ディレクトリ種別確認を行う。

    Parameters & Returns & Raises: 上記の定義を参照
    """
    if not allowed_base_dirs:
        msg = "許可されたベースディレクトリのリストが空です。パス検証を実行できません。"
        logger.error(msg)
        raise ConfigError(msg)

    try:
        resolved_allowed_bases = [Path(d).resolve(strict=False) for d in allowed_base_dirs] # strict=Falseで存在しなくても解決試行
    except Exception as e:
        msg = f"許可されたベースディレクトリの解決中にエラー: {e}"
        logger.error(msg, exc_info=True)
        raise ConfigError(msg) from e

    target_path = Path(path_to_check)

    # 絶対パスの扱い
    if target_path.is_absolute():
        if not allow_absolute:
            msg = f"絶対パスは許可されていません: {target_path}"
            logger.warning(msg)
            raise ValueError(msg)
        # resolve() はシンボリックリンクを解決する。ここでは不要かもしれない。
        # パスが存在しない可能性も考慮して strict=False
        resolved_target = target_path.resolve(strict=False)
    else:
        # 相対パスはプロジェクトルート基準で解決
        resolved_target = (get_project_root() / target_path).resolve(strict=False)

    # パスが許可されたディレクトリのいずれかの下にあるか検証
    is_allowed = False
    for allowed_base in resolved_allowed_bases:
        try:
            # Path.is_relative_to は Python 3.9+ が必要
            # resolved_target が allowed_base と同じ場合も True になる
            if hasattr(resolved_target, 'is_relative_to'):
                 if resolved_target == allowed_base or resolved_target.is_relative_to(allowed_base):
                     is_allowed = True
                     break
            # フォールバック (Python 3.9未満) - 注意: シンボリックリンクで問題の可能性
            elif str(resolved_target).startswith(str(allowed_base)):
                 is_allowed = True
                 break
        except ValueError:
            # is_relative_to がドライブ違いなどで ValueError を出す場合
            continue

    if not is_allowed:
        allowed_dirs_str = ", ".join(map(str, resolved_allowed_bases))
        msg = f"指定されたパス ({resolved_target}) は許可されたディレクトリ ({allowed_dirs_str}) 内にありません。"
        logger.warning(msg)
        raise FileError(msg)

    # 存在チェック
    if check_existence:
        if not resolved_target.exists():
            msg = f"指定されたパスが存在しません: {resolved_target}"
            logger.warning(msg)
            raise FileError(msg)
        # 存在する場合、シンボリックリンクでないことを確認する (オプション)
        # if resolved_target.is_symlink():
        #     msg = f"指定されたパスはシンボリックリンクです (許可されていません): {resolved_target}"
        #     logger.warning(msg)
        #     raise FileError(msg)

    # ファイル/ディレクトリ種別チェック (存在する場合のみ)
    if check_existence and check_is_file is not None:
        if check_is_file and not resolved_target.is_file():
            msg = f"指定されたパスはファイルではありません: {resolved_target}"
            logger.warning(msg)
            raise FileError(msg)
        elif not check_is_file and not resolved_target.is_dir():
            msg = f"指定されたパスはディレクトリではありません: {resolved_target}"
            logger.warning(msg)
            raise FileError(msg)

    return resolved_target # 検証済みの絶対パスを返す