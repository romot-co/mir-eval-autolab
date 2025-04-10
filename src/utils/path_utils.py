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

def load_environment_variables(dotenv_path: Optional[Union[str, Path]] = None) -> bool:
    """
    .envファイルから環境変数を読み込む。

    指定されたパス、またはプロジェクトルートの .env を読み込む。

    Parameters
    ----------
    dotenv_path : Optional[Union[str, Path]], optional
        読み込む .env ファイルのパス。Noneの場合、プロジェクトルートの .env を探す。

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
    else:
        try:
            # プロジェクトルートを取得 (まだキャッシュされていない可能性あり)
            root = get_project_root(use_cache=False) # キャッシュを使わずに再検索
            default_env_path = root / '.env'
            if default_env_path.exists():
                load_path = default_env_path
            else:
                 # .env.example があれば情報ログを出す
                 example_path = root / '.env.example'
                 if example_path.exists():
                     logger.info(f".env ファイルが見つかりません ({default_env_path})。.env.example をコピーして設定できます。")
        except ConfigError:
             logger.warning("プロジェクトルート特定前に .env を探しています。.env の場所を明示的に指定するか、先にプロジェクトルートを設定してください。")
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
        logger.debug(f".env ファイルが見つからなかったか、ファイルではありません: {load_path}")
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

def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    指定されたパスのディレクトリが存在することを確認し、なければ作成する。
    パスオブジェクトを返す。

    Parameters
    ----------
    dir_path : Union[str, Path]
        確認・作成するディレクトリのパス。

    Returns
    -------
    Path
        存在する(または作成された)ディレクトリのパスオブジェクト。

    Raises
    ------
    FileError
        パスがディレクトリでない、または作成に失敗した場合。
    PermissionError
        ディレクトリ作成に必要な権限がない場合。
    """
    path_obj = Path(dir_path).resolve()
    try:
        if path_obj.exists():
            if not path_obj.is_dir():
                msg = f"指定されたパスは存在しますが、ディレクトリではありません: {path_obj}"
                logger.error(msg)
                raise FileError(msg)
        else:
            # parents=True で中間ディレクトリも作成
            # exist_ok=True で既に存在する場合にエラーにしない
            path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(f"ディレクトリを作成しました: {path_obj}")
        return path_obj
    except PermissionError as pe:
        msg = f"ディレクトリ作成/アクセスに必要な権限がありません: {path_obj}"
        logger.error(msg)
        raise PermissionError(msg) from pe
    except OSError as ose:
        msg = f"ディレクトリ作成/アクセス中にOSエラーが発生しました: {path_obj} - {ose}"
        logger.error(msg)
        raise FileError(msg) from ose


def get_workspace_dir(config: Optional[Dict[str, Any]] = None) -> Path:
    """
    ワークスペースディレクトリのパスを取得・検証する。
    優先順位: 環境変数 MIREX_WORKSPACE > config.yaml > デフォルトパス。
    パスが存在しないか書き込み不可の場合はエラーを発生させる。

    Parameters
    ----------
    config : Optional[Dict[str, Any]], optional
        読み込み済みの設定辞書 (config.yaml の値を含む), by default None。
        Noneの場合、デフォルトパスのみが考慮される。

    Returns
    -------
    Path
        検証済みのワークスペースディレクトリのパス

    Raises
    ------
    FileError
        指定されたパスが無効、存在しない、または書き込みできない場合
    PermissionError
        書き込み権限がない場合
    ConfigError
        パスを決定できなかった場合
    """
    workspace_path_str: Optional[str] = None
    source: str = ""

    # 1. 環境変数 MIREX_WORKSPACE を確認
    # 環境変数は .env 読み込み後であるべき -> get_project_root() を先に呼んでおく
    try:
         get_project_root() # これにより内部で .env が読み込まれるはず
    except ConfigError:
         pass # ルートが見つからなくても、環境変数だけはチェックできる
    env_workspace = os.environ.get('MIREX_WORKSPACE')
    if env_workspace:
        workspace_path_str = env_workspace
        source = "環境変数 MIREX_WORKSPACE"

    # 2. config.yaml の設定を確認 (config が提供された場合)
    if not workspace_path_str and config and isinstance(config.get('paths'), dict):
        config_workspace = config['paths'].get('workspace')
        if config_workspace:
            workspace_path_str = config_workspace
            source = "config.yaml の paths.workspace"

    # 3. デフォルトパスを使用
    if not workspace_path_str:
        try:
            project_root = get_project_root() # ここで ConfigError が発生する可能性
            default_workspace_path = project_root / 'mcp_workspace'
            workspace_path_str = str(default_workspace_path)
            source = "デフォルトパス (プロジェクトルート/mcp_workspace)"
        except ConfigError:
             # プロジェクトルートが見つからない場合、ワークスペースパスを決定できない
             msg = "ワークスペースパスを決定できませんでした (プロジェクトルート不明、環境変数/設定もなし)。"
             logger.critical(msg)
             raise ConfigError(msg)


    # パスの検証
    if not workspace_path_str:
        # 通常ここには到達しないはずだが念のため
        raise ConfigError("ワークスペースパスを決定できませんでした。")

    workspace_dir = Path(workspace_path_str).resolve()
    logger.info(f"使用するワークスペースパス ({source}): {workspace_dir}")

    try:
        # ディレクトリを確保 (ensure_dir が存在確認・作成・権限チェックを行う)
        ensure_dir(workspace_dir)

        # 書き込み権限テスト (ensure_dir内で行うべきかもしれないが、念のため)
        test_file_path = workspace_dir / f".write_test_{os.urandom(4).hex()}.tmp"
        try:
            test_file_path.touch(exist_ok=True) # exist_ok=True を追加
            # 書き込んだ後すぐに削除
            test_file_path.unlink()
            logger.debug(f"ワークスペースディレクトリへの書き込み権限を確認しました: {workspace_dir}")
        except (PermissionError, OSError) as e:
            # touch や unlink でエラーが発生した場合
            msg = f"ワークスペースディレクトリへの書き込み/削除テストに失敗しました: {workspace_dir} - {e}"
            logger.critical(msg)
            # PermissionError はそのまま、他のOSErrorはFileErrorにラップ
            if isinstance(e, PermissionError):
                raise PermissionError(msg) from e
            else:
                raise FileError(msg) from e
        finally:
            # テストファイルが残った場合に備えて再度削除試行
            if test_file_path.exists():
                try: test_file_path.unlink()
                except OSError: pass

        return workspace_dir

    except (FileError, PermissionError) as e:
        # ensure_dir や書き込みテストで発生したエラーをそのまま上に投げる
        raise e
    except Exception as e:
        # その他の予期せぬエラー
        msg = f"ワークスペースディレクトリの準備中に予期せぬエラー ({source}): {workspace_dir} - {e}"
        logger.critical(msg, exc_info=True)
        raise FileError(msg) from e

# --- 各種ディレクトリ取得関数 (get_workspace_dir を使用するように修正) ---

def get_evaluation_results_dir() -> Path:
    """評価結果ディレクトリのパスを取得する (ワークスペース基準)。"""
    env_output_dir = os.environ.get('MIREX_OUTPUT_DIR')
    if env_output_dir:
        results_dir = Path(env_output_dir).resolve()
    else:
        results_dir = get_workspace_dir() / 'evaluation_results' # get_workspace_dir() を呼ぶ
    return ensure_dir(results_dir)

def get_grid_search_results_dir() -> Path:
    """グリッドサーチ結果ディレクトリのパスを取得する (ワークスペース基準)。"""
    env_grid_dir = os.environ.get('MIREX_GRID_SEARCH_DIR')
    if env_grid_dir:
        results_dir = Path(env_grid_dir).resolve()
    else:
        results_dir = get_workspace_dir() / 'grid_search_results' # get_workspace_dir() を呼ぶ
    return ensure_dir(results_dir)

def get_improved_versions_dir() -> Path:
    """改善バージョンディレクトリのパスを取得する (ワークスペース基準)。"""
    env_versions_dir = os.environ.get('MIREX_VERSIONS_DIR')
    if env_versions_dir:
        versions_dir = Path(env_versions_dir).resolve()
    else:
        versions_dir = get_workspace_dir() / 'improved_versions' # get_workspace_dir() を呼ぶ
    return ensure_dir(versions_dir)

def get_session_dir(session_id: str) -> Path:
    """セッションディレクトリのパスを取得する (ワークスペース基準)。"""
    env_session_base = os.environ.get('MIREX_SESSION_DIR')
    if env_session_base:
        session_base_dir = Path(env_session_base).resolve()
    else:
        session_base_dir = get_workspace_dir() / 'sessions' # get_workspace_dir() を呼ぶ
    # セッションIDが安全な文字列か検証
    if not is_safe_path_component(session_id):
        raise ValueError(f"セッションIDに使用できない文字が含まれています: {session_id}")
    session_dir = session_base_dir / session_id
    return ensure_dir(session_dir)

def get_state_dir() -> Path:
    """状態保存ディレクトリのパスを取得する (ワークスペース基準)。"""
    env_state_dir = os.environ.get('MCP_STATE_DIR')
    if env_state_dir:
        state_dir = Path(env_state_dir).resolve()
    else:
        state_dir = get_workspace_dir() / 'improvement_states' # get_workspace_dir() を呼ぶ
    return ensure_dir(state_dir)

def get_db_dir() -> Path:
    """データベースファイル用ディレクトリのパスを取得する (ワークスペース基準)。"""
    # 環境変数等で変更できるようにしても良い
    db_dir = get_workspace_dir() / 'db' # get_workspace_dir() を呼ぶ
    return ensure_dir(db_dir)

# --- データセット関連のパス取得 (プロジェクトルート基準が多い) ---

def get_datasets_dir() -> Path:
    """データセットのベースディレクトリパスを取得する (プロジェクトルート基準)。"""
    # 環境変数 MIREX_DATASETS_DIR を優先
    env_datasets_dir = os.environ.get('MIREX_DATASETS_DIR')
    if env_datasets_dir:
        datasets_dir = Path(env_datasets_dir).resolve()
    else:
        datasets_dir = get_project_root() / 'datasets'
    # ensure_dir(datasets_dir) # 存在しなくてもパスは返す
    if not datasets_dir.is_dir():
         logger.warning(f"データセットベースディレクトリが存在しません: {datasets_dir}")
    return datasets_dir

def get_synthesized_datasets_dir() -> Path:
    """合成データセットのベースディレクトリパスを取得する。"""
    synth_dir = get_datasets_dir() / 'synthesized'
    # ensure_dir(synth_dir)
    if not synth_dir.is_dir():
         logger.warning(f"合成データセットディレクトリが存在しません: {synth_dir}")
    return synth_dir

def get_audio_dir() -> Path:
    """デフォルトの合成音声ファイルディレクトリのパスを取得する。"""
    env_audio_dir = os.environ.get('MIREX_AUDIO_DIR')
    if env_audio_dir:
        audio_dir = Path(env_audio_dir).resolve()
    else:
        audio_dir = get_synthesized_datasets_dir() / 'audio'
    # ensure_dir(audio_dir)
    if not audio_dir.is_dir():
        logger.warning(f"デフォルト音声ディレクトリが存在しません: {audio_dir}")
    return audio_dir

def get_label_dir() -> Path:
    """デフォルトの合成ラベルファイルディレクトリのパスを取得する。"""
    env_label_dir = os.environ.get('MIREX_LABEL_DIR')
    if env_label_dir:
        label_dir = Path(env_label_dir).resolve()
    else:
        label_dir = get_synthesized_datasets_dir() / 'labels'
    # ensure_dir(label_dir)
    if not label_dir.is_dir():
        logger.warning(f"デフォルトラベルディレクトリが存在しません: {label_dir}")
    return label_dir

# --- 検出器コード関連のパス取得 ---

def get_detectors_src_dir() -> Path:
    """`src/detectors` ディレクトリのパスを取得する。"""
    return get_project_root() / 'src' / 'detectors'

def get_detector_path(detector_name: str, version: Optional[str] = None) -> Path:
    """
    指定された検出器名とバージョンに対応するPythonファイルのパスを取得する。

    検索順序:
    1. バージョン指定あり: `improved_versions` ディレクトリ内 (`<name>_<version>.py`)
    2. バージョン指定なし: `improved_versions` ディレクトリ内の最新版 (タイムスタンプベース)
    3. 上記で見つからない場合: `src/detectors` ディレクトリ内 (`<name>.py`)

    Parameters
    ----------
    detector_name : str
        検出器名 (ベース名)。
    version : Optional[str], optional
        取得したいバージョンタグ。Noneの場合は最新の改善版またはベースコードを探す, by default None

    Returns
    -------
    Path
        検出器ファイルの絶対パス。

    Raises
    ------
    FileNotFoundError
        指定された検出器ファイルが見つからない場合。
    ValueError
        検出器名やバージョン名に不正な文字が含まれる場合。
    """
    if not is_safe_path_component(detector_name):
        raise ValueError(f"検出器名に使用できない文字が含まれています: {detector_name}")
    if version and not is_safe_path_component(version):
         raise ValueError(f"バージョン名に使用できない文字が含まれています: {version}")

    improved_dir = get_improved_versions_dir() # ワークスペース基準
    src_detectors_dir = get_detectors_src_dir() # プロジェクトルート基準

    # 1. バージョン指定がある場合
    if version:
        versioned_file = improved_dir / f"{detector_name}_{version}.py"
        if versioned_file.is_file():
            logger.debug(f"指定バージョン '{version}' を improved_versions で発見: {versioned_file}")
            return versioned_file.resolve()
        else:
            # 指定バージョンが見つからない場合はエラー
            msg = f"指定されたバージョンの検出器が見つかりません: name='{detector_name}', version='{version}' in {improved_dir}"
            logger.error(msg)
            raise FileNotFoundError(msg)

    # 2. バージョン指定なし -> 最新の改善版を探す
    latest_improved_file: Optional[Path] = None
    latest_mtime: float = -1.0

    if improved_dir.is_dir():
        # detector_name で始まるファイルを探す
        pattern = f"{detector_name}_*.py"
        for file_path in improved_dir.glob(pattern):
            if file_path.is_file():
                try:
                    mtime = file_path.stat().st_mtime
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_improved_file = file_path
                except OSError as e:
                    logger.warning(f"改善版ファイルの最終更新日時取得エラー: {file_path} ({e})")
                    continue # 次のファイルへ

    if latest_improved_file:
        logger.debug(f"最新の改善バージョンを発見 (更新日時ベース): {latest_improved_file}")
        return latest_improved_file.resolve()

    # 3. 改善版が見つからない -> src/detectors を探す
    base_file = src_detectors_dir / f"{detector_name}.py"
    if base_file.is_file():
        logger.debug(f"改善バージョンが見つからず。ベースコードを使用: {base_file}")
        return base_file.resolve()

    # 4. 小文字ファイル名も試す (フォールバック)
    base_file_lower = src_detectors_dir / f"{detector_name.lower()}.py"
    if base_file_lower.is_file():
        logger.warning(f"ベースコード '{detector_name}.py' が見つからず、小文字版を使用: {base_file_lower}")
        return base_file_lower.resolve()

    # 最終的に見つからない場合
    msg = f"検出器ファイルが見つかりませんでした: name='{detector_name}' (src: {src_detectors_dir}, improved: {improved_dir})"
    logger.error(msg)
    raise FileNotFoundError(msg)


def get_dataset_paths(config: Dict[str, Any], dataset_name: str) -> Tuple[Path, Path, str]:
    """
    設定とデータセット名から音声ディレクトリ、ラベルディレクトリ、ラベルパターンを取得する。

    Parameters
    ----------
    config : Dict[str, Any]
        プロジェクト設定辞書。
    dataset_name : str
        データセット名 (`config['datasets']` のキー)。

    Returns
    -------
    Tuple[Path, Path, str]
        (音声ディレクトリパス, ラベルディレクトリパス, ラベルファイルパターン)

    Raises
    ------
    ConfigError
        データセット設定が見つからない、またはパスが無効な場合。
    FileNotFoundError
        指定されたディレクトリが存在しない場合。
    """
    if 'datasets' not in config or dataset_name not in config['datasets']:
        msg = f"設定ファイルにデータセット '{dataset_name}' の定義が見つかりません。"
        logger.error(msg)
        raise ConfigError(msg)

    dataset_config = config['datasets'][dataset_name]
    audio_dir_str = dataset_config.get('audio_dir')
    label_dir_str = dataset_config.get('label_dir')
    label_pattern = dataset_config.get('label_pattern', '*.csv') # デフォルト *.csv

    if not audio_dir_str or not label_dir_str:
        msg = f"データセット '{dataset_name}' の設定に 'audio_dir' または 'label_dir' が不足しています。"
        logger.error(msg)
        raise ConfigError(msg)

    # パスを解決 (プロジェクトルート基準の可能性あり)
    audio_dir = get_absolute_path(audio_dir_str)
    label_dir = get_absolute_path(label_dir_str)

    # ディレクトリの存在確認
    if not audio_dir.is_dir():
        msg = f"データセット '{dataset_name}' の音声ディレクトリが見つからないか、ディレクトリではありません: {audio_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if not label_dir.is_dir():
        msg = f"データセット '{dataset_name}' のラベルディレクトリが見つからないか、ディレクトリではありません: {label_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"データセット '{dataset_name}' のパスを取得: Audio='{audio_dir}', Labels='{label_dir}', Pattern='{label_pattern}'")
    return audio_dir, label_dir, label_pattern

def get_output_dir(base_dir: Path, unique_suffix: str) -> Path:
    """
    一意のサフィックスを持つ出力ディレクトリパスを生成し、存在を確認/作成する。

    Parameters
    ----------
    base_dir : Path
        出力のベースとなるディレクトリ。
    unique_suffix : str
        サブディレクトリ名となる一意のサフィックス。安全な文字列か検証される。

    Returns
    -------
    Path
        生成された出力ディレクトリのパス。

    Raises
    ------
    ValueError
        unique_suffix に不正な文字が含まれる場合。
    FileError, PermissionError
        ディレクトリ作成に失敗した場合。
    """
    if not is_safe_path_component(unique_suffix):
        raise ValueError(f"出力ディレクトリのサフィックスに使用できない文字が含まれています: {unique_suffix}")
    output_dir = base_dir / unique_suffix
    return ensure_dir(output_dir) # ensure_dir がエラー処理を行う

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


# --- モジュールのインポート時に実行 ---
# 1. PYTHONPATH設定
setup_python_path()
# 2. 環境変数読み込み（setup_python_path内で呼ばれるが、明示的に呼んでも良い）
# load_environment_variables()

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