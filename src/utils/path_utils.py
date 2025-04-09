#!/usr/bin/env python3
"""
パス操作ユーティリティモジュール

このモジュールは、プロジェクト内でのパス操作に関するユーティリティ関数を提供します。
絶対パスの解決、プロジェクトルートの検出、ディレクトリの作成などの機能を含みます。
"""

import os
import sys
import glob
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Union, Tuple

# .envファイルのサポートを追加（利用可能な場合）
try:
    import dotenv
    has_dotenv = True
except ImportError:
    has_dotenv = False

logger = logging.getLogger(__name__)

# プロジェクトルートを特定するマーカーファイル
ROOT_MARKERS = [
    'mcp_server.py',
    'setup_claude_integration.py',
    'run_evaluate.py',
    'run_grid_search.py',
    'auto_improver.py',
    'src/detectors',
    'src/evaluation',
    'MCP_README.md'
]

# キャッシュしたプロジェクトルートパス
_project_root = None

def load_environment_variables():
    """
    .envファイルがある場合は環境変数を読み込む
    """
    if has_dotenv:
        # プロジェクトルートにある.envファイルを探す
        root_path = get_project_root()
        env_path = root_path / '.env'
        
        if env_path.exists():
            dotenv.load_dotenv(env_path)
            logger.debug(f"{env_path} から環境変数を読み込みました")
        else:
            # .env.exampleファイルが存在するかを確認
            example_path = root_path / '.env.example'
            if example_path.exists():
                logger.info(f".envファイルが見つかりません。.env.exampleをコピーして設定できます: cp {example_path} {env_path}")

def find_project_root(start_dir=None, max_levels=5) -> Path:
    """
    プロジェクトルートディレクトリを特定する
    
    Parameters
    ----------
    start_dir : str or Path, optional
        検索を開始するディレクトリ（デフォルト: 現在のスクリプト）
    max_levels : int, optional
        親ディレクトリを遡る最大レベル数（デフォルト: 5）
        
    Returns
    -------
    Path
        プロジェクトルートディレクトリのパス、見つからない場合は現在の作業ディレクトリ
    """
    global _project_root
    
    # キャッシュがある場合はそれを返す
    if _project_root is not None:
        return _project_root
    
    # 開始ディレクトリの設定
    if start_dir is None:
        # 呼び出し元のスクリプトのディレクトリ
        frame = sys._getframe(1)
        filename = frame.f_code.co_filename
        start_dir = Path(filename).resolve().parent
    else:
        start_dir = Path(start_dir).resolve()
    
    # 現在のディレクトリから上位ディレクトリを順に探索
    current_dir = start_dir
    for _ in range(max_levels):
        # マーカーファイル/ディレクトリのいずれかが存在するか確認
        for marker in ROOT_MARKERS:
            marker_path = current_dir / marker
            if marker_path.exists():
                _project_root = current_dir
                logger.debug(f"プロジェクトルートを検出: {current_dir}")
                return current_dir
        
        # 親ディレクトリに移動
        parent_dir = current_dir.parent
        if parent_dir == current_dir:  # ルートディレクトリに到達
            break
        current_dir = parent_dir
    
    # 見つからなかった場合は現在の作業ディレクトリを返す
    logger.warning(f"プロジェクトルートが見つかりません。カレントディレクトリを使用: {os.getcwd()}")
    return Path(os.getcwd())

def get_project_root() -> Path:
    """
    プロジェクトルートディレクトリを取得する（キャッシュ使用）
    
    Returns
    -------
    Path
        プロジェクトルートディレクトリのパス
    """
    global _project_root
    
    if _project_root is None:
        # スクリプトの位置から検索
        script_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = find_project_root(script_dir)
    
    return _project_root

def get_absolute_path(relative_path: Union[str, Path]) -> Path:
    """
    プロジェクトルートからの相対パスを絶対パスに変換する
    
    Parameters
    ----------
    relative_path : str or Path
        プロジェクトルートからの相対パス
        
    Returns
    -------
    Path
        絶対パス
    """
    if str(relative_path).startswith('/') or (isinstance(relative_path, str) and os.path.isabs(relative_path)):
        # 既に絶対パスの場合はそのまま返す
        return Path(relative_path)
    
    return get_project_root() / relative_path

def ensure_dir(dir_path: Path):
    """指定されたパスのディレクトリが存在することを確認し、なければ作成する"""
    dir_path.mkdir(parents=True, exist_ok=True)

def get_workspace_dir() -> Path:
    """
    ワークスペースディレクトリのパスを取得する
    
    環境変数 MIREX_WORKSPACE が設定されている場合はその値を使用し、
    設定されていない場合はデフォルトパスを使用します。
    
    Returns
    -------
    Path
        ワークスペースディレクトリのパス
    """
    # 環境変数 MIREX_WORKSPACE を優先
    env_workspace = os.environ.get('MIREX_WORKSPACE')
    if env_workspace:
        workspace_dir = Path(env_workspace).resolve()
        logger.debug(f"環境変数 MIREX_WORKSPACE からワークスペースを取得: {workspace_dir}")
        ensure_dir(workspace_dir)
        return workspace_dir
    
    # .envファイル読み込み後に再度確認
    load_environment_variables()
    env_workspace = os.environ.get('MIREX_WORKSPACE')
    if env_workspace:
        workspace_dir = Path(env_workspace).resolve()
        logger.debug(f"環境変数 MIREX_WORKSPACE (.env読み込み後) からワークスペースを取得: {workspace_dir}")
        ensure_dir(workspace_dir)
        return workspace_dir

    # デフォルトパスの設定（プロジェクトルート > ホームディレクトリ > 一時ディレクトリ）
    project_root = get_project_root()
    default_paths = [
        project_root / 'mcp_workspace',
        Path.home() / '.mirex_workspace'
    ]

    for path in default_paths:
        try:
            # 存在確認と書き込みテスト
            if path.exists() or path.parent.exists():
                ensure_dir(path)
                # 書き込みテスト
                test_file = path / f".write_test_{os.urandom(4).hex()}"
                test_file.touch()
                test_file.unlink()
                logger.debug(f"デフォルトワークスペースとして使用: {path}")
                return path
        except (PermissionError, OSError) as e:
            logger.warning(f"デフォルトワークスペース候補への書き込み不可: {path} ({e})")
            continue

    # 最終手段: 一時ディレクトリ
    temp_workspace = Path(tempfile.gettempdir()) / f"mirex_workspace_{os.urandom(4).hex()}"
    try:
        ensure_dir(temp_workspace)
        # 書き込みテスト
        test_file = temp_workspace / f".write_test_{os.urandom(4).hex()}"
        test_file.touch()
        test_file.unlink()
        logger.warning(f"デフォルトワークスペースが見つからない/書き込めないため、一時ディレクトリを使用: {temp_workspace}")
        return temp_workspace
    except (PermissionError, OSError) as e:
         logger.critical(f"一時ワークスペースディレクトリへの書き込みも失敗: {temp_workspace} ({e})")
         # ここでエラーを発生させるか、より安全なフォールバックを検討
         # 例: プロジェクトルート直下にフォールバック
         fallback_workspace = project_root / '_fallback_workspace'
         try:
             ensure_dir(fallback_workspace)
             logger.warning(f"最終フォールバックとしてプロジェクトルート直下を使用: {fallback_workspace}")
             return fallback_workspace
         except Exception as final_e:
             logger.critical(f"最終フォールバックワークスペースの作成も失敗: {final_e}")
             raise RuntimeError("書き込み可能なワークスペースディレクトリを確保できませんでした")

def get_evaluation_results_dir() -> Path:
    """
    評価結果ディレクトリのパスを取得する
    
    環境変数 MIREX_OUTPUT_DIR が設定されている場合はその値を使用し、
    設定されていない場合はデフォルトパス (ワークスペース内) を使用します。
    
    Returns
    -------
    Path
        評価結果ディレクトリのパス
    """
    env_output_dir = os.environ.get('MIREX_OUTPUT_DIR')
    if env_output_dir:
        results_dir = Path(env_output_dir).resolve()
    else:
        # デフォルトはワークスペース内の evaluation_results
        results_dir = get_workspace_dir() / 'evaluation_results'
    
    ensure_dir(results_dir)
    return results_dir

def get_grid_search_results_dir() -> Path:
    """
    グリッドサーチ結果ディレクトリのパスを取得する
    
    Returns
    -------
    Path
        グリッドサーチ結果ディレクトリのパス
    """
    # 環境変数 MIREX_GRID_SEARCH_DIR を優先
    env_grid_dir = os.environ.get('MIREX_GRID_SEARCH_DIR')
    if env_grid_dir:
        results_dir = Path(env_grid_dir).resolve()
    else:
        # デフォルトはワークスペース内の grid_search_results
        results_dir = get_workspace_dir() / 'grid_search_results'
    ensure_dir(results_dir)
    return results_dir

def get_improved_versions_dir() -> Path:
    """
    改善バージョンディレクトリのパスを取得する
    
    Returns
    -------
    Path
        改善バージョンディレクトリのパス
    """
    # 環境変数 MIREX_VERSIONS_DIR を優先
    env_versions_dir = os.environ.get('MIREX_VERSIONS_DIR')
    if env_versions_dir:
        versions_dir = Path(env_versions_dir).resolve()
    else:
        # デフォルトはワークスペース内の improved_versions
        versions_dir = get_workspace_dir() / 'improved_versions'
    ensure_dir(versions_dir)
    return versions_dir

def get_session_dir(session_id: str) -> Path:
    """
    セッションディレクトリのパスを取得する
    
    Parameters
    ----------
    session_id : str
        セッションID
        
    Returns
    -------
    Path
        セッションディレクトリのパス
    """
    # 環境変数 MIREX_SESSION_DIR を優先 (パターンとして)
    env_session_base = os.environ.get('MIREX_SESSION_DIR')
    if env_session_base:
        session_base_dir = Path(env_session_base).resolve()
    else:
        # デフォルトはワークスペース内の sessions
        session_base_dir = get_workspace_dir() / 'sessions'
    session_dir = session_base_dir / session_id
    ensure_dir(session_dir)
    return session_dir

def get_audio_dir() -> Path:
    """
    音声ファイルディレクトリのパスを取得する
    
    環境変数 MIREX_AUDIO_DIR が設定されている場合はその値を使用し、
    設定されていない場合はデフォルトパスを使用します。
    
    Returns
    -------
    Path
        音声ファイルディレクトリのパス
    """
    env_audio_dir = os.environ.get('MIREX_AUDIO_DIR')
    if env_audio_dir:
        audio_dir = Path(env_audio_dir).resolve()
    else:
        # デフォルトはプロジェクトルート下の datasets/synthesized/audio
        audio_dir = get_project_root() / 'datasets' / 'synthesized' / 'audio'
    
    # 存在しない場合もパスは返すが、警告を出す
    if not audio_dir.exists():
        logger.warning(f"音声ディレクトリが存在しません: {audio_dir}")
    # 読み取り可能かどうかのチェックは呼び出し元で行うべき
    
    return audio_dir

def get_label_dir() -> Path:
    """
    ラベルファイルディレクトリのパスを取得する
    
    環境変数 MIREX_LABEL_DIR が設定されている場合はその値を使用し、
    設定されていない場合はデフォルトパスを使用します。
    
    Returns
    -------
    Path
        ラベルファイルディレクトリのパス
    """
    env_label_dir = os.environ.get('MIREX_LABEL_DIR')
    if env_label_dir:
        label_dir = Path(env_label_dir).resolve()
    else:
        # デフォルトはプロジェクトルート下の datasets/synthesized/labels
        label_dir = get_project_root() / 'datasets' / 'synthesized' / 'labels'

    # 存在しない場合もパスは返すが、警告を出す
    if not label_dir.exists():
        logger.warning(f"ラベルディレクトリが存在しません: {label_dir}")
    # 読み取り可能かどうかのチェックは呼び出し元で行うべき

    return label_dir

def get_state_dir() -> Path:
    """
    状態保存ディレクトリのパスを取得する

    環境変数 MCP_STATE_DIR が設定されている場合はその値を使用し、
    設定されていない場合はデフォルトパス (ワークスペース内) を使用します。

    Returns
    ------
    Path
        状態保存ディレクトリのパス
    """
    # 環境変数 MCP_STATE_DIR を優先
    env_state_dir = os.environ.get('MCP_STATE_DIR')
    if env_state_dir:
        state_dir = Path(env_state_dir).resolve()
    else:
        # デフォルトはワークスペース内の improvement_states
        state_dir = get_workspace_dir() / 'improvement_states'
    ensure_dir(state_dir)
    return state_dir

def get_detector_path(detector_name: str, use_improved: bool = False) -> Optional[Path]:
    """
    検出器ファイルのパスを取得する
    
    Parameters
    ----------
    detector_name : str
        検出器名（クラス名またはファイル名）
    use_improved : bool, optional
        改善バージョンを使用するかどうか、デフォルトはFalse
        
    Returns
    -------
    Path or None
        検出器ファイルのパス、見つからない場合はNone
    """
    # 改善バージョンを確認
    if use_improved:
        improved_dir = get_improved_versions_dir()
        # バージョン管理を単純化: 検出器名がバージョンを含む場合（例: MyDetector_v123）
        if '_' in detector_name:
            possible_path = improved_dir / f"{detector_name}.py"
            if possible_path.exists():
                logger.debug(f"改善バージョンを使用 (直接指定): {possible_path}")
                return possible_path
            
            # ベース名での検索も試みる (例: MyDetector_v123 -> MyDetector)
            base_name = detector_name.split('_')[0]
            detector_files = list(improved_dir.glob(f'{base_name}_v*.py'))
        else:
            detector_files = list(improved_dir.glob(f'{detector_name}_v*.py'))
            
        if detector_files:
            # 最新バージョン (ファイル名でソート、または更新日時)
            # version_pattern = re.compile(rf'{re.escape(detector_name)}_v(\d+)\.py')
            # versions = []
            # for f in detector_files:
            #     match = version_pattern.search(f.name)
            #     if match:
            #         versions.append((int(match.group(1)), f))
            # if versions:
            #     versions.sort(key=lambda x: x[0], reverse=True)
            #     latest_version_path = versions[0][1]
            #     logger.debug(f"改善バージョンを使用 (最新): {latest_version_path}")
            #     return latest_version_path
            
            # 更新日時でソート (より堅牢)
            detector_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            latest_version_path = detector_files[0]
            logger.debug(f"改善バージョンを使用 (最新更新日時): {latest_version_path}")
            return latest_version_path
    
    # 標準の検出器ディレクトリをチェック (src/detectors)
    detectors_base_dir = get_project_root() / 'src' / 'detectors'
    detector_path = detectors_base_dir / f'{detector_name}.py'
    if detector_path.exists():
        return detector_path
    
    # 小文字でファイル名を試す
    detector_path_lower = detectors_base_dir / f'{detector_name.lower()}.py'
    if detector_path_lower.exists():
        return detector_path_lower
    
    # ファイル名が一致しない場合、クラス名としてファイルを探索
    try:
        for file_path in detectors_base_dir.glob('*.py'):
            if file_path.is_file() and file_path.name != '__init__.py':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # ASTを使ってクラス定義を検出（より正確）
                        import ast
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef) and node.name == detector_name:
                                # BaseDetector を継承しているかなどのチェックも可能
                                logger.debug(f"検出器クラス '{detector_name}' をファイル内で発見: {file_path}")
                                return file_path
                except FileNotFoundError:
                    continue # ファイルが見つからない場合はスキップ
                except SyntaxError:
                     logger.warning(f"構文エラーのためスキップ: {file_path}")
                except Exception as e:
                    logger.warning(f"ファイル読み込み/解析中にエラーが発生しました: {file_path}, {str(e)}")
    except Exception as e:
        logger.error(f"検出器の検索中にエラーが発生しました: {e}")
    
    # 見つからない場合はNoneを返す
    logger.warning(f"検出器が見つかりませんでした: {detector_name}")
    return None

def get_dataset_paths(audio_dir: Path, ref_dir: Path, ref_pattern: str) -> Tuple[List[Path], List[Path]]:
    """指定されたディレクトリから音声ファイルと対応する参照ファイルのパスリストを取得する"""
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"音声ディレクトリが見つかりません: {audio_dir}")
    if not ref_dir.exists() or not ref_dir.is_dir():
        raise FileNotFoundError(f"参照ディレクトリが見つかりません: {ref_dir}")

    audio_paths = sorted(list(audio_dir.glob('*.wav')) + 
                         list(audio_dir.glob('*.mp3')) + 
                         list(audio_dir.glob('*.flac')))
    ref_paths_found = sorted(list(ref_dir.glob(ref_pattern)))

    if not audio_paths:
        raise FileNotFoundError(f"音声ファイルがディレクトリ内に見つかりません: {audio_dir}")
    if not ref_paths_found:
        raise FileNotFoundError(f"参照ファイルがパターン '{ref_pattern}' で見つかりません: {ref_dir}")

    # ファイル名のベースでマッチング
    audio_basenames = {p.stem: p for p in audio_paths}
    ref_basenames = {p.stem: p for p in ref_paths_found}

    matched_audio = []
    matched_ref = []
    missing_refs = []
    missing_audios = []

    for basename, audio_path in audio_basenames.items():
        if basename in ref_basenames:
            matched_audio.append(audio_path)
            matched_ref.append(ref_basenames[basename])
        else:
            missing_refs.append(basename)

    for basename in ref_basenames:
        if basename not in audio_basenames:
            missing_audios.append(basename)

    if missing_refs:
        logger.warning(f"参照ファイルが見つからない音声ファイルがあります ({len(missing_refs)}件): {missing_refs[:5]}...")
    if missing_audios:
        logger.warning(f"音声ファイルが見つからない参照ファイルがあります ({len(missing_audios)}件): {missing_audios[:5]}...")

    if not matched_audio:
        raise FileNotFoundError(f"対応する音声ファイルと参照ファイルが見つかりませんでした。")

    logger.info(f"{len(matched_audio)} 組の音声/参照ファイルを検出しました。")
    return matched_audio, matched_ref

def get_output_dir(base_dir: Path, unique_suffix: str) -> Path:
    """一意のサフィックスを持つ出力ディレクトリパスを生成する"""
    output_dir = base_dir / unique_suffix
    ensure_dir(output_dir)
    return output_dir

def setup_python_path() -> None:
    """
    PYTHONPATHを設定してプロジェクトモジュールをインポートできるようにする
    """
    project_root = str(get_project_root())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.debug(f"PYTHONPATHにプロジェクトルートを追加: {project_root}")
    
    # srcディレクトリがある場合、それもパスに追加
    src_dir = os.path.join(project_root, 'src')
    if os.path.exists(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        logger.debug(f"PYTHONPATHにsrcディレクトリを追加: {src_dir}")
    
    # 環境変数を読み込む
    load_environment_variables()

# モジュールのインポート時にPythonパスを設定
setup_python_path() 