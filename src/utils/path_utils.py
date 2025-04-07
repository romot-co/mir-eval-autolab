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
from typing import List, Optional, Union

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
    # 環境変数からワークスペースパスを取得
    env_workspace = os.environ.get('MIREX_WORKSPACE')
    if env_workspace:
        workspace_dir = Path(env_workspace)
        logger.debug(f"環境変数からワークスペースを取得: {workspace_dir}")
    else:
        # デフォルトパスの設定（優先順位: プロジェクトルート > ホームディレクトリ > 一時ディレクトリ）
        project_workspace = get_project_root() / 'mcp_workspace'
        home_workspace = Path.home() / '.mirex_workspace'
        
        if project_workspace.exists() or project_workspace.parent.exists():
            workspace_dir = project_workspace
        elif home_workspace.exists() or home_workspace.parent.exists():
            workspace_dir = home_workspace
        else:
            # 最終手段として一時ディレクトリを使用
            workspace_dir = Path(tempfile.gettempdir()) / 'mirex_workspace'
            logger.warning(f"ワークスペースディレクトリが見つかりません。一時ディレクトリを使用: {workspace_dir}")
    
    # ディレクトリが存在しない場合は作成
    workspace_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"ワークスペースディレクトリ: {workspace_dir}")
    
    return workspace_dir

def get_evaluation_results_dir() -> Path:
    """
    評価結果ディレクトリのパスを取得する
    
    環境変数 MIREX_OUTPUT_DIR が設定されている場合はその値を使用し、
    設定されていない場合はデフォルトパスを使用します。
    
    Returns
    -------
    Path
        評価結果ディレクトリのパス
    """
    env_output_dir = os.environ.get('MIREX_OUTPUT_DIR')
    if env_output_dir:
        results_dir = Path(env_output_dir)
    else:
        results_dir = get_project_root() / 'evaluation_results'
    
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def get_grid_search_results_dir() -> Path:
    """
    グリッドサーチ結果ディレクトリのパスを取得する
    
    Returns
    -------
    Path
        グリッドサーチ結果ディレクトリのパス
    """
    results_dir = get_project_root() / 'grid_search_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def get_improved_versions_dir() -> Path:
    """
    改善バージョンディレクトリのパスを取得する
    
    Returns
    -------
    Path
        改善バージョンディレクトリのパス
    """
    versions_dir = get_workspace_dir() / 'improved_versions'
    versions_dir.mkdir(parents=True, exist_ok=True)
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
    session_dir = get_workspace_dir() / 'sessions' / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
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
        audio_dir = Path(env_audio_dir)
    else:
        audio_dir = get_project_root() / 'data' / 'synthesized' / 'audio'
    
    if not audio_dir.exists():
        logger.warning(f"音声ディレクトリが存在しません: {audio_dir}")
    
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
        label_dir = Path(env_label_dir)
    else:
        label_dir = get_project_root() / 'data' / 'synthesized' / 'labels'
    
    if not label_dir.exists():
        logger.warning(f"ラベルディレクトリが存在しません: {label_dir}")
    
    return label_dir

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
        detector_versions_dir = improved_dir / detector_name
        
        if detector_versions_dir.exists():
            # 最新バージョンを検索
            version_files = list(detector_versions_dir.glob(f'**/{detector_name}_v*.py'))
            if version_files:
                version_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                logger.debug(f"改善バージョンを使用: {version_files[0]}")
                return version_files[0]
    
    # 標準の検出器ディレクトリをチェック
    detector_path = get_project_root() / 'src' / 'detectors' / f'{detector_name}.py'
    if detector_path.exists():
        return detector_path
    
    # 小文字でファイル名を試す
    detector_path = get_project_root() / 'src' / 'detectors' / f'{detector_name.lower()}.py'
    if detector_path.exists():
        return detector_path
    
    # クラス名から検索
    detectors_dir = get_project_root() / 'src' / 'detectors'
    try:
        for file_path in detectors_dir.glob('*.py'):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if f'class {detector_name}' in content and ('Detector' in detector_name or 'detector' in content.lower()):
                            logger.debug(f"検出器ファイルを見つけました: {file_path}")
                            return file_path
                except Exception as e:
                    logger.warning(f"ファイル読み込み中にエラーが発生しました: {file_path}, {str(e)}")
    except Exception as e:
        logger.error(f"検出器の検索中にエラーが発生しました: {e}")
    
    # 見つからない場合はNoneを返す
    logger.warning(f"検出器が見つかりませんでした: {detector_name}")
    return None

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