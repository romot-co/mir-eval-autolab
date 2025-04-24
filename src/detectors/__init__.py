"""
検出器モジュール

このモジュールは、音楽検出アルゴリズムの基底クラスと動的ロードメカニズムを提供します。
検出器クラスはBaseDetectorを継承し、@register_detectorデコレータで登録することができます。
"""

import glob
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

# ロガーの設定
logger = logging.getLogger(__name__)

# 検出器クラスのレジストリ
DETECTOR_REGISTRY = {}
# 検出器ファイルのレジストリ（ファイル名 -> モジュール名のマッピング）
DETECTOR_FILE_REGISTRY = {}


class DetectorMetadata:
    """検出器クラスのメタデータを格納するクラス"""

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "1.0",
        params: Dict[str, Any] = None,
        file_path: str = None,
    ):
        """
        検出器メタデータの初期化

        Parameters
        ----------
        name : str
            検出器クラスの名前
        description : str, optional
            検出器の説明（デフォルト: ""）
        version : str, optional
            検出器のバージョン（デフォルト: "1.0"）
        params : Dict[str, Any], optional
            デフォルトパラメータ（デフォルト: None）
        file_path : str, optional
            検出器のソースファイルパス（デフォルト: None）
        """
        self.name = name
        self.description = description
        self.version = version
        self.params = params or {}
        self.file_path = file_path

    def to_dict(self) -> Dict[str, Any]:
        """
        メタデータを辞書形式で取得

        Returns
        -------
        Dict[str, Any]
            メタデータの辞書表現
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "params": self.params,
            "file_path": self.file_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectorMetadata":
        """
        辞書からメタデータオブジェクトを生成

        Parameters
        ----------
        data : Dict[str, Any]
            メタデータの辞書表現

        Returns
        -------
        DetectorMetadata
            メタデータオブジェクト
        """
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            params=data.get("params", {}),
            file_path=data.get("file_path"),
        )


# 相対インポートから絶対インポートに変更
from src.detectors.base_detector import BaseDetector


def register_detector(
    cls=None,
    *,
    name: str = None,
    description: str = None,
    version: str = None,
    params: Dict[str, Any] = None,
):
    """
    検出器クラスをレジストリに登録するデコレータ

    Parameters
    ----------
    cls : Type, optional
        登録する検出器クラス
    name : str, optional
        検出器クラスの名前（指定しない場合はクラス名を使用）
    description : str, optional
        検出器の説明
    version : str, optional
        検出器のバージョン
    params : Dict[str, Any], optional
        デフォルトパラメータ

    Returns
    -------
    Callable
        デコレータ関数
    """

    def decorator(detector_cls):
        # クラスがBaseDetectorを継承しているか確認
        if not issubclass(detector_cls, BaseDetector):
            logger.warning(f"{detector_cls.__name__}はBaseDetectorを継承していません")
            return detector_cls

        # メタデータを設定
        detector_name = name or detector_cls.__name__
        detector_description = description or detector_cls.__doc__ or ""
        detector_version = version or getattr(detector_cls, "version", "1.0")
        detector_params = params or getattr(detector_cls, "default_params", {})

        # クラスのソースファイルパスを取得
        try:
            source_file = inspect.getfile(detector_cls)
        except (TypeError, ValueError):
            source_file = None

        # メタデータオブジェクトを作成
        metadata = DetectorMetadata(
            name=detector_name,
            description=detector_description,
            version=detector_version,
            params=detector_params,
            file_path=source_file,
        )

        # クラスにメタデータを設定
        detector_cls.metadata = metadata

        # レジストリに登録
        DETECTOR_REGISTRY[detector_name] = detector_cls
        logger.debug(f"検出器 {detector_name} をレジストリに登録しました")

        return detector_cls

    # デコレータとして直接呼び出された場合
    if cls is not None:
        return decorator(cls)

    # パラメータ付きでデコレータが呼び出された場合
    return decorator


def get_registered_detector(detector_name):
    """登録済みの検出器を取得する"""
    if detector_name in DETECTOR_REGISTRY:
        return DETECTOR_REGISTRY[detector_name]
    return None


def load_detector_from_file(
    file_path: str, detector_name: str = None
) -> Type[BaseDetector]:
    """
    ファイルから検出器クラスを動的にロードする

    Parameters
    ----------
    file_path : str
        検出器ファイルのパス
    detector_name : str, optional
        ロードする検出器クラスの名前（指定しない場合は自動検出）

    Returns
    -------
    Type[BaseDetector]
        ロードされた検出器クラス

    Raises
    ------
    ImportError
        ファイルのロードに失敗した場合
    ValueError
        検出器クラスが見つからない場合
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise ImportError(f"ファイルが存在しません: {file_path}")

    try:
        # モジュール名を生成
        module_name = f"dynamic_detector_{file_path.stem}"

        # モジュールをロード
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"モジュール仕様の取得に失敗しました: {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # DETECTOR_FILE_REGISTRYに登録
        DETECTOR_FILE_REGISTRY[str(file_path)] = module_name

        # 検出器クラスが明示的に指定されている場合
        if detector_name is not None:
            # 直接クラスを取得
            if hasattr(module, detector_name):
                detector_cls = getattr(module, detector_name)
                if issubclass(detector_cls, BaseDetector):
                    # レジストリに登録されていない場合は登録
                    if detector_name not in DETECTOR_REGISTRY:
                        register_detector(detector_cls)
                    return detector_cls

            # レジストリから取得
            if detector_name in DETECTOR_REGISTRY:
                return DETECTOR_REGISTRY[detector_name]

            raise ValueError(
                f"検出器クラス {detector_name} が見つかりません: {file_path}"
            )

        # 検出器クラスが指定されていない場合は自動検出
        # レジストリに登録されているクラスを優先
        found_detectors = []

        # モジュール内のすべてのクラスを検査
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseDetector) and obj != BaseDetector:
                found_detectors.append(obj)
                # レジストリに登録されていない場合は登録
                if name not in DETECTOR_REGISTRY:
                    register_detector(obj)

        if not found_detectors:
            raise ValueError(
                f"BaseDetectorを継承するクラスが見つかりません: {file_path}"
            )

        # 複数のクラスが見つかった場合は警告
        if len(found_detectors) > 1:
            logger.warning(
                f"複数の検出器クラスが見つかりました。最初のクラスを使用します: {[cls.__name__ for cls in found_detectors]}"
            )

        return found_detectors[0]

    except Exception as e:
        logger.error(f"検出器のロードに失敗しました: {file_path}")
        logger.error(traceback.format_exc())
        raise ImportError(f"検出器のロードに失敗しました: {e}")


def get_detector_class(detector_name: str) -> Type[BaseDetector]:
    """
    レジストリまたはファイルから検出器クラスを取得する

    Parameters
    ----------
    detector_name : str
        検出器名

    Returns
    -------
    Type[BaseDetector]
        検出器クラス

    Raises
    ------
    ValueError
        検出器が見つからない場合
    """
    # レジストリから検出器クラスを取得
    if detector_name in DETECTOR_REGISTRY:
        return DETECTOR_REGISTRY[detector_name]

    # 以下のパスを検索:
    # 1. プロジェクト内の標準検出器ディレクトリ (src/detectors)
    # 2. 改善バージョンディレクトリ (mcp_workspace/improved_versions)
    search_paths = []

    # src/detectorsディレクトリ
    module_dir = Path(__file__).resolve().parent
    search_paths.append(module_dir)

    # mcp_workspace/improved_versionsディレクトリ
    try:
        # まず、システムパスからプロジェクトルートを取得
        sys_path_items = [Path(p) for p in sys.path]
        project_roots = [
            p for p in sys_path_items if (p / "src" / "detectors").exists()
        ]

        if project_roots:
            project_root = project_roots[0]
            workspace_dir = project_root / "mcp_workspace"
            if workspace_dir.exists():
                improved_versions_dir = workspace_dir / "improved_versions"
                if improved_versions_dir.exists():
                    search_paths.append(improved_versions_dir)

        # カスタムワークスペースディレクトリ
        if "MIREX_WORKSPACE" in os.environ:
            workspace_dir = Path(os.environ["MIREX_WORKSPACE"])
            improved_versions_dir = workspace_dir / "improved_versions"
            if improved_versions_dir.exists():
                search_paths.append(improved_versions_dir)
    except Exception as e:
        logger.warning(f"ワークスペースディレクトリの検索中にエラーが発生しました: {e}")

    # 各パスで検出器ファイルを検索
    for search_path in search_paths:
        detector_file = search_path / f"{detector_name}.py"
        if detector_file.exists():
            try:
                return load_detector_from_file(detector_file, detector_name)
            except Exception as e:
                logger.warning(
                    f"検出器ファイルのロードに失敗しました: {detector_file}: {e}"
                )

    raise ValueError(f"検出器 {detector_name} が見つかりません")


def get_all_detectors() -> Dict[str, Type[BaseDetector]]:
    """
    利用可能なすべての検出器クラスを取得する

    Returns
    -------
    Dict[str, Type[BaseDetector]]
        検出器名とクラスのマッピング
    """
    # 既に登録されている検出器を取得
    detectors = DETECTOR_REGISTRY.copy()

    # プロジェクト内の標準検出器ディレクトリからファイルを検索
    module_dir = Path(__file__).resolve().parent
    detector_files = list(module_dir.glob("*.py"))

    # 各ファイルから検出器をロード
    for file_path in detector_files:
        if file_path.name == "__init__.py" or file_path.name == "base_detector.py":
            continue

        try:
            detector_class = load_detector_from_file(file_path)
            detector_name = detector_class.__name__
            detectors[detector_name] = detector_class
        except Exception as e:
            logger.warning(f"検出器ファイルのロードに失敗しました: {file_path}: {e}")

    return detectors


def create_detector(detector_name: str, **params) -> BaseDetector:
    """
    検出器インスタンスを作成する

    Parameters
    ----------
    detector_name : str
        検出器名
    **params
        検出器のパラメータ

    Returns
    -------
    BaseDetector
        検出器インスタンス

    Raises
    ------
    ValueError
        検出器が見つからない場合
    """
    detector_class = get_detector_class(detector_name)
    return detector_class(**params)


def _setup_python_path():
    """
    PYTHONPATHを設定し、プロジェクトモジュールをインポートできるようにする
    """
    try:
        # src.utils.path_utils の機能を使用
        from src.utils.path_utils import setup_python_path

        setup_python_path()
    except ImportError:
        # path_utils が利用できない場合は内部実装を使用
        logger.debug("src.utils.path_utils が利用できないため、内部実装を使用します")

        # 現在のディレクトリから親ディレクトリを探索
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # src
        root_dir = os.path.dirname(parent_dir)  # プロジェクトルート

        # プロジェクトルートをPYTHONPATHに追加
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
            logger.debug(f"PYTHONPATHにプロジェクトルートを追加: {root_dir}")

        # srcディレクトリをPYTHONPATHに追加
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            logger.debug(f"PYTHONPATHにsrcディレクトリを追加: {parent_dir}")


# モジュールの初期化時にPYTHONPATHを設定
_setup_python_path()

# 既存の検出器モジュールをロード
detector_files = sorted(
    [
        f
        for f in Path(__file__).resolve().parent.glob("*.py")
        if f.name
        not in [
            "__init__.py",
            "base_detector.py",
            "numba_helpers.py",
            "sihr_hypothesis.py",
        ]
    ]
)
for detector_file in detector_files:
    try:
        load_detector_from_file(detector_file)
    except Exception as e:
        logger.warning(f"検出器の初期ロードに失敗しました: {detector_file.name}: {e}")

# KROMARDetectorを明示的にインポート
try:
    from src.detectors.kromar_detector import KROMARDetector
except ImportError:
    logger.warning("KROMARDetectorのインポートに失敗しました")

# ノート検出器 (mir_evalのnote-based評価) と、
# フレームベースのピッチ検出器 (mir_evalのframe-based評価) を統一的に扱うため、
# 以下のキーを返すことを推奨します：

# 【ノート単位 (必須)】
#  - intervals: shape=(N,2), [[onset, offset], ...]
#  - note_pitches: shape=(N,), 各ノートのピッチ(Hz)

# 【フレーム単位 (オプション)】
#  - times: shape=(M,), フレームの先頭時刻(秒)
#  - freqs: shape=(M,), フレームごとの推定ピッチ(Hz)

# 【共通メタ情報】
#  - detector_name: str
#  - detection_time: float

# ノート評価には intervals/note_pitches を、
# フレーム評価には times/freqs を使用します。

# 必要な関数とベースクラスを公開 (既存の静的インポートのみをリスト)
__all__ = [
    "BaseDetector",
    "register_detector",
    "get_registered_detector",
    "DETECTOR_REGISTRY",  # 公開レジストリを追加
    # 必要に応じて、明示的にエクスポートしたい「静的な」検出器クラス
    "CriteriaDetector",
    "PZSTDDetector",
    "KROMARDetector",  # KROMARDetectorを追加
]
