"""
ロギングの設定を統一するためのユーティリティモジュール

このモジュールは、アプリケーション全体でロギングの設定を統一するためのユーティリティ関数を提供します。
"""

import logging
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(name: str, output_dir: Optional[str] = None, 
                 level: int = logging.INFO, 
                 console_level: Optional[int] = None,
                 file_level: Optional[int] = None) -> logging.Logger:
    """
    ロガーの設定
    
    Parameters
    ----------
    name : str
        ロガー名
    output_dir : str, optional
        ログファイルの出力ディレクトリ, by default None
    level : int, optional
        ログレベル, by default logging.INFO
    console_level : int, optional
        コンソールハンドラのログレベル (Noneの場合はlevelと同じ)
    file_level : int, optional
        ファイルハンドラのログレベル (Noneの場合はlevelと同じ)
        
    Returns
    -------
    logging.Logger
        設定済みのロガー
    """
    logger = logging.getLogger(name)

    # --- 早期リターンを復活させる --- #
    # ハンドラがすでに設定されている場合は、レベル変更も含めて何もしないで返す
    # (意図しない設定上書きを防ぐ)
    if logger.hasHandlers(): # .handlersリストを直接チェックするよりhasHandlers()が推奨される
        # 既存ロガーのレベルが意図せず変更されないように注意
        # logger.setLevel(level) # ここではレベルを設定しない
        return logger

    # ハンドラがない場合のみ、新しい設定を適用
    # # 念のため、既存のハンドラをクリアしてから新しいハンドラを追加する
    # # (テスト環境などで予期せぬハンドラが存在する可能性に対処)
    # for handler in logger.handlers[:]:
    #     logger.removeHandler(handler)
    #     handler.close()
    # # logger.handlers.clear() # より直接的なクリア方法だが、上記で十分か。

    # --- ロガー自体のレベル設定は行わない --- 
    # ロガー自体のレベルは設定せず (デフォルトはNOTSET=0)、
    # ハンドラのレベルでフィルタリングするようにする
    # logger.setLevel(level)
    # propagateをTrueに設定（デフォルトのはずだが明示的に）
    logger.propagate = True

    # --- ルートロガーや他のロガーへの影響は最小限にする --- 
    # ルートロガーのレベル設定は、他のライブラリのログに影響する可能性があるので注意
    # root_logger = logging.getLogger()
    # root_logger.setLevel(level) # 必要性が明確でない限りコメントアウト推奨
    
    # 他の既存ロガーのレベルを一括で変更する処理も、影響範囲が大きいので削除
    # for log_name in logging.root.manager.loggerDict:
    #     if log_name != name: # 自分自身は除く
    #         logging.getLogger(log_name).setLevel(level)
    
    # --- ハンドラの設定 --- 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # コンソールハンドラの設定
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level if console_level is not None else level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラの設定（出力ディレクトリが指定されている場合）
    if output_dir:
        try:
            output_path = Path(output_dir).resolve()
            output_path.mkdir(parents=True, exist_ok=True)
            # ログファイル名にプロセスIDを含めるなどして衝突を避けることを検討
            log_file = output_path / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.log'
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8') # エンコーディング指定
            file_handler.setLevel(file_level if file_level is not None else level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # ファイルハンドラ設定でエラーが起きてもコンソールログは機能するように
            logger.error(f"Failed to set up file handler for logger '{name}': {e}", exc_info=True)
    
    return logger


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    ロガーを取得します。既存のロガーがある場合はそれを返し、なければ新しく作成します。

    Parameters
    ----------
    name : str
        ロガーの名前
    log_level : int, optional
        ログレベル, by default logging.INFO

    Returns
    -------
    logging.Logger
        ロガー
    """
    logger = logging.getLogger(name)
    
    # ロガーがすでに設定されている場合はそのまま返す
    if logger.handlers:
        return logger
    
    # 新しいロガーを設定
    logger.setLevel(log_level)
    
    # コンソールハンドラの設定
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # フォーマッタの設定
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # ロガーにハンドラを追加
    logger.addHandler(console_handler)
    
    return logger 