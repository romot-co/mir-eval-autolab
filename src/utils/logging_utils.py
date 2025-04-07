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
        
    Returns
    -------
    logging.Logger
        設定済みのロガー
    """
    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 指定された名前のロガーを取得
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # すべてのロガーのレベルを設定（サブモジュールのロガーも含む）
    for log_name in logging.root.manager.loggerDict:
        logging.getLogger(log_name).setLevel(level)
    
    # ハンドラがすでに設定されている場合は追加しない
    if logger.handlers:
        return logger
    
    # コンソールハンドラの設定
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level if console_level is not None else level)
    
    # フォーマッタの設定
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    
    # ロガーにハンドラを追加
    logger.addHandler(console_handler)
    
    # ファイルハンドラの設定（出力ディレクトリが指定されている場合）
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level if file_level is not None else level)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
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