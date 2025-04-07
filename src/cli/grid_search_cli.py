#!/usr/bin/env python
"""
グリッドサーチのコマンドラインインターフェース。

このモジュールは、グリッドサーチ機能をコマンドラインから実行するためのインターフェースを提供します。
検出器のパラメータを最適化するためのグリッドサーチを実行したり、グリッドサーチの設定ファイルを作成したりするコマンドを提供します。
"""

import os
import sys
import argparse
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple
import yaml
import json

# 新しいパッケージからインポート
from src.evaluation.grid_search.core import run_grid_search, create_grid_config
from src.utils.logging_utils import get_logger, setup_logging


def find_matching_files(directory: str, pattern: str) -> List[str]:
    """
    指定されたディレクトリからパターンに一致するファイルを検索します。
    
    Parameters
    ----------
    directory : str
        検索するディレクトリのパス
    pattern : str
        検索するファイル名パターン（例: "*.wav"）
        
    Returns
    -------
    List[str]
        一致したファイルパスのリスト
    """
    import glob
    import os
    
    search_path = os.path.join(directory, pattern)
    return sorted(glob.glob(search_path))


def match_audio_and_reference_files(audio_files: List[str], ref_files: List[str], 
                                    audio_pattern: str, ref_pattern: str) -> List[Tuple[str, str]]:
    """
    音声ファイルと参照ファイルをマッチングしてペアを作成します。
    
    Parameters
    ----------
    audio_files : List[str]
        音声ファイルパスのリスト
    ref_files : List[str]
        参照ファイルパスのリスト
    audio_pattern : str
        音声ファイルの検索パターン
    ref_pattern : str
        参照ファイルの検索パターン
        
    Returns
    -------
    List[Tuple[str, str]]
        マッチングされた (音声ファイル, 参照ファイル) のペアのリスト
    """
    import os
    
    # ファイル名を拡張子なしで取得する関数
    def get_basename_without_ext(filepath: str) -> str:
        return os.path.splitext(os.path.basename(filepath))[0]
    
    # 参照ファイルのマップを作成
    ref_map = {get_basename_without_ext(ref): ref for ref in ref_files}
    
    # マッチングするペアを作成
    file_pairs = []
    for audio in audio_files:
        base_name = get_basename_without_ext(audio)
        if base_name in ref_map:
            file_pairs.append((audio, ref_map[base_name]))
    
    return file_pairs


def parse_args():
    """
    コマンドライン引数を解析します。

    Returns
    -------
    argparse.Namespace
        解析されたコマンドライン引数
    """
    parser = argparse.ArgumentParser(
        description='検出器のパラメータ最適化のためのグリッドサーチツール'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='実行するコマンド')
    
    # グリッドサーチ実行コマンド
    run_parser = subparsers.add_parser('run', help='グリッドサーチを実行')
    run_parser.add_argument('--config', required=True, help='基本設定ファイルのパス')
    run_parser.add_argument('--grid-config', required=True, help='グリッドサーチ設定ファイルのパス')
    run_parser.add_argument('--output-dir', required=True, help='結果を保存するディレクトリのパス')
    run_parser.add_argument('--best-metric', default='note.f_measure', 
                          help='最適化する評価指標 (例: note.f_measure, onset.precision) (デフォルト: note.f_measure)')
    run_parser.add_argument('--save-plots', action='store_true', help='検出結果のプロットを保存する')
    run_parser.add_argument('--verbose', action='store_true', help='詳細なログ出力を有効にする')
    
    # グリッドサーチ設定ファイル作成コマンド
    create_config_parser = subparsers.add_parser('create-config', help='グリッドサーチ設定ファイルを作成')
    create_config_parser.add_argument('--detector', required=True, help='検出器の名前')
    create_config_parser.add_argument('--audio-dir', required=True, help='音声ファイルのディレクトリパス')
    create_config_parser.add_argument('--reference-dir', required=True, help='参照ファイルのディレクトリパス')
    create_config_parser.add_argument('--output', required=True, help='出力する設定ファイルのパス')
    create_config_parser.add_argument('--param', action='append', nargs='+', 
                                    help='パラメータ名とその値の範囲 (例: --param onset_threshold 0.1 0.2 0.3)')
    create_config_parser.add_argument('--save-plots', action='store_true', help='検出結果のプロットを保存する')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    設定ファイルを読み込みます。
    
    Parameters
    ----------
    config_path : str
        設定ファイルのパス
        
    Returns
    -------
    Dict[str, Any]
        読み込まれた設定
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"設定ファイルの読み込みに失敗しました: {str(e)}")


def load_grid_config(grid_config_path: str) -> Dict[str, Any]:
    """
    グリッドサーチ設定ファイルを読み込みます。
    
    Parameters
    ----------
    grid_config_path : str
        グリッドサーチ設定ファイルのパス
        
    Returns
    -------
    Dict[str, Any]
        読み込まれたグリッドサーチ設定
    """
    try:
        with open(grid_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"グリッドサーチ設定ファイルの読み込みに失敗しました: {str(e)}")


def main():
    """
    メイン実行関数
    """
    # コマンドライン引数の解析
    args = parse_args()
    
    # ロギングの設定
    logger = setup_logging(args.log_level)
    logger.info("グリッドサーチを開始します...")
    
    # 設定ファイルの読み込み
    config = load_config(args.config)
    grid_config = load_grid_config(args.grid_config)
    
    # 出力ディレクトリの作成
    output_dir = args.output_dir or "grid_search_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # run_grid_search関数を直接呼び出す
    try:
        # インポートしている関数を直接使用
        results = run_grid_search(
            config_path=args.config,
            grid_config_path=args.grid_config,
            output_dir=output_dir,
            best_metric=args.best_metric,
            logger=logger
        )
        
        if results:
            # 結果を表示
            print("\n" + "=" * 60)
            print(f"グリッドサーチの結果:")
            print(f"  検出器: {grid_config.get('detector')}")
            print(f"  評価指標: {args.best_metric}")
            print(f"  最適なパラメータ:")
            for param, value in results.get('params', {}).items():
                print(f"    - {param}: {value}")
            
            # 最適なスコアを取得
            metrics = results.get('metrics', {})
            if args.best_metric in metrics:
                print(f"  スコア: {metrics[args.best_metric]:.4f}")
            else:
                category, metric = args.best_metric.split('.')
                if category in metrics and isinstance(metrics[category], dict) and metric in metrics[category]:
                    print(f"  スコア: {metrics[category][metric]:.4f}")
            
            print(f"  詳細な結果: {os.path.join(output_dir, 'grid_search_results.csv')}")
            print("=" * 60)
            
            return 0
        else:
            logger.error("グリッドサーチが失敗しました")
            return 1
    
    except Exception as e:
        logger.error(f"グリッドサーチの実行中にエラーが発生しました: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 