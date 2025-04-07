#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
グリッドサーチ実行用のCLIラッパースクリプト
プロジェクトルートから呼び出すことを想定
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import yaml  # yamlファイルを読み込むためのインポート追加

# プロジェクトルートをパスに追加
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

# 必要なモジュールをインポート
from src.evaluation.grid_search.core import run_grid_search
from src.utils.logging_utils import get_logger, setup_logging

def load_config(config_path):
    """
    設定ファイルを読み込む
    
    Parameters
    ----------
    config_path : str
        設定ファイルのパス
        
    Returns
    -------
    dict
        設定データ
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

logger = get_logger(__name__)

def parse_args():
    """
    コマンドライン引数を解析する
    """
    parser = argparse.ArgumentParser(description="オンセット検出のグリッドサーチ実行ツール")
    
    # 必須パラメータ
    parser.add_argument("--config", required=True, help="設定ファイルのパス")
    parser.add_argument("--grid-config", required=True, help="グリッドサーチ設定ファイルのパス")
    
    # 出力関連
    parser.add_argument("--output-dir", help="結果の出力ディレクトリ（デフォルト: grid_search_results）")
    parser.add_argument("--best-metric", default="f_measure", help="最適化する評価指標（デフォルト: f_measure）")
    
    # ログレベル
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="ログレベル（デフォルト: INFO）")
    
    return parser.parse_args()

def main():
    """
    メイン実行関数
    """
    # コマンドライン引数の解析
    args = parse_args()
    
    # ロギングの設定
    logger = setup_logging(args.log_level)
    logger.info("グリッドサーチを開始します...")
    
    # 設定ファイルの存在確認
    if not os.path.exists(args.config):
        logger.error(f"設定ファイル {args.config} が見つかりません")
        return 1
    
    if not os.path.exists(args.grid_config):
        logger.error(f"グリッドサーチ設定ファイル {args.grid_config} が見つかりません")
        return 1
    
    # 出力ディレクトリの設定
    output_dir = args.output_dir or "grid_search_results"
    
    # グリッドサーチの実行
    try:
        # 直接run_grid_search関数を呼び出す
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
            
            # グリッド設定を読み込む
            grid_config = load_config(args.grid_config)
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
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 