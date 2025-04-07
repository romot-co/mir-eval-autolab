#!/usr/bin/env python3
"""
自動改善ランナー

このスクリプトはauto_improver.pyの機能を含む自動改善サイクルを実行するためのクライアントです。

使用方法:
    python run_auto_improver.py --detector PZSTDDetector --goal "ノイズに強い検出器にする" --iterations 3
"""

import os
import sys
import logging
import argparse

# プロジェクトルートをPYTHONPATHに追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# src.utils.path_utils を使用して環境変数とパスを設定
from src.utils.path_utils import setup_python_path
setup_python_path()

from auto_improver import AutoImprover

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('auto_improver.log')
    ]
)
logger = logging.getLogger('run_auto_improver')

def main():
    parser = argparse.ArgumentParser(
        description='自動改善ランナー - 科学的発見と可視化機能付き',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python run_auto_improver.py --detector PZSTDDetector --goal "ノイズに強い検出器にする"
  python run_auto_improver.py --detector PZSTDDetector --goal "低周波成分の検出精度を向上させる" --iterations 5 --no-visualization
        """
    )
    
    parser.add_argument('--detector', required=True, help='改善対象の検出器名')
    parser.add_argument('--goal', required=True, help='改善の目標')
    parser.add_argument('--iterations', type=int, 
                      default=int(os.environ.get('MCP_MAX_ITERATIONS', 3)), 
                      help='改善イテレーションの回数（デフォルト: 環境変数またはファイル設定、指定なしなら3）')
    parser.add_argument('--no-visualization', action='store_true', help='可視化機能を無効にする')
    parser.add_argument('--no-science', action='store_true', help='科学的発見の自動化機能を無効にする')
    parser.add_argument('--server-url', 
                      default=os.environ.get('MCP_SERVER_URL'),
                      help='MCPサーバーのURL（デフォルト: 環境変数の値）')
    parser.add_argument('--config', 
                      default=os.environ.get('MCP_CONFIG_PATH', 'config.yaml'),
                      help='設定ファイルのパス（デフォルト: 環境変数またはconfig.yaml）')
    
    args = parser.parse_args()
    
    # 環境変数の設定
    if args.server_url:
        os.environ["MCP_SERVER_URL"] = args.server_url
    
    # 改善サイクルの実行
    improver = AutoImprover(config_path=args.config, server_url=args.server_url)
    
    try:
        result = improver.run_improvement_cycle(
            detector_name=args.detector,
            goal=args.goal,
            iterations=args.iterations,
            enable_visualization=not args.no_visualization,
            enable_science=not args.no_science
        )
        
        if result['status'] == 'success':
            logger.info(f"改善サイクルが完了しました: {result['session_id']}")
            if 'scientific_outputs' in result and result['scientific_outputs']:
                logger.info("科学的成果物:")
                for output_type, path in result['scientific_outputs'].items():
                    logger.info(f"- {output_type}: {path}")
        else:
            logger.error(f"改善サイクルが失敗しました: {result.get('error', '不明なエラー')}")
            return 1
        
        return 0
    except Exception as e:
        logger.error(f"改善サイクル中にエラーが発生しました: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 