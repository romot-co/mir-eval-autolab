#!/usr/bin/env python3
"""
音楽検出アルゴリズム評価実行スクリプト

このスクリプトは、音楽検出アルゴリズムの評価を実行するためのコマンドラインインターフェースを提供します。
評価対象の音声ファイルと正解ラベルを指定して、検出器の性能を測定します。

使用例:
    # 基本的な使用方法（ディレクトリ内の全ての音声ファイルを評価）
    python run_evaluate.py --detectors PZSTDDetector --audio-dir data/synthesized/audio --reference-dir data/synthesized/labels
    
    # 複数の検出器を同時に評価
    python run_evaluate.py --detectors PZSTDDetector,ONDEDetector --audio-dir data/synthesized/audio --reference-dir data/synthesized/labels
    
    # 検出器のパラメータを指定して評価
    python run_evaluate.py --detectors PZSTDDetector --detector-params '{"PZSTDDetector": {"f0_score_threshold": 0.3, "HCF_ONSET_PEAK_THRESH": 0.15}}' --audio-dir data/synthesized/audio --reference-dir data/synthesized/labels
    
    # 結果をCSVとJSONで保存
    python run_evaluate.py --detectors PZSTDDetector --audio-dir data/synthesized/audio --reference-dir data/synthesized/labels --save-results-csv --save-results-json
    
    # 詳細なログを出力
    python run_evaluate.py --detectors PZSTDDetector --audio-dir data/synthesized/audio --reference-dir data/synthesized/labels --log-level DEBUG
"""

import os
import sys
import json
import argparse
import tempfile
import logging
from pathlib import Path
import tabulate
import pandas as pd
from colorama import init, Fore, Style

# カラー出力の初期化
init()

# ロガー設定
logger = logging.getLogger("run_evaluate")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="音楽検出アルゴリズム評価スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 入力ファイル関連の引数グループ
    input_group = parser.add_argument_group('入力ファイル')
    input_file_group = input_group.add_mutually_exclusive_group()
    input_file_group.add_argument('--audio', help='評価する音声ファイルのパス')
    input_file_group.add_argument('--audio-dir', help='評価する音声ファイルが含まれるディレクトリ')
    
    ref_file_group = input_group.add_mutually_exclusive_group()
    ref_file_group.add_argument('--reference', help='正解ラベルファイルのパス')
    ref_file_group.add_argument('--reference-dir', help='正解ラベルファイルが含まれるディレクトリ')
    
    input_group.add_argument('--audio-pattern', default='*.wav', help='音声ファイルのパターン（デフォルト: *.wav）')
    input_group.add_argument('--reference-pattern', default='*.txt', help='正解ラベルファイルのパターン（デフォルト: *.txt）')
    
    # 検出器関連の引数グループ
    detector_group = parser.add_argument_group('検出器')
    detector_group.add_argument('--detectors', required=True, help='評価する検出器の名前（カンマ区切りで複数指定可能）')
    detector_group.add_argument('--detector-params', help='検出器のパラメータ（JSON形式）')
    
    # 評価パラメータの引数グループ
    eval_group = parser.add_argument_group('評価パラメータ')
    eval_group.add_argument('--frame-size', type=float, default=0.01, help='評価フレームサイズ（秒）（デフォルト: 0.01）')
    eval_group.add_argument('--tolerance', type=float, default=0.05, help='評価トレランス（秒）（デフォルト: 0.05）')
    
    # 出力関連の引数グループ
    output_group = parser.add_argument_group('出力オプション')
    output_group.add_argument('--output-dir', help='評価結果の出力ディレクトリ（デフォルト: evaluation_results）')
    output_group.add_argument('--save-results-csv', action='store_true', help='評価結果をCSVとして保存')
    output_group.add_argument('--save-results-json', action='store_true', help='評価結果をJSONとして保存')
    output_group.add_argument('--save-plots', action='store_true', help='評価結果のプロットを保存')
    output_group.add_argument('--plot-format', default='png', help='プロットのファイル形式（デフォルト: png）')
    
    # その他の引数グループ
    other_group = parser.add_argument_group('その他の設定')
    other_group.add_argument('--config', help='設定ファイルのパス')
    other_group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='ログレベル（デフォルト: INFO）')
    
    return parser.parse_args()

def format_results(results):
    """評価結果を見やすい形式にフォーマット"""
    if results['status'] != 'success':
        return f"評価に失敗しました: {results.get('error', '不明なエラー')}"
    
    # 結果からデータフレームを作成
    rows = []
    
    for detector_name, detector_results in results['detector_results'].items():
        for params_key, metrics in detector_results.items():
            # パラメータキーからパラメータを抽出
            param_str = "default" if params_key == "default" else params_key
            
            # 評価指標を行に追加
            rows.append({
                "検出器": detector_name,
                "パラメータ": param_str,
                "適合率": metrics.get('precision', 0),
                "再現率": metrics.get('recall', 0),
                "F値": metrics.get('f_measure', 0),
                "精度": metrics.get('accuracy', 0),
            })
    
    # データフレームに変換
    df = pd.DataFrame(rows)
    
    # 結果をテーブル形式でフォーマット
    headers = ["検出器", "パラメータ", "適合率", "再現率", "F値", "精度"]
    table_data = []
    
    for _, row in df.iterrows():
        formatted_row = [
            Fore.CYAN + str(row["検出器"]) + Style.RESET_ALL,
            Fore.YELLOW + str(row["パラメータ"]) + Style.RESET_ALL,
            f"{float(row['適合率']):.4f}",
            f"{float(row['再現率']):.4f}",
            Fore.GREEN + f"{float(row['F値']):.4f}" + Style.RESET_ALL,
            f"{float(row['精度']):.4f}"
        ]
        table_data.append(formatted_row)
    
    return tabulate.tabulate(table_data, headers=headers, tablefmt="grid")

def main():
    """メイン実行関数"""
    args = parse_arguments()
    
    # ログレベルの設定
    logger.setLevel(getattr(logging, args.log_level))
    
    # 一時ファイルを作成して結果を受け取る
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        result_file_path = tmp.name
    
    try:
        # 環境変数に結果ファイルのパスを設定
        os.environ['EVALUATE_RESULT_FILE'] = result_file_path
        
        # evaluate_cli.pyを呼び出す
        cmd_args = sys.argv[1:]
        
        # src/cliディレクトリへの相対パスを取得
        script_dir = Path(__file__).resolve().parent
        cli_path = script_dir / "src" / "cli" / "evaluate_cli.py"
        
        if not cli_path.exists():
            # プロジェクトルートからの相対パスも試す
            cli_path = script_dir / "src" / "cli" / "evaluate_cli.py"
            if not cli_path.exists():
                logger.error("evaluate_cli.pyが見つかりません")
                return 1
        
        # evaluate_cli.pyを実行
        logger.info(f"評価を開始します...")
        exit_code = os.system(f"python {cli_path} {' '.join(cmd_args)}")
        
        if exit_code != 0:
            logger.error(f"evaluate_cliの実行に失敗しました（終了コード: {exit_code}）")
            return exit_code
        
        # 結果ファイルから結果を読み込む
        try:
            with open(result_file_path, 'r') as f:
                results = json.load(f)
                
            # 結果を表示
            formatted_results = format_results(results)
            print("\n" + "=" * 80)
            print("評価結果サマリー:")
            print(formatted_results)
            print("=" * 80)
            
            # 出力ディレクトリ情報を表示
            if 'output_dir' in results:
                output_dir = results['output_dir']
                if os.path.exists(output_dir) and os.listdir(output_dir):
                    print(f"\n評価結果は {output_dir} ディレクトリに保存されました\n")
            
        except Exception as e:
            logger.error(f"結果の読み込みに失敗しました: {str(e)}")
            return 1
        
    finally:
        # 一時ファイルを削除
        try:
            if os.path.exists(result_file_path):
                os.unlink(result_file_path)
        except:
            pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 