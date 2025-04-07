#!/usr/bin/env python
"""
音楽検出アルゴリズムの評価スクリプト（改良版）

このスクリプトは、音楽検出アルゴリズムの性能を評価するためのコマンドラインインターフェースを提供します。
単一のCLIから複数の検出器と複数のファイルをまとめて評価し、MIREX準拠のオンセット/オフセット/ピッチ/ノート評価を
一括で行います。結果はJSON+PNG形式で出力します。
"""

import os
import sys
import json
import argparse
import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple
import glob
import traceback
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np

# 評価関連のインポート
from src.utils.detector_utils import get_detector_class
from src.evaluation.evaluation_runner import run_evaluation
from src.evaluation.evaluation_io import print_summary_statistics, create_summary_dataframe
from src.utils.logging_utils import get_logger


def parse_args():
    """
    コマンドライン引数を解析します。
        
    Returns
    -------
    argparse.Namespace
        解析されたコマンドライン引数
    """
    parser = argparse.ArgumentParser(
        description='MIREX準拠の音楽検出評価スクリプト'
    )
    
    # 入力ファイル関連の引数
    input_group = parser.add_argument_group('入力')
    input_group.add_argument('--audio', help='評価する単一の音声ファイルのパス')
    input_group.add_argument('--reference', help='単一の参照データファイルのパス')
    input_group.add_argument('--audio-dir', help='評価する音声ファイルが含まれるディレクトリ')
    input_group.add_argument('--reference-dir', help='参照データファイルが含まれるディレクトリ')
    input_group.add_argument('--audio-pattern', default='*.wav', help='音声ファイルのパターン（デフォルト: *.wav）')
    input_group.add_argument('--reference-pattern', default='*.json', help='参照ファイルのパターン（デフォルト: *.json）')
    
    # 検出器関連の引数
    detector_group = parser.add_argument_group('検出器')
    detector_group.add_argument('--detectors', default='CriteriaDetector', help='使用する検出器の名前（カンマ区切りで複数指定可能）')
    detector_group.add_argument('--detector-params', help='検出器のパラメータをJSON形式で指定 (例: \'{"PZSTDDetector": {"threshold": 0.5}, "ONDEDetector": {"paramX": 10}}\')。config設定を上書きします。')
    
    # 評価関連の引数
    eval_group = parser.add_argument_group('評価')
    eval_group.add_argument('--config', help='設定ファイルのパス（YAML形式）')
    
    # 出力関連の引数
    output_group = parser.add_argument_group('出力')
    output_group.add_argument('--output-dir', help='出力ディレクトリ')
    output_group.add_argument('--save-plots', action='store_true', help='結果のプロットを保存する')
    output_group.add_argument('--plot-format', default='png', help='プロットの形式（デフォルト: png）')
    output_group.add_argument('--save-results-json', action='store_true', default=True, help='JSON形式で結果を保存する（デフォルト: True）')
    
    # その他の引数
    other_group = parser.add_argument_group('その他')
    other_group.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='ログレベル')
    
    return parser.parse_args()


def setup_logging(log_level: str) -> logging.Logger:
    """
    ロギングを設定します。
    
    Parameters
    ----------
    log_level : str
        ログレベル
        
    Returns
    -------
    logging.Logger
        設定されたロガー
    """
    # ログレベルの設定
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # ロガーの設定
    logging.basicConfig(
        level=numeric_level, # ルートロガーのレベルを設定 (ただし既に設定されている場合は効果がない可能性あり)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # PZSTDDetector が使用するロガーを取得し、レベルを明示的に設定
    detector_logger = logging.getLogger('src.detectors.pzstd_detector')
    detector_logger.setLevel(numeric_level)
    # 念のため、他の主要なロガーのレベルも設定 (basicConfigがルートを設定していれば不要かもしれないが確実性のため)
    logging.getLogger('src').setLevel(numeric_level)

    # 以前と同様に 'evaluate' ロガーを返す
    main_logger = logging.getLogger('evaluate') 
    main_logger.setLevel(numeric_level) # 'evaluate' ロガーのレベルも設定
    return main_logger


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    設定ファイルをロードします。
    
    Parameters
    ----------
    config_path : Optional[str]
        設定ファイルのパス
        
    Returns
    -------
    Dict[str, Any]
        設定辞書
    """
    config = {}
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"設定ファイルのロード中にエラーが発生しました: {str(e)}")
    
    return config


def find_matching_files(directory: str, pattern: str) -> List[str]:
    """
    ディレクトリ内でパターンにマッチするファイルを検索します。
    
    Parameters
    ----------
    directory : str
        ディレクトリのパス
    pattern : str
        ファイルパターン
        
    Returns
    -------
    List[str]
        マッチするファイルパスのリスト
    """
    # パスの正規化
    if not os.path.isdir(directory):
        logging.error(f"ディレクトリが存在しません: {directory}")
        return []
    
    # ファイルの検索
    path = Path(directory)
    files = sorted(path.glob(pattern))
    
    return [str(f) for f in files]


def match_audio_and_reference_files(audio_files: List[str], ref_files: List[str], 
                                    audio_pattern: str = "*.wav", ref_pattern: str = "*.json") -> List[Tuple[str, str]]:
    """
    音声ファイルと参照ファイルをマッチングします。
    
    Parameters
    ----------
    audio_files : List[str]
        音声ファイルのリスト
    ref_files : List[str]
        参照ファイルのリスト
    audio_pattern : str, optional
        音声ファイルのパターン, by default "*.wav"
    ref_pattern : str, optional
        参照ファイルのパターン, by default "*.json"
        
    Returns
    -------
    List[Tuple[str, str]]
        マッチングした(音声ファイル, 参照ファイル)のタプルリスト
    """
    pairs = []
    
    # ファイル名のマッピングを作成
    audio_basename_map = {}
    for audio_file in audio_files:
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        audio_basename_map[basename] = audio_file
    
    ref_basename_map = {}
    for ref_file in ref_files:
        basename = os.path.splitext(os.path.basename(ref_file))[0]
        ref_basename_map[basename] = ref_file
    
    # 共通のファイル名を探して対応付け
    common_basenames = set(audio_basename_map.keys()) & set(ref_basename_map.keys())
    
    for basename in common_basenames:
        pairs.append((audio_basename_map[basename], ref_basename_map[basename]))
    
    return pairs


def prepare_detector_params(args, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    検出器のパラメータを準備します。
    
    優先順位:
    1. CLIで指定されたパラメータ (--detector-params)
    2. config.yamlで指定されたパラメータ
    3. 検出器のデフォルトパラメータ
    
    注意: 旧形式（'detectors'リスト）のconfigは非推奨です。
    
    Parameters
    ----------
    args : argparse.Namespace
        コマンドライン引数
    config : Dict[str, Any]
        設定辞書
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        検出器名をキーとするパラメータの辞書
    """
    detector_params = {}
    detector_names = args.detectors.split(',')
    
    # 1. まず設定ファイルから検出器パラメータを取得
    if config:
        # 新形式: 'detector_name'と'detector_params'
        if 'detector_name' in config and 'detector_params' in config:
            detector_name = config['detector_name']
            if detector_name in detector_names:
                detector_params[detector_name] = config['detector_params']
                logging.info(f"検出器 '{detector_name}' のパラメータをconfig.yamlから読み込みました。")
    
    # 2. 次にCLI引数からのパラメータを取得（こちらが優先）
    if args.detector_params:
        try:
            cmd_params_all = json.loads(args.detector_params)
            if not isinstance(cmd_params_all, dict):
                raise ValueError("JSONはオブジェクト形式である必要があります (例: {\"DetectorName\": {...}})")

            requested_detectors = args.detectors.split(',')

            for det_name_cli, params_cli in cmd_params_all.items():
                if det_name_cli in requested_detectors: # CLIで指定された検出器が実行対象か確認
                    if det_name_cli not in detector_params:
                        detector_params[det_name_cli] = {}
                    # CLIで指定されたパラメータは、configファイルの設定を上書き・マージします。
                    # CLIで指定されなかったパラメータはconfigファイルの値が維持されます。
                    detector_params[det_name_cli].update(params_cli) 
                    logging.info(f"検出器 '{det_name_cli}' のパラメータがコマンドライン引数で更新されました。")
                else:
                    logging.warning(f"CLIパラメータで指定された検出器 '{det_name_cli}' は --detectors 引数に含まれていません。無視されます。")
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"検出器パラメータのJSONパースまたは形式エラー: {args.detector_params}, Error: {e}")
    
    return detector_params


def prepare_evaluator_config(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    評価設定を準備します。MIREX準拠のオフセット評価を行います。
    
    注意: 旧形式（'evaluation'キー）のconfigは非推奨です。
    
    Parameters
    ----------
    args : argparse.Namespace
        コマンドライン引数
    config : Dict[str, Any]
        設定辞書
        
    Returns
    -------
    Dict[str, Any]
        評価設定辞書
    """
    evaluator_config = {
        'tolerance_onset': 0.05,  # デフォルト値: 50ms (MIREX標準)
        'offset_ratio': 0.2,      # デフォルト値: 0.2 (MIREX標準)
        'offset_min_tolerance': 0.05, # デフォルト値: 50ms (MIREX標準)
        'tolerance_pitch': 50.0,  # デフォルト値: 50セント (MIREX標準)
        'use_offset': True,      # 常にオフセット評価を行う
        'use_pitch_chroma': False
    }
    
    # 設定ファイルからの値を適用
    if config:
        # 新形式: 'evaluator_config'
        if 'evaluator_config' in config:
            eval_config = config['evaluator_config']
            
            # onset_tolerance
            if 'onset_tolerance' in eval_config:
                evaluator_config['tolerance_onset'] = eval_config['onset_tolerance']
            
            # offset_tolerance
            if 'offset_tolerance' in eval_config:
                evaluator_config['offset_min_tolerance'] = eval_config['offset_tolerance']
            
            # pitch_tolerance
            if 'pitch_tolerance' in eval_config:
                evaluator_config['tolerance_pitch'] = eval_config['pitch_tolerance']
            
            # その他の評価設定
            if 'metrics_to_compute' in eval_config:
                evaluator_config['metrics'] = eval_config['metrics_to_compute']
    
    return evaluator_config


def main():
    """
    メイン実行関数
    """
    # コマンドライン引数の解析
    args = parse_args()
    
    # ロギングの設定
    logger = setup_logging(args.log_level)
    logger.info("評価を開始します...")
    
    # 設定ファイルの読み込み
    config = load_config(args.config)
    
    # 出力ディレクトリの作成
    output_dir = args.output_dir or "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 音声ファイルと参照ファイルの準備
    audio_paths = []
    ref_paths = []
    
    # 単一ファイルの場合
    if args.audio and args.reference:
        if os.path.exists(args.audio) and os.path.exists(args.reference):
            audio_paths = [args.audio]
            ref_paths = [args.reference]
        else:
            logger.error("指定されたファイルが存在しません")
            return 1
    
    # ディレクトリから複数ファイルの場合
    elif args.audio_dir and args.reference_dir:
        if not os.path.exists(args.audio_dir) or not os.path.exists(args.reference_dir):
            logger.error("指定されたディレクトリが存在しません")
            return 1
        
        # ファイルのマッチング
        audio_files = find_matching_files(args.audio_dir, args.audio_pattern)
        ref_files = find_matching_files(args.reference_dir, args.reference_pattern)
        
        # ファイルのペアを作成
        file_pairs = match_audio_and_reference_files(
            audio_files, ref_files, 
            args.audio_pattern, args.reference_pattern
        )
        
        if not file_pairs:
            logger.error("評価可能なファイルペアが見つかりませんでした")
            return 1
        
        # リストに追加
        for audio_file, ref_file in file_pairs:
            audio_paths.append(audio_file)
            ref_paths.append(ref_file)
    
    else:
        logger.error("評価対象のファイル（--audio/--reference または --audio-dir/--reference-dir）を指定してください。")
        return 1
    
    # 検出器のパラメータを準備
    all_detector_params = prepare_detector_params(args, config)
    
    # 評価設定の準備
    evaluator_config = prepare_evaluator_config(args, config)
    
    # 評価の実行
    logger.info(f"{len(audio_paths)}個のファイルと{len(all_detector_params)}個の検出器で評価を実行します")
    
    # 出力関連のパラメータをまとめる
    output_params = {
        "output_dir": args.output_dir or "evaluation_results",
        "save_plots": args.save_plots,
        "plot_format": args.plot_format,
    }
    
    try:
        # run_evaluationの呼び出し
        results = run_evaluation(
            audio_paths=audio_paths,
            ref_paths=ref_paths,
            detector_names=args.detectors.split(','),
            detector_params=all_detector_params,
            evaluator_config=evaluator_config,
            **output_params
        )
        
        # 結果の処理
        if results["status"] == "success":
            logger.info("評価が正常に完了しました")
            # 評価結果のサマリを表示
            if results["status"] in ["success", "partial_success"]:
                summary_df = create_summary_dataframe([results])
                print_summary_statistics(summary_df, logger)
            
            # 結果の保存先を表示
            if args.save_results_json:
                logger.info(f"詳細な結果は {output_dir} ディレクトリに保存されました")
            
            # 結果を一時ファイルに書き込む（run_evaluate.pyから読み取るため）
            result_file_path = os.environ.get('EVALUATE_RESULT_FILE')
            if result_file_path:
                try:
                    # 出力ディレクトリも含める
                    results['output_dir'] = output_dir
                    
                    with open(result_file_path, 'w') as f:
                        json.dump(results, f)
                except Exception as e:
                    logger.error(f"結果ファイルの書き込みに失敗しました: {str(e)}")
        else:
            logger.error(f"評価中にエラーが発生しました: {results.get('error', '不明なエラー')}")
            return 1
            
    except Exception as e:
        logger.error(f"評価実行中に例外が発生しました: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 