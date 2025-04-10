"""
評価のコア処理を統合したモジュール

このモジュールは、オンセット/オフセット/ピッチなどの音楽的特徴の検出評価に
関連する主要な処理を統合します。複数の場所に散在していた評価ロジックを一元化します。

## 評価関連の機能について

フレームベース評価関連の機能は `src.evaluation.evaluation_frame` モジュールを使用してください。
評価実行機能は `src.evaluation.evaluation_runner` モジュールの `run_evaluation` 関数を使用してください。

このモジュールは以前の評価関連機能を集約していましたが、現在は各責務に応じたモジュールに
機能が移行されています。新しい実装では、各評価機能を専門のモジュールから直接インポートして
使用してください。

【重要】: 評価実行はsrc.evaluation.evaluation_runnerモジュール内のrun_evaluation関数を使用してください。
"""

import os
import glob
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import mir_eval
import pandas as pd
import time
import warnings
from typing import Dict, Any, List, Union, Tuple, Optional, Callable
from pathlib import Path
import datetime
import librosa
import traceback
import inspect
from collections import defaultdict

from tqdm import tqdm

from src.utils.audio_utils import load_audio_file, load_reference_data, load_audio_and_reference, make_output_path
from src.utils.exception_utils import log_exception, create_error_result
from src.utils.json_utils import NumpyEncoder
from src.utils.detection_result import DetectionResult
from src.utils.pitch_utils import hz_to_midi, midi_to_hz
from src.visualization.plots import plot_detection_results
from src.utils.detector_utils import get_detector_class
from src.evaluation.evaluation_io import save_evaluation_result

if __name__ == '__main__':
    import argparse
    import sys
    
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='音楽ノート検出の評価を実行')
    parser.add_argument('audio_path', help='評価対象の音声ファイルのパス')
    parser.add_argument('ref_path', help='参照ラベルファイルのパス')
    parser.add_argument('--detector', default='BasicDetector', help='使用する検出器の名前')
    parser.add_argument('--output-dir', help='出力ディレクトリ')
    parser.add_argument('--save-plots', action='store_true', help='プロットを保存するかどうか')
    parser.add_argument('--plot-format', default='png', help='プロットの形式')
    parser.add_argument('--log-level', default='INFO', help='ログレベル')
    
    args = parser.parse_args()
    
    # ログレベルの設定
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 評価の実行
        from src.evaluation.evaluation_runner import run_evaluation
        result = run_evaluation(
            audio_paths=args.audio_path,
            ref_paths=args.ref_path,
            detector_names=args.detector,
            output_dir=args.output_dir,
            save_plots=args.save_plots,
            plot_format=args.plot_format
        )
        
        # 結果の出力
        if result['status'] == 'success' and result['results']:
            print("\n評価結果:")
            print(f"検出器: {result['results'][0]['detector_name']}")
            print(f"音声ファイル: {result['results'][0]['audio_path']}")
            print(f"参照ファイル: {result['results'][0]['ref_path']}")
            print(f"検出時間: {result['results'][0]['detection_time']:.3f}秒")
            print("\n評価指標:")
            for key, value in result['results'][0]['evaluation'].items():
                if isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
        elif result['status'] == 'error':
            print(f"エラー: {result['error']}")
        else:
            print("評価結果が正しく取得できませんでした。")
    except Exception as e:
        print(f"評価実行中にエラーが発生しました: {str(e)}")
        traceback.print_exc()
        sys.exit(1)