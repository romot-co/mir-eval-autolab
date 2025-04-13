"""
検出アルゴリズムの評価機能を提供するモジュール

このモジュールでは、音楽検出アルゴリズムのパフォーマンス評価に関する機能を集約しています。
主な機能:
- 単一ファイル評価
- 複数ファイル評価
- オンセット、オフセット、ピッチ検出の評価
- 評価結果の可視化
- グリッドサーチによるパラメータ最適化
"""

# 評価関連モジュールのパッケージ

# 評価関数モジュール
from src.evaluation.evaluation_frame import evaluate_frame_pitches

# 評価I/O関数
from src.evaluation.evaluation_io import load_evaluation_result, save_evaluation_result

# 評価実行関数
from src.evaluation.evaluation_runner import evaluate_detection_result, run_evaluation

# グリッドサーチ機能
from src.evaluation.grid_search.core import (
    _save_results_to_csv,
    create_grid_config,
    run_grid_search,
)
