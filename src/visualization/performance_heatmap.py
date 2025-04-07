"""
性能ヒートマップ生成モジュール

このモジュールはパラメータ空間における性能分布のヒートマップを可視化します。
グリッドサーチ結果から、異なるパラメータ設定に対する性能メトリックをヒートマップとして表示します。
"""

import numpy as np
import pandas as pd
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    # seabornが利用できない場合はフォールバックソリューション
    SEABORN_AVAILABLE = False
    print("警告: seabornライブラリが見つかりません。pip install seabornでインストールしてください。")
    print("基本的なmatplotlibを使用します。")
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging

# ロギングの設定
logger = logging.getLogger(__name__)

class PerformanceHeatmap:
    """パラメータ空間における性能分布のヒートマップ可視化"""
    
    def __init__(self, grid_search_results: List[Dict[str, Any]]):
        """
        パラメータ:
            grid_search_results: グリッドサーチの結果リスト
              各要素は {'params': {...}, 'metrics': {...}} 形式
        """
        self.results = grid_search_results
        self.metric_keys = self._extract_available_metrics()
    
    def _extract_available_metrics(self) -> List[str]:
        """結果から利用可能なメトリックのリストを抽出"""
        metrics = []
        if self.results and 'metrics' in self.results[0]:
            sample_metrics = self.results[0]['metrics']
            # ネストされたメトリックを平坦化 (note.f_measure → note_f_measure)
            for category, values in sample_metrics.items():
                if isinstance(values, dict):
                    for metric_name in values:
                        metrics.append(f"{category}_{metric_name}")
                else:
                    metrics.append(category)
        return metrics
    
    def create_heatmap(self, param_x: str, param_y: str, metric: str, 
                      output_path: Optional[str] = None,
                      title: Optional[str] = None) -> str:
        """
        2つのパラメータに対する性能メトリックのヒートマップを作成
        
        パラメータ:
            param_x: X軸のパラメータ名
            param_y: Y軸のパラメータ名
            metric: 可視化するメトリック (例: "note_f_measure")
            output_path: 出力ファイルパス (省略時は表示のみ)
            title: グラフタイトル
        """
        # 結果からデータフレームを作成
        data = []
        for result in self.results:
            if 'params' not in result or param_x not in result['params'] or param_y not in result['params']:
                continue
                
            # メトリック値の取得 (カテゴリ_メトリック形式を解析)
            metric_value = None
            if '_' in metric:
                category, metric_name = metric.split('_', 1)
                if ('metrics' in result and category in result['metrics'] and 
                    isinstance(result['metrics'][category], dict) and 
                    metric_name in result['metrics'][category]):
                    metric_value = result['metrics'][category][metric_name]
            else:
                if 'metrics' in result and metric in result['metrics']:
                    metric_value = result['metrics'][metric]
                    
            if metric_value is not None:
                data.append({
                    param_x: result['params'][param_x],
                    param_y: result['params'][param_y],
                    'metric': metric_value
                })
        
        if not data:
            error_msg = f"指定されたパラメータ({param_x}, {param_y})またはメトリック({metric})のデータが見つかりません"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        df = pd.DataFrame(data)
        
        # ピボットテーブルを作成
        pivot_table = df.pivot(index=param_y, columns=param_x, values='metric')
        
        # ヒートマップ生成
        plt.figure(figsize=(12, 9))
        
        if SEABORN_AVAILABLE:
            # seabornが利用可能な場合
            sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f', 
                      linewidths=.5, cbar_kws={'label': metric})
        else:
            # matplotlib単体で代替実装
            ax = plt.gca()
            im = ax.imshow(pivot_table.values, cmap='viridis', interpolation='nearest')
            
            # 軸ラベルの設定
            ax.set_xticks(np.arange(len(pivot_table.columns)))
            ax.set_yticks(np.arange(len(pivot_table.index)))
            ax.set_xticklabels(pivot_table.columns)
            ax.set_yticklabels(pivot_table.index)
            
            # 値のアノテーション表示
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    ax.text(j, i, f"{pivot_table.values[i, j]:.3f}",
                            ha="center", va="center", color="w")
            
            # カラーバー
            cbar = plt.colorbar(im)
            cbar.set_label(metric)
        
        plt.title(title or f'Performance Heatmap: {metric}')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"ヒートマップを保存しました: {output_path}")
            return output_path
        else:
            plt.show()
            logger.info("ヒートマップを表示しました")
            return "Displayed heatmap"
    
    def create_multiple_heatmaps(self, 
                                param_x: str, 
                                param_y: str, 
                                metrics: List[str],
                                output_dir: str,
                                prefix: str = "heatmap") -> List[str]:
        """
        複数のメトリックに対するヒートマップをまとめて生成
        
        パラメータ:
            param_x: X軸のパラメータ名
            param_y: Y軸のパラメータ名
            metrics: 可視化するメトリックのリスト
            output_dir: 出力ディレクトリ
            prefix: 出力ファイル名の接頭辞
            
        戻り値:
            生成されたファイルパスのリスト
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = []
        for metric in metrics:
            try:
                filename = f"{prefix}_{metric}.png"
                output_path = os.path.join(output_dir, filename)
                
                self.create_heatmap(
                    param_x=param_x,
                    param_y=param_y,
                    metric=metric,
                    output_path=output_path,
                    title=f"Performance Heatmap: {metric}"
                )
                
                output_files.append(output_path)
                
            except Exception as e:
                logger.error(f"メトリック {metric} のヒートマップ生成に失敗しました: {e}")
        
        return output_files
    
    def get_available_metrics(self) -> List[str]:
        """利用可能なメトリックの一覧を取得"""
        return self.metric_keys
    
    def get_optimal_parameters(self, metric: str) -> Dict[str, Any]:
        """指定したメトリックに対する最適パラメータを取得"""
        best_value = None
        best_params = {}
        
        for result in self.results:
            # メトリック値の取得
            metric_value = None
            if '_' in metric:
                category, metric_name = metric.split('_', 1)
                if ('metrics' in result and category in result['metrics'] and 
                    isinstance(result['metrics'][category], dict) and 
                    metric_name in result['metrics'][category]):
                    metric_value = result['metrics'][category][metric_name]
            else:
                if 'metrics' in result and metric in result['metrics']:
                    metric_value = result['metrics'][metric]
            
            if metric_value is not None:
                if best_value is None or metric_value > best_value:
                    best_value = metric_value
                    best_params = result['params'].copy()
        
        return {
            'value': best_value,
            'params': best_params
        } 