"""
研究ノート生成モジュール

このモジュールは改善プロセスの結果を実験ノートとして自動生成します。
改善セッションの履歴データを分析し、実験の経過や結論を構造化します。
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# ロギングの設定
logger = logging.getLogger(__name__)

class ResearchNotebook:
    """改善プロセスの結果を実験ノートとして自動生成"""
    
    def __init__(self, template_path: str = None):
        """
        パラメータ:
            template_path: レポートテンプレートのパス
        """
        self.template_path = template_path
        self.template = self._load_template()
        self.mcp_server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:5002")
    
    def _load_template(self) -> str:
        """レポートテンプレートを読み込む"""
        if self.template_path and os.path.exists(self.template_path):
            try:
                with open(self.template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"テンプレートの読み込みに失敗しました: {e}")
        
        # デフォルトテンプレート
        default_template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "templates", 
            "report_template.md"
        )
        
        if os.path.exists(default_template_path):
            try:
                with open(default_template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"デフォルトテンプレートの読み込みに失敗しました: {e}")
        
        # 組み込みテンプレート
        return """# 実験ノート: {experiment_title}

## 実験概要
- **日付:** {date}
- **研究者:** 自動研究システム
- **対象:** {subject}
- **期間:** {duration}

## 目的
{objective}

## 実験設定
{setup}

## イテレーション
{iterations}

## 結果
{results}

## 分析
{analysis}

## 結論
{conclusion}

## 次のステップ
{next_steps}
"""
    
    def request_llm(self, prompt: str) -> str:
        """LLMにリクエストを送信"""
        try:
            # MCPサーバーに直接リクエスト
            url = f"{self.mcp_server_url}/api/llm/generate"
            response = requests.post(url, json={"prompt": prompt}, timeout=120)
            response.raise_for_status()
            return response.json().get("text", "")
        except requests.RequestException as e:
            logger.error(f"LLMリクエスト中にエラーが発生しました: {e}")
            # フォールバック：直接の応答生成
            return f"LLMへのリクエストに失敗しました: {e}"
    
    def generate_notebook(self, 
                         session_id: str, 
                         session_data: Dict[str, Any],
                         output_path: Optional[str] = None) -> str:
        """
        セッションデータから実験ノートを生成
        
        パラメータ:
            session_id: セッションID
            session_data: セッション全体のデータ
            output_path: 出力ファイルパス
            
        戻り値:
            生成された実験ノートのパス
        """
        # 全ての履歴イベントを時系列で取得
        history = session_data.get('history', [])
        
        # 改善サイクルの基本情報
        base_algorithm = session_data.get('base_algorithm', '不明')
        start_time_timestamp = session_data.get('start_time', 0)
        start_time = datetime.fromtimestamp(start_time_timestamp) if start_time_timestamp else datetime.now()
        
        # イテレーション情報の抽出
        iterations_data = self._organize_iterations(history)
        
        # 成功と失敗の両方をカウント
        success_count = sum(1 for it in iterations_data.values() if it.get('improved', False))
        failure_count = sum(1 for it in iterations_data.values() if not it.get('improved', False))
        
        # LLMにレポート生成を依頼
        prompt = f"""以下のデータを持つアルゴリズム改善実験の詳細な研究ノートを生成してください：

セッションID: {session_id}
基本アルゴリズム: {base_algorithm}
開始時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
イテレーション数: {len(iterations_data)}
成功した改善: {success_count}
失敗した改善試行: {failure_count}

詳細なイテレーションデータ:
{json.dumps(iterations_data, indent=2, ensure_ascii=False)}

ノートブックには以下を含めてください：

1. 簡潔な実験概要
2. 改善プロセスの明確な目的
3. 詳細な実験設定
4. 各イテレーションの時系列的な記録（以下を含む）：
   - 初期条件
   - 加えられた変更
   - 評価結果
   - 成功または失敗の分析
   - **重要**: 成功と失敗の両方、すべてのイテレーションを含める
5. 全体的な結果分析
6. 改善プロセスからの知見とパターン
7. 主要な発見を含む結論
8. 今後の研究のための推奨ステップ

マークダウン形式で明確なセクションヘッダー、適切な表、簡潔な技術文章でノートブックをフォーマットしてください。技術的詳細について正確に、観察について分析的に記述してください。
"""

        # LLMリクエスト
        notebook_content = self.request_llm(prompt)
        
        # 指定されたパスに保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path or f"notebooks/experiment_{session_id}_{timestamp}.md"
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(notebook_content)
        
        logger.info(f"研究ノートを保存しました: {output_file}")
        return output_file
    
    def _organize_iterations(self, history: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """履歴イベントからイテレーション情報を抽出して構造化"""
        iterations = {}
        current_iteration = None
        
        for event in history:
            event_type = event.get('type')
            event_data = event.get('data', {})
            timestamp = event.get('timestamp')
            
            # イテレーション開始イベントを検出
            if event_type == 'iteration_start':
                current_iteration = event_data.get('iteration')
                iterations[current_iteration] = {
                    'start_time': timestamp,
                    'events': [],
                    'improved': False
                }
            
            # イテレーションに関連するイベントを記録
            if current_iteration is not None:
                iterations[current_iteration]['events'].append({
                    'type': event_type,
                    'data': event_data,
                    'timestamp': timestamp
                })
                
                # 評価結果を保存
                if event_type == 'evaluation_complete':
                    iterations[current_iteration]['evaluation_result'] = event_data
                
                # 改善の有無を記録
                elif event_type == 'no_improvement':
                    iterations[current_iteration]['improved'] = False
                
                # 最良バージョンの更新は改善成功
                elif event_type == 'update_best_version':
                    iterations[current_iteration]['improved'] = True
                    iterations[current_iteration]['best_version'] = event_data
        
        return iterations
    
    def _extract_metrics_summary(self, iterations_data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """全イテレーションからメトリクスのサマリーを抽出"""
        metrics_summary = {
            'f_measure': [],
            'precision': [],
            'recall': []
        }
        
        for iter_num, iter_data in iterations_data.items():
            eval_result = iter_data.get('evaluation_result', {})
            metrics = eval_result.get('metrics_preview', '{}')
            
            # 文字列をJSONとして解析
            try:
                if isinstance(metrics, str):
                    metrics = metrics.replace("'", '"')
                    metrics_dict = json.loads(metrics)
                else:
                    metrics_dict = metrics
                
                # note.f_measure または note_f_measure を取得
                if 'note' in metrics_dict and isinstance(metrics_dict['note'], dict):
                    f_measure = metrics_dict['note'].get('f_measure')
                    precision = metrics_dict['note'].get('precision')
                    recall = metrics_dict['note'].get('recall')
                    
                    if f_measure is not None:
                        metrics_summary['f_measure'].append((iter_num, f_measure))
                    if precision is not None:
                        metrics_summary['precision'].append((iter_num, precision))
                    if recall is not None:
                        metrics_summary['recall'].append((iter_num, recall))
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"イテレーション {iter_num} のメトリクス解析に失敗しました: {e}")
        
        return metrics_summary 