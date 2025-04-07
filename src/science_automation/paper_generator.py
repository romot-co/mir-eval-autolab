"""
論文ドラフト生成モジュール

このモジュールは改善結果から学術論文ドラフトを自動生成します。
改善セッションのデータを分析し、研究内容を論文形式にまとめます。
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

class PaperGenerator:
    """改善結果から学術論文ドラフトを自動生成"""
    
    def __init__(self, template_path: str = None):
        """
        パラメータ:
            template_path: 論文テンプレートのパス
        """
        self.template_path = template_path
        self.template = self._load_template()
        self.mcp_server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:5002")
    
    def _load_template(self) -> str:
        """論文テンプレートを読み込む"""
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
            "paper_template.md"
        )
        
        if os.path.exists(default_template_path):
            try:
                with open(default_template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"デフォルトテンプレートの読み込みに失敗しました: {e}")
        
        # 組み込みテンプレート
        return """# {title}

## 概要
{abstract}

## 1. はじめに
{introduction}

## 2. 関連研究
{related_work}

## 3. 方法論
{methodology}

## 4. 実験
{experiments}

## 5. 結果
{results}

## 6. 考察
{discussion}

## 7. 結論
{conclusion}

## 参考文献
{references}
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
    
    def generate_paper(self, 
                      session_id: str, 
                      session_data: Dict[str, Any], 
                      best_version_info: Dict[str, Any],
                      hypothesis_data: List[Dict[str, str]],
                      output_path: Optional[str] = None) -> str:
        """
        セッションデータから学術論文ドラフトを生成
        
        パラメータ:
            session_id: セッションID
            session_data: セッション全体のデータ
            best_version_info: 最良バージョンの情報
            hypothesis_data: 生成された仮説データ
            output_path: 出力ファイルパス
            
        戻り値:
            生成された論文ドラフトのパス
        """
        # 改善履歴をセクションごとに構造化
        improvement_history = self._organize_improvement_history(session_data)
        
        # ベストバージョンのコードと結果
        best_version = best_version_info.get('tag', 'Unknown')
        best_f_measure = best_version_info.get('f_measure', 0)
        
        # 仮説情報を整形
        hypotheses_text = "\n\n".join([
            f"**仮説 {i+1}: {h['title']}**\n\n{h['description']}\n\n" + 
            f"*根拠*: {h['evidence']}\n\n*検証可能性*: {h['testability']}"
            for i, h in enumerate(hypothesis_data)
        ])
        
        # LLMに論文ドラフト生成を依頼
        prompt = f"""あなたは音楽情報検索（MIR）の研究者です。以下のアルゴリズム改善実験に基づいて研究論文を執筆してください：

セッションID: {session_id}
基本アルゴリズム: {session_data.get('base_algorithm', '不明')}
最良バージョン: {best_version}
最良F値: {best_f_measure}

改善履歴:
{json.dumps(improvement_history, indent=2, ensure_ascii=False)}

主要な仮説:
{hypotheses_text}

以下のセクションを含む完全な学術論文ドラフトを作成してください：
1. タイトル: 記述的で具体的なタイトル
2. 概要: 研究、方法論、主な発見を要約
3. はじめに: 背景、動機、研究課題を提供
4. 関連研究: 関連するMIR文献の中での位置づけ
5. 方法論: アルゴリズム改善プロセスと技術の説明
6. 実験: 評価データセットとメトリクスの説明
7. 結果: 定量的結果と改善点の提示
8. 考察: 発見、限界、意義の分析
9. 結論: 貢献と今後の研究のまとめ
10. 参考文献: 関連する引用文献

明快さ、科学的厳密さ、結果を広範なMIR研究に結びつけることに重点を置いてください。
Markdownフォーマットで作成してください。
"""

        # LLMリクエスト
        paper_content = self.request_llm(prompt)
        
        # 指定されたパスに保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path or f"papers/paper_{session_id}_{timestamp}.md"
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(paper_content)
        
        logger.info(f"論文ドラフトを保存しました: {output_file}")
        return output_file
    
    def _organize_improvement_history(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """セッションデータから改善履歴を抽出して構造化"""
        history = session_data.get('history', [])
        
        # イテレーションごとの改善内容を整理
        iterations = {}
        current_iteration = None
        
        for event in history:
            event_type = event.get('type')
            event_data = event.get('data', {})
            
            if event_type == 'iteration_start':
                current_iteration = event_data.get('iteration')
                iterations[current_iteration] = {'events': []}
            
            if current_iteration is not None:
                iterations[current_iteration]['events'].append({
                    'type': event_type,
                    'data': event_data
                })
                
                # 評価結果を保存
                if event_type == 'evaluation_complete':
                    iterations[current_iteration]['metrics'] = event_data.get('metrics_preview', '{}')
                    
                # LLM改善を保存
                elif event_type == 'llm_improve_complete':
                    iterations[current_iteration]['improvement'] = event_data.get('result_preview', '')
        
        return iterations 