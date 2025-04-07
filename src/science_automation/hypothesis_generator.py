"""
仮説生成モジュール

このモジュールは改善パターンから科学的仮説を自動生成します。
改善セッションの履歴データを分析し、新たなアルゴリズム設計原則の仮説を生成します。
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import json
import logging
import requests
import os

# ロギングの設定
logger = logging.getLogger(__name__)

class HypothesisGenerator:
    """改善パターンから新たなアルゴリズム設計原則の仮説を自動生成"""
    
    def __init__(self, session_history: List[Dict[str, Any]]):
        """
        パラメータ:
            session_history: 改善セッションの履歴データ
        """
        self.history = session_history
        self.improvements = self._extract_improvements()
        self.mcp_server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:5002")
    
    def _extract_improvements(self) -> List[Dict[str, Any]]:
        """履歴から成功した改善イテレーションを抽出"""
        successful_improvements = []
        
        current_best_f_measure = 0
        for event in self.history:
            if event.get('type') == 'evaluation_complete':
                metrics = event.get('data', {}).get('metrics_preview', '{}')
                try:
                    # 文字列をJSONとして解析（'を"に置換する必要がある場合がある）
                    if isinstance(metrics, str):
                        metrics = metrics.replace("'", '"')
                        metrics_dict = json.loads(metrics)
                    else:
                        metrics_dict = metrics
                    
                    # note.f_measure または note_f_measure を取得
                    f_measure = None
                    if 'note' in metrics_dict and isinstance(metrics_dict['note'], dict):
                        f_measure = metrics_dict['note'].get('f_measure', 0)
                    elif 'note_f_measure' in metrics_dict:
                        f_measure = metrics_dict['note_f_measure']
                    
                    if f_measure and f_measure > current_best_f_measure:
                        current_best_f_measure = f_measure
                        successful_improvements.append({
                            'metrics': metrics_dict,
                            'improvement': event.get('data', {}).get('improvement_summary', '')
                        })
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"メトリクスの解析に失敗しました: {e}")
                    
        return successful_improvements
    
    def request_llm(self, prompt: str) -> str:
        """LLMにリクエストを送信"""
        try:
            # MCPサーバーに直接リクエスト
            url = f"{self.mcp_server_url}/api/llm/generate"
            response = requests.post(url, json={"prompt": prompt}, timeout=60)
            response.raise_for_status()
            return response.json().get("text", "")
        except requests.RequestException as e:
            logger.error(f"LLMリクエスト中にエラーが発生しました: {e}")
            # フォールバック：直接の応答生成
            return f"LLMへのリクエストに失敗しました: {e}"
    
    def generate_hypotheses(self, num_hypotheses: int = 3) -> List[Dict[str, str]]:
        """
        パターンを分析して新たな仮説を生成
        
        パラメータ:
            num_hypotheses: 生成する仮説の数
            
        戻り値:
            仮説リスト [{title, description, evidence, testability}, ...]
        """
        if not self.improvements:
            logger.warning("成功した改善が見つかりません。仮説生成のためのデータが不足しています。")
            return [{
                "title": "改善データが不足しています", 
                "description": "仮説生成に必要な改善データが不足しています", 
                "evidence": "N/A", 
                "testability": "N/A"
            }]
        
        # 改善サマリーを集約
        improvement_texts = []
        for i, imp in enumerate(self.improvements):
            metrics_text = json.dumps(imp.get('metrics', {}), indent=2)
            improvement_summary = imp.get('improvement', '')
            improvement_texts.append(f"改善 {i+1}:\n{improvement_summary}\n\nメトリクス:\n{metrics_text}")
        
        improvement_context = "\n\n---\n\n".join(improvement_texts)
        
        # LLMに仮説生成を依頼
        prompt = f"""以下の成功したアルゴリズム改善に基づいて、音声処理アルゴリズムの設計原則に関する{num_hypotheses}つの科学的仮説を生成してください：

{improvement_context}

各仮説について、以下を提供してください：
1. タイトル：仮説の簡潔なタイトル
2. 説明：仮説の詳細な説明
3. 根拠：提供された改善から得られた裏付けとなる証拠
4. 検証可能性：この仮説がどのように実験的に検証できるか

各仮説をこれら4つのフィールドを持つJSONオブジェクトとしてフォーマットしてください。
{num_hypotheses}個の仮説オブジェクトの配列を返してください。
"""
        
        # LLMリクエスト
        response = self.request_llm(prompt)
        
        try:
            # 応答のパース
            hypotheses = json.loads(response)
            if isinstance(hypotheses, list) and len(hypotheses) > 0:
                logger.info(f"{len(hypotheses)}個の仮説を生成しました")
                return hypotheses
            else:
                logger.warning("仮説の応答形式が予期しないものでした")
                return [{
                    "title": "仮説の解析エラー", 
                    "description": "予期しない応答形式", 
                    "evidence": response[:100], 
                    "testability": "N/A"
                }]
        except json.JSONDecodeError:
            # テキスト応答からの緊急抽出を試みる
            logger.warning("JSON解析に失敗しました。テキスト応答から抽出を試みます。")
            hypotheses = []
            sections = response.split("仮説 ") if "仮説 " in response else response.split("Hypothesis ")
            
            for i, section in enumerate(sections[1:], 1):
                if i > num_hypotheses:
                    break
                    
                lines = section.split("\n")
                title = lines[0].strip().replace(":", "") if lines else f"仮説 {i}"
                
                # 説明、根拠、検証可能性を抽出
                description = []
                evidence = ""
                testability = ""
                
                evidence_found = False
                testability_found = False
                
                for line in lines[1:]:
                    if "根拠:" in line or "Evidence:" in line:
                        evidence_found = True
                        evidence = line.replace("根拠:", "").replace("Evidence:", "").strip()
                        continue
                    
                    if "検証可能性:" in line or "Testability:" in line:
                        testability_found = True
                        testability = line.replace("検証可能性:", "").replace("Testability:", "").strip()
                        continue
                    
                    if not evidence_found and not testability_found:
                        description.append(line)
                
                hypotheses.append({
                    "title": title,
                    "description": "\n".join(description).strip(),
                    "evidence": evidence,
                    "testability": testability
                })
            
            return hypotheses if hypotheses else [{
                "title": "仮説生成に失敗しました", 
                "description": "応答の解析に失敗しました", 
                "evidence": "N/A", 
                "testability": "N/A"
            }] 