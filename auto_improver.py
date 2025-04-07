# auto_improver.py

import requests
import time
import os
import sys
import json
import logging
import uuid
import inspect
from typing import Dict, Any, Optional, List
from pathlib import Path
import platform

# Add project root to sys.path to allow importing detectors
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Attempt to import detectors dynamically (optional, for fetching code)
try:
    from src.detectors import DETECTOR_REGISTRY
except ImportError as e:
    print(f"Warning: Could not import DETECTOR_REGISTRY from src.detectors. ({e})")
    DETECTOR_REGISTRY = {}

# MCPクライアントライブラリ
import mcp.client

# 環境変数からの設定取得
DEFAULT_MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:5002")
DEFAULT_POLL_INTERVAL = int(os.environ.get("MCP_POLL_INTERVAL", "5"))  # seconds
DEFAULT_JOB_TIMEOUT = int(os.environ.get("MCP_JOB_TIMEOUT", "600"))  # seconds (10 minutes for individual jobs)
DEFAULT_SESSION_TIMEOUT = int(os.environ.get("MCP_SESSION_TIMEOUT", "3600"))  # seconds (1 hour for the entire session)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('auto_improver')

class AutoImprover:
    """MCP対応サーバーを利用して、アルゴリズム改善プロセスを調整します。"""

    def __init__(self, config_path=None, server_url=None):
        # 設定ファイルの読み込み
        self.config = self._load_config(config_path)
        
        # サーバーURLの設定（優先順位: 引数 > 環境変数 > 設定ファイル > デフォルト）
        self.server_url = server_url or os.environ.get("MCP_SERVER_URL") or self.config.get('server_url', DEFAULT_MCP_SERVER_URL)
        
        # 改善サイクルの設定
        self.max_iterations = int(os.environ.get("MCP_MAX_ITERATIONS", self.config.get('max_iterations', 5)))
        self.improvement_threshold = float(os.environ.get("MCP_IMPROVEMENT_THRESHOLD", self.config.get('improvement_threshold', 0.05)))
        self.grid_search_enabled = os.environ.get("MCP_GRID_SEARCH_ENABLED", "").lower() in ("true", "1", "yes") \
                             if "MCP_GRID_SEARCH_ENABLED" in os.environ \
                             else self.config.get('grid_search_enabled', True)
        
        # グリッドサーチパラメータ
        self.grid_search_params = self.config.get('grid_search_params', {
            'params_to_search': ['f0_score_threshold', 'HCF_ONSET_PEAK_THRESH'],
            'value_ranges': {
                'f0_score_threshold': [0.2, 0.25, 0.3, 0.35, 0.4],
                'HCF_ONSET_PEAK_THRESH': [0.1, 0.15, 0.2, 0.25, 0.3]
            }
        })
        
        # セッション情報
        self.session_id = None
        self.session_data = None
        self.current_detector_name = None
        self.current_detector_code = None
        self.last_evaluation_results = None
        
        # ログレベルの設定
        log_level = os.environ.get("MCP_LOG_LEVEL", self.config.get('log_level', 'INFO'))
        logging.getLogger('auto_improver').setLevel(getattr(logging, log_level.upper()))
        
        # MCPクライアントの初期化
        self.client = mcp.client.Client(self.server_url)
        logger.info(f"MCPクライアントを初期化しました（サーバーURL: {self.server_url}）")

    def _load_config(self, config_path):
        """
        設定ファイルを読み込み、適切な設定を返します。
        環境変数を考慮し、プラットフォーム固有の問題に対応します。
        
        Returns:
            dict: 設定情報を含む辞書
        """
        config = {}
        
        # 設定ファイルのパス
        config_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'),
            os.path.join(os.path.expanduser('~'), '.mirex', 'config.json')
        ]
        
        # 環境変数から追加の設定ファイルパスを取得
        env_config_path = os.environ.get('MIREX_CONFIG_PATH')
        if env_config_path:
            # パスを正規化（プラットフォーム固有の区切り文字を処理）
            norm_path = os.path.normpath(env_config_path)
            if os.path.isfile(norm_path):
                config_paths.append(norm_path)
            else:
                logger.warning(f"環境変数MIREX_CONFIG_PATHで指定されたファイルが見つかりません: {norm_path}")
        
        # 設定ファイルを探索
        loaded_config_path = None
        for path in config_paths:
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        loaded_config = json.load(f)
                        if isinstance(loaded_config, dict):
                            config.update(loaded_config)
                            loaded_config_path = path
                            logger.info(f"設定ファイルを読み込みました: {path}")
                        else:
                            logger.error(f"設定ファイルは辞書形式でなければなりません: {path}")
                except json.JSONDecodeError:
                    logger.error(f"設定ファイルの解析に失敗しました: {path}")
                except Exception as e:
                    logger.error(f"設定ファイルの読み込み中にエラーが発生しました: {path}: {e}")
        
        if loaded_config_path is None:
            logger.warning("設定ファイルが見つかりませんでした。デフォルト設定を使用します。")
        
        # 環境変数から直接設定を上書き
        for key in ['MCP_SERVER_URL', 'LOG_LEVEL', 'MAX_IMPROVEMENT_CYCLES']:
            env_var = os.environ.get(f'MIREX_{key}')
            if env_var:
                # 型変換を試みる
                if key == 'MAX_IMPROVEMENT_CYCLES':
                    try:
                        config[key.lower()] = int(env_var)
                    except ValueError:
                        logger.error(f"環境変数 MIREX_{key} の値 '{env_var}' を整数に変換できません")
                else:
                    config[key.lower()] = env_var
                logger.info(f"環境変数から設定を上書き: {key.lower()}")
        
        # 必須設定の検証
        if 'mcp_server_url' not in config or not config['mcp_server_url']:
            # デフォルトのMCP_SERVER_URL
            config['mcp_server_url'] = "http://localhost:5002"
            logger.info(f"MCP_SERVER_URLが設定されていません。デフォルト値を使用します: {config['mcp_server_url']}")
        
        # プラットフォーム固有の設定を適用
        platform_name = platform.system().lower()
        if f"{platform_name}_settings" in config:
            platform_settings = config[f"{platform_name}_settings"]
            if isinstance(platform_settings, dict):
                # プラットフォーム固有の設定を上位レベルにマージ
                for key, value in platform_settings.items():
                    if key not in config:
                        config[key] = value
                        logger.debug(f"プラットフォーム固有の設定を適用: {key}")
        
        return config

    # --- MCP Client Interaction --- 
    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """MCPツールを呼び出すヘルパー関数"""
        try:
            logger.debug(f"ツール呼び出し: {tool_name}, 引数: {kwargs}")
            result = await self.client.call_tool(tool_name, **kwargs)
            logger.debug(f"ツール呼び出し結果: {tool_name}")
            return result
        except Exception as e:
            logger.error(f"ツール呼び出しエラー: {tool_name} - {e}")
            raise

    async def poll_job_status(self, job_id: str, timeout: int = DEFAULT_JOB_TIMEOUT) -> Dict[str, Any]:
        """非同期ジョブのステータスをポーリングし、完了したら結果を返します。"""
        start_time = time.time()
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                logger.error(f"ジョブ {job_id} がタイムアウトしました（{timeout}秒）")
                if self.session_id:
                    try:
                        await self.call_tool("add_session_history", 
                                      session_id=self.session_id, 
                                      event_type="job_timeout", 
                                      event_data=json.dumps({"job_id": job_id, "timeout_duration": timeout}))
                    except Exception as hist_err:
                        logger.warning(f"セッション履歴へのジョブタイムアウト記録に失敗しました: {hist_err}")
                raise TimeoutError(f"ジョブ {job_id} がタイムアウトしました（{timeout}秒）")

            try:
                logger.info(f"ジョブ {job_id} のステータス確認中...")
                job_status = await self.call_tool("get_job_status", job_id=job_id)
                status = job_status.get("status")

                if status == 'completed':
                    logger.info(f"ジョブ {job_id} が正常に完了しました")
                    return job_status
                elif status == 'failed':
                    error_msg = job_status.get("error", "不明なエラー")
                    logger.error(f"ジョブ {job_id} が失敗しました: {error_msg}")
                    if self.session_id:
                         try:
                            await self.call_tool("add_session_history", 
                                          session_id=self.session_id, 
                                          event_type="job_failed", 
                                          event_data=json.dumps({"job_id": job_id, "error": error_msg}))
                         except Exception as hist_err:
                            logger.warning(f"セッション履歴へのジョブ失敗記録に失敗しました: {hist_err}")
                    raise RuntimeError(f"サーバーでジョブ {job_id} が失敗しました: {error_msg}")
                elif status == 'pending' or status == 'running':
                    logger.info(f"ジョブ {job_id} のステータス: {status}。待機中...")
                    time.sleep(DEFAULT_POLL_INTERVAL)
                else:
                    logger.warning(f"予期しないジョブステータス '{status}' (ジョブ {job_id})。再試行します...")
                    time.sleep(DEFAULT_POLL_INTERVAL)

            except Exception as e:
                logger.error(f"ジョブ {job_id} のポーリング中にエラーが発生しました: {e}")
                time.sleep(DEFAULT_POLL_INTERVAL)

    async def add_history_event(self, event_type: str, data: Dict[str, Any]):
        """セッション履歴にイベントを追加します。"""
        if not self.session_id:
            logger.warning("アクティブなセッションがないため、履歴イベントを追加できません")
            return
        try:
            await self.call_tool("add_session_history", 
                          session_id=self.session_id, 
                          event_type=event_type, 
                          event_data=json.dumps(data))
            logger.info(f"セッション {self.session_id} に履歴イベント '{event_type}' を追加しました")
        except Exception as e:
            logger.error(f"セッション {self.session_id} への履歴イベント '{event_type}' の追加に失敗しました: {e}")

    async def start_improvement_session(self, base_detector_name: str) -> str:
        """MCPサーバーで新しい改善セッションを開始します。"""
        logger.info(f"検出器 {base_detector_name} の新しい改善セッションを開始します")
        try:
            self.session_data = await self.call_tool("start_session", base_algorithm=base_detector_name)
            self.session_id = self.session_data.get('id')
            self.current_detector_name = base_detector_name
            logger.info(f"セッション開始（ID: {self.session_id}）")
            return self.session_id
        except Exception as e:
            logger.error(f"セッション開始に失敗しました: {e}")
            raise

    async def get_code(self, detector_name: str) -> str:
        """検出器のソースコードを取得します。"""
        logger.info(f"検出器 {detector_name} のコードを取得します")
        try:
            source_code = await self.call_tool("get_code", detector_name=detector_name)
            self.current_detector_code = source_code
            self.current_detector_name = detector_name
            return source_code
        except Exception as e:
            logger.error(f"検出器コードの取得に失敗しました: {e}")
            raise

    async def save_code(self, detector_name: str, code: str, version: str = None) -> Dict[str, Any]:
        """改善されたコードを保存します。"""
        logger.info(f"検出器 {detector_name} の改善コードを保存します")
        try:
            if version is None:
                version = f"v{int(time.time())}"
            result = await self.call_tool("save_code", detector_name=detector_name, code=code, version=version)
            logger.info(f"コードを保存しました: {result.get('file_path')}")
            return result
        except Exception as e:
            logger.error(f"コードの保存に失敗しました: {e}")
            raise

    async def improve_code(self, prompt: str) -> Dict[str, Any]:
        """LLMを使用してコードを改善します。"""
        logger.info("コード改善リクエストを送信します")
        try:
            # 改善ジョブを開始
            job_result = await self.call_tool("improve_code", prompt=prompt, session_id=self.session_id)
            job_id = job_result.get("job_id")
            
            # ジョブが完了するまで待機
            result = await self.poll_job_status(job_id)
            
            if "result" in result:
                improved_code = result["result"]
                logger.info("コード改善が完了しました")
                return {"improved_code": improved_code}
            else:
                raise RuntimeError("コード改善結果が含まれていません")
        except Exception as e:
            logger.error(f"コード改善に失敗しました: {e}")
            raise

    async def run_evaluation(self, detector_name: str, audio_dir: str = None, ref_dir: str = None) -> Dict[str, Any]:
        """検出器の評価を実行します。"""
        logger.info(f"検出器 {detector_name} の評価を開始します")
        try:
            # 評価ジョブを開始
            job_result = await self.call_tool("run_evaluation", detector_name=detector_name, 
                                       audio_dir=audio_dir, reference_dir=ref_dir)
            job_id = job_result.get("job_id")
            
            # ジョブが完了するまで待機
            result = await self.poll_job_status(job_id)
            
            if "result" in result:
                evaluation_result = result["result"]
                logger.info(f"検出器 {detector_name} の評価が完了しました")
                self.last_evaluation_results = evaluation_result
                return evaluation_result
            else:
                raise RuntimeError("評価結果が含まれていません")
        except Exception as e:
            logger.error(f"評価実行に失敗しました: {e}")
            raise

    async def run_grid_search(self, detector_name: str, grid_params: Dict[str, List[Any]],
                        audio_dir: str = None, ref_dir: str = None) -> Dict[str, Any]:
        """パラメータグリッドサーチを実行します。"""
        logger.info(f"検出器 {detector_name} のグリッドサーチを開始します")
        try:
            # パラメータをJSON形式に変換
            grid_params_json = json.dumps(grid_params)
            
            # グリッドサーチジョブを開始
            job_result = await self.call_tool("run_grid_search", detector_name=detector_name,
                                       grid_params_json=grid_params_json,
                                       audio_dir=audio_dir, reference_dir=ref_dir)
            job_id = job_result.get("job_id")
            
            # ジョブが完了するまで待機
            result = await self.poll_job_status(job_id)
            
            if "result" in result:
                grid_search_result = result["result"]
                logger.info(f"検出器 {detector_name} のグリッドサーチが完了しました")
                return grid_search_result
            else:
                raise RuntimeError("グリッドサーチ結果が含まれていません")
        except Exception as e:
            logger.error(f"グリッドサーチ実行に失敗しました: {e}")
            raise

    async def analyze_code_segment(self, code_segment: str, question: str = "") -> Dict[str, Any]:
        """コードセグメントを分析します。"""
        logger.info("コード分析リクエストを送信します")
        try:
            # 分析ジョブを開始
            job_result = await self.call_tool("analyze_code_segment", code_segment=code_segment, question=question)
            job_id = job_result.get("job_id")
            
            # ジョブが完了するまで待機
            result = await self.poll_job_status(job_id)
            
            if "result" in result:
                analysis_result = result["result"]
                logger.info("コード分析が完了しました")
                return {"analysis": analysis_result}
            else:
                raise RuntimeError("分析結果が含まれていません")
        except Exception as e:
            logger.error(f"コード分析に失敗しました: {e}")
            raise

    async def suggest_parameters(self, source_code: str, context: str = "") -> Dict[str, Any]:
        """アルゴリズムコードからパラメータ調整範囲を提案します。"""
        logger.info("パラメータ提案リクエストを送信します")
        try:
            # 提案ジョブを開始
            job_result = await self.call_tool("suggest_parameters", source_code=source_code, context=context)
            job_id = job_result.get("job_id")
            
            # ジョブが完了するまで待機
            result = await self.poll_job_status(job_id)
            
            if "result" in result:
                suggestion_result = result["result"]
                logger.info("パラメータ提案が完了しました")
                return suggestion_result
            else:
                raise RuntimeError("提案結果が含まれていません")
        except Exception as e:
            logger.error(f"パラメータ提案に失敗しました: {e}")
            raise

    def _format_evaluation_results(self, results: Dict[str, Any]) -> str:
        """評価結果を読みやすい形式に整形します。"""
        if not results:
            return "評価結果がありません"
        
        formatted = []
        overall = results.get("overall_metrics", {})
        
        if "note" in overall:
            note_metrics = overall["note"]
            formatted.append("## 音符検出性能")
            formatted.append(f"- F値 (F-measure): {note_metrics.get('f_measure', 0):.4f}")
            formatted.append(f"- 適合率 (Precision): {note_metrics.get('precision', 0):.4f}")
            formatted.append(f"- 再現率 (Recall): {note_metrics.get('recall', 0):.4f}")
        
        if "onset" in overall:
            onset_metrics = overall["onset"]
            formatted.append("## オンセット検出性能")
            formatted.append(f"- F値 (F-measure): {onset_metrics.get('f_measure', 0):.4f}")
            formatted.append(f"- 適合率 (Precision): {onset_metrics.get('precision', 0):.4f}")
            formatted.append(f"- 再現率 (Recall): {onset_metrics.get('recall', 0):.4f}")
        
        if "pitch" in overall:
            pitch_metrics = overall["pitch"]
            formatted.append("## ピッチ検出性能")
            formatted.append(f"- Raw Pitch Accuracy: {pitch_metrics.get('rpa', 0):.4f}")
            formatted.append(f"- Raw Chroma Accuracy: {pitch_metrics.get('rca', 0):.4f}")
            formatted.append(f"- Chroma Score: {pitch_metrics.get('chroma_score', 0):.4f}")
        
        if "file_metrics" in results:
            file_metrics = results["file_metrics"]
            formatted.append("## ファイル別性能")
            for filename, metrics in file_metrics.items():
                formatted.append(f"\n### {filename}")
                if "note" in metrics:
                    note = metrics["note"]
                    formatted.append(f"- ノートF値: {note.get('f_measure', 0):.4f}")
                if "onset" in metrics:
                    onset = metrics["onset"]
                    formatted.append(f"- オンセットF値: {onset.get('f_measure', 0):.4f}")
        
        return "\n".join(formatted)

    async def run_improvement_cycle(self, detector_name, goal=None, iterations=None, session_id=None):
        """
        アルゴリズム改善サイクルを実行します。
        
        Args:
            detector_name: 改善対象の検出器名
            goal: 改善目標（説明文）
            iterations: 改善サイクルの最大繰り返し回数
            session_id: 既存のセッションID（省略時は新しいセッションを開始）
            
        Returns:
            Dict[str, Any]: 改善結果の情報
        """
        # パラメータの設定
        if iterations is None:
            iterations = self.max_iterations
        
        if not goal:
            goal = "全体的な性能向上、特にリバーブとノイズに対する耐性を改善する"
        
        # セッション開始
        if not session_id:
            self.session_id = await self.start_improvement_session(detector_name)
        else:
            self.session_id = session_id
            self.current_detector_name = detector_name
        
        logger.info(f"改善サイクルを開始します（検出器: {detector_name}, 最大{iterations}回）")
        
        # 初期コードの取得
        try:
            current_code = await self.get_code(detector_name)
            logger.info(f"検出器 {detector_name} のコードを取得しました（{len(current_code)}バイト）")
        except Exception as e:
            logger.error(f"初期コードの取得に失敗しました: {e}")
            return {"status": "failed", "error": f"初期コードの取得に失敗: {e}"}
        
        # ベースライン評価の実行
        try:
            base_eval = await self.run_evaluation(detector_name)
            base_f_measure = base_eval.get("overall_metrics", {}).get("note", {}).get("f_measure", 0)
            logger.info(f"ベースラインF値: {base_f_measure:.4f}")
            
            # 評価結果の整形
            formatted_eval = self._format_evaluation_results(base_eval)
            
            # セッション履歴に追加
            await self.add_history_event("baseline_evaluation", {
                "detector_name": detector_name,
                "f_measure": base_f_measure,
                "evaluation_result": base_eval
            })
        except Exception as e:
            logger.error(f"ベースライン評価に失敗しました: {e}")
            return {"status": "failed", "error": f"ベースライン評価に失敗: {e}"}
        
        # 現在のベスト情報
        best_code = current_code
        best_f_measure = base_f_measure
        best_version = "base"
        
        # 改善サイクルの開始
        for iteration in range(1, iterations + 1):
            logger.info(f"改善サイクル {iteration}/{iterations} を開始します")
            
            try:
                # 1. コード改善プロンプトの作成
                improvement_prompt = f"""
                以下の音楽情報検出アルゴリズムを改善してください。

                ## 現在のコード
                ```python
                {current_code}
                ```

                ## 評価結果
                {formatted_eval}

                ## 改善目標
                {goal}

                ## 具体的な指示
                - コードの構造と処理フローを保ちつつ、アルゴリズムの精度を改善してください
                - 特にノイズやリバーブなどの環境での耐性を強化してください
                - コード内のコメントは保持または改善してください
                - 機能が明確に理解できるよう、適切な変数名とコメントを使用してください
                - 完全なPythonスクリプトとして動作する最終的なコードを提供してください

                ## 回答形式
                ```python
                # ここに改善されたコード全体を記載
                ```
                """
                
                # 2. コード改善の実行
                improved_result = await self.improve_code(improvement_prompt)
                improved_code = improved_result.get("improved_code")
                
                if not improved_code:
                    logger.error("コード改善に失敗しました：空のコードが返されました")
                    continue
                    
                # 3. 改善コードの保存
                version = f"v{iteration}"
                save_result = await self.save_code(detector_name, improved_code, version)
                improved_detector_name = f"{detector_name}_{version}"
                
                # 4. 改善コードの評価
                eval_result = await self.run_evaluation(improved_detector_name)
                current_f_measure = eval_result.get("overall_metrics", {}).get("note", {}).get("f_measure", 0)
                
                # 5. 評価結果の整形とログ
                formatted_eval = self._format_evaluation_results(eval_result)
                logger.info(f"サイクル {iteration} - F値: {current_f_measure:.4f} (変化: {current_f_measure - best_f_measure:.4f})")
                
                # 6. 履歴へ追加
                await self.add_history_event("improvement_cycle", {
                    "iteration": iteration,
                    "detector_name": improved_detector_name,
                    "f_measure": current_f_measure,
                    "change": current_f_measure - best_f_measure,
                    "evaluation_result": eval_result
                })
                
                # 7. 改善があればベスト情報を更新
                if current_f_measure > best_f_measure:
                    improvement_percentage = ((current_f_measure - best_f_measure) / best_f_measure * 100)
                    logger.info(f"改善を検出しました: +{improvement_percentage:.2f}%")
                    best_f_measure = current_f_measure
                    best_code = improved_code
                    best_version = version
                    
                    # 大幅な改善があればグリッドサーチを実行
                    if self.grid_search_enabled and improvement_percentage >= 5.0:
                        logger.info(f"大幅な改善を検出 (+{improvement_percentage:.2f}%) - グリッドサーチを実行します")
                        try:
                            # パラメータ提案の取得
                            params_suggestion = await self.suggest_parameters(improved_code)
                            
                            # グリッドサーチ用パラメータの準備
                            grid_params = {}
                            if isinstance(params_suggestion, dict) and "params" in params_suggestion:
                                for param in params_suggestion["params"]:
                                    name = param.get("name")
                                    suggested_range = param.get("suggested_range")
                                    if name and suggested_range and isinstance(suggested_range, list) and len(suggested_range) == 2:
                                        min_val, max_val = suggested_range
                                        step = param.get("step", (max_val - min_val) / 4)
                                        values = [min_val + i * step for i in range(5)]
                                        grid_params[name] = values
                            
                            # グリッドサーチの実行
                            if grid_params:
                                grid_search_result = await self.run_grid_search(improved_detector_name, grid_params)
                                
                                # グリッドサーチ結果から最適パラメータを取得
                                best_params = grid_search_result.get("best_params")
                                optimized_f_measure = grid_search_result.get("best_f_measure", 0)
                                
                                # 履歴に追加
                                await self.add_history_event("grid_search", {
                                    "detector_name": improved_detector_name,
                                    "best_params": best_params,
                                    "best_f_measure": optimized_f_measure,
                                    "improvement": optimized_f_measure - current_f_measure
                                })
                                
                                # さらに改善があれば更新
                                if optimized_f_measure > best_f_measure:
                                    best_f_measure = optimized_f_measure
                                    logger.info(f"グリッドサーチにより最適化: F値 {optimized_f_measure:.4f}")
                        except Exception as grid_err:
                            logger.error(f"グリッドサーチ実行中にエラーが発生しました: {grid_err}")
                
                # 次のサイクルのためにコード更新
                current_code = best_code
                
            except Exception as cycle_err:
                logger.error(f"改善サイクル {iteration} 中にエラーが発生しました: {cycle_err}")
                # エラーがあってもサイクルは続行
        
        # 最終結果の準備
        final_result = {
            "status": "completed",
            "base_f_measure": base_f_measure,
            "best_f_measure": best_f_measure,
            "improvement": best_f_measure - base_f_measure,
            "improvement_percentage": ((best_f_measure - base_f_measure) / base_f_measure * 100) if base_f_measure > 0 else 0,
            "best_version": best_version,
            "iterations_completed": iterations,
            "session_id": self.session_id
        }
        
        logger.info(f"改善サイクルが完了しました: {final_result}")
        return final_result

# --- Main Execution --- 
async def main():
    """
    メイン実行関数
    """
    # コマンドライン引数の解析
    import argparse
    parser = argparse.ArgumentParser(description="MCP対応アルゴリズム自動改善ツール")
    parser.add_argument("--detector", type=str, required=True, help="改善対象の検出器名")
    parser.add_argument("--goal", type=str, default=None, help="改善目標の説明")
    parser.add_argument("--iterations", type=int, default=None, help="改善サイクルの最大回数")
    parser.add_argument("--server-url", type=str, default=None, help="MCPサーバーのURL")
    parser.add_argument("--config", type=str, default=None, help="設定ファイルパス")
    args = parser.parse_args()
    
    # 自動改善ツールの初期化
    improver = AutoImprover(config_path=args.config, server_url=args.server_url)
    
    try:
        # 改善サイクルの実行
        result = await improver.run_improvement_cycle(
            detector_name=args.detector,
            goal=args.goal,
            iterations=args.iterations
        )
        
        # 結果の表示
        print("\n=== 改善結果 ===")
        print(f"ベースラインF値: {result['base_f_measure']:.4f}")
        print(f"最終F値: {result['best_f_measure']:.4f}")
        print(f"改善率: {result['improvement_percentage']:.2f}%")
        print(f"最良バージョン: {result['best_version']}")
        print(f"完了したサイクル数: {result['iterations_completed']}")
        print(f"セッションID: {result['session_id']}")
        
        # 成功終了
        return 0
    except Exception as e:
        logger.error(f"実行中にエラーが発生しました: {e}")
        print(f"エラー: {e}")
        return 1

if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main())) 