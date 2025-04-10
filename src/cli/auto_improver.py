# auto_improver.py

import requests
import time
import os
import sys
import json
import logging
import uuid
import inspect
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import platform
import numpy as np
import yaml # yaml をインポート

# Add project root to sys.path to allow importing detectors
# project_root = os.path.dirname(os.path.abspath(__file__))
# スクリプト実行時のパスではなく、utilsを使ってプロジェクトルートを取得する
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# path_utils をインポート (get_project_root など必要な関数)
from src.utils.path_utils import (
    get_project_root,
    load_environment_variables,
    get_workspace_dir,
    get_evaluation_results_dir,
    get_grid_search_results_dir,
    get_improved_versions_dir,
    get_audio_dir,
    get_label_dir,
    # get_state_dir # 削除: 状態はサーバーで管理
)
from src.utils.json_utils import NumpyEncoder # シリアライズ用
# 修正: 科学的自動化戦略クラスをインポート
from src.science_automation.exploration_strategy import ExplorationStrategy

# retry_on_exception をインポート
from src.utils.exception_utils import retry_on_exception, LLMError, EvaluationError # LLMError もインポート

# Attempt to import detectors dynamically (optional, for fetching code)
try:
    from src.detectors import DETECTOR_REGISTRY
except ImportError as e:
    print(f"Warning: Could not import DETECTOR_REGISTRY from src.detectors. ({e})")
    DETECTOR_REGISTRY = {}

# MCPクライアントライブラリ
from mcp.client import Client as MCPClient # クラス名を直接インポート

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('auto_improver')

# --- Default Configuration Values --- (ハードコードされたデフォルト値)
DEFAULT_MCP_SERVER_URL = "http://localhost:5002"
DEFAULT_POLL_INTERVAL = 5  # seconds
DEFAULT_JOB_TIMEOUT = 600  # seconds (10 minutes for individual jobs)
DEFAULT_SESSION_TIMEOUT = 3600  # seconds (1 hour for the entire session)
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_IMPROVEMENT_THRESHOLD = 0.005 # 修正指示書に合わせる (0.05 -> 0.005)
DEFAULT_GRID_SEARCH_ENABLED = True
DEFAULT_GRID_PARAMS = {
    'params_to_search': ['f0_score_threshold', 'HCF_ONSET_PEAK_THRESH'],
    'value_ranges': {
        'f0_score_threshold': [0.2, 0.25, 0.3, 0.35, 0.4],
        'HCF_ONSET_PEAK_THRESH': [0.1, 0.15, 0.2, 0.25, 0.3]
    }
}
DEFAULT_LOG_LEVEL = 'INFO'

# config.yaml の読み込み関数 (mcp_server.py から流用)
def load_yaml_config(config_path: Path) -> dict:
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"{config_path.name} の読み込みに失敗しました: {e}")
            return {}
    else:
        logger.warning(f"{config_path.name} が見つかりません: {config_path}")
        return {}

class AutoImprover:
    """MCP対応サーバーを利用して、アルゴリズム改善プロセスを調整します。"""

    def __init__(self, config_path=None, server_url=None):
        # プロジェクトルートとデフォルト設定ファイルのパス
        project_root = get_project_root()
        default_config_path = project_root / 'config.yaml'

        # .env ファイルをロード
        load_environment_variables()

        # 設定ファイルの読み込み (指定されたパス > デフォルトパス)
        config_to_load = Path(config_path) if config_path else default_config_path
        self.config = load_yaml_config(config_to_load)

        # サーバーURLの設定（優先順位: 引数 > 環境変数 > config.yaml > デフォルト）
        self.server_url = server_url or \
                          os.environ.get("MCP_SERVER_URL") or \
                          self.config.get('mcp_server', {}).get('url', DEFAULT_MCP_SERVER_URL)

        # ポーリング間隔とタイムアウト (環境変数 > config.yaml > デフォルト)
        mcp_config = self.config.get('mcp_server', {})
        self.poll_interval = int(os.environ.get("MCP_POLL_INTERVAL") or \
                               mcp_config.get('poll_interval', DEFAULT_POLL_INTERVAL))
        self.job_timeout = int(os.environ.get("MCP_JOB_TIMEOUT") or \
                             mcp_config.get('job_timeout', DEFAULT_JOB_TIMEOUT))
        self.session_timeout = int(os.environ.get("MCP_SESSION_TIMEOUT") or \
                                 mcp_config.get('session_timeout', DEFAULT_SESSION_TIMEOUT))

        # 改善サイクルの設定 (環境変数 > config.yaml > デフォルト)
        improvement_config = self.config.get('improvement', {})
        self.max_iterations = int(os.environ.get("MCP_MAX_ITERATIONS") or \
                                improvement_config.get('max_iterations', DEFAULT_MAX_ITERATIONS))
        self.improvement_threshold = float(os.environ.get("MCP_IMPROVEMENT_THRESHOLD") or \
                                         improvement_config.get('improvement_threshold', DEFAULT_IMPROVEMENT_THRESHOLD))

        # グリッドサーチ設定 (環境変数 > config.yaml > デフォルト)
        grid_search_config = self.config.get('grid_search', {})
        grid_search_enabled_env = os.environ.get("MCP_GRID_SEARCH_ENABLED")
        if grid_search_enabled_env is not None:
            self.grid_search_enabled = grid_search_enabled_env.lower() in ("true", "1", "yes")
        else:
            self.grid_search_enabled = grid_search_config.get('enabled', DEFAULT_GRID_SEARCH_ENABLED)

        # グリッドサーチパラメータ (環境変数(JSON) > config.yaml > デフォルト)
        grid_params_env = os.environ.get("MCP_GRID_PARAMS")
        if grid_params_env:
            try:
                self.grid_search_params = json.loads(grid_params_env)
                logger.info("環境変数 MCP_GRID_PARAMS からグリッドサーチパラメータを読み込みました")
            except json.JSONDecodeError as e:
                logger.warning(f"環境変数 MCP_GRID_PARAMS の解析に失敗しました: {e}。config.yamlまたはデフォルト値を使用します。")
                self.grid_search_params = grid_search_config.get('parameters', DEFAULT_GRID_PARAMS)
        else:
            self.grid_search_params = grid_search_config.get('parameters', DEFAULT_GRID_PARAMS)

        # セッション情報
        self.session_id = None
        self.session_data = None # サーバーから取得したセッション情報全体
        self.current_detector_name = None
        # self.current_detector_code = None # バージョン管理はサーバーに任せる
        self.last_evaluation_results = None

        # 改善サイクルの状態管理 (サーバーの状態のローカルキャッシュ)
        self.cycle_state = {} # ループ開始時にサーバーからロード

        # ログレベルの設定 (環境変数 > config.yaml > デフォルト)
        logging_config = self.config.get('logging', {})
        log_level_env = os.environ.get("MCP_LOG_LEVEL")
        log_level = log_level_env or logging_config.get('level', DEFAULT_LOG_LEVEL)
        logging.getLogger('auto_improver').setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # MCPクライアントの初期化
        self.client = MCPClient(self.server_url)
        logger.info(f"MCPクライアントを初期化しました（サーバーURL: {self.server_url}）")
        logger.info(f"Improvement settings: max_iterations={self.max_iterations}, threshold={self.improvement_threshold}, grid_search={self.grid_search_enabled}")


    # --- 状態管理メソッド (削除) ---
    # save_state, load_state はサーバー中心のため削除

    # --- MCPツール呼び出しラッパー ---
    @retry_on_exception(logger=logger, log_message_prefix="MCP Tool Call")
    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        MCPツールをリトライ付きで呼び出します。
        Args:
            tool_name: 呼び出すツール名
            **kwargs: ツールに渡す引数
        Returns:
            Any: ツールの実行結果 (ジョブID辞書 or 直接結果)
        Raises:
            LLMError: ネットワークエラーやタイムアウトの場合
            Exception: その他の予期せぬエラー
        """
        logger.debug(f"Calling tool '{tool_name}' with args: {kwargs}")
        try:
            # client.call_tool はジョブIDまたは直接の結果を返す
            result = await self.client.call_tool(tool_name, **kwargs)
            logger.debug(f"Tool '{tool_name}' call successful (raw result: {result}).")

            # 結果がエラーを示す形式か確認 (これはツール自体の実行結果のエラー)
            if isinstance(result, dict) and ('error' in result or result.get('status') == 'failed'):
                error_msg = result.get('error', 'Unknown error from tool execution')
                logger.warning(f"Tool '{tool_name}' execution returned an error state: {error_msg}")
                # ここではエラー応答をそのまま返す (呼び出し元でハンドリング)

            return result
        except (requests.exceptions.RequestException, requests.Timeout) as e:
            logger.error(f"Network/Timeout error calling tool '{tool_name}': {e}")
            raise LLMError(f"Network/Timeout error calling tool '{tool_name}': {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error calling tool '{tool_name}': {e}", exc_info=True)
            raise # 予期せぬエラーはそのまま再 raise

    # --- ジョブステータスポーリング ---
    async def poll_job_status(self, job_id: str, timeout: int = DEFAULT_JOB_TIMEOUT) -> Dict[str, Any]:
        """非同期ジョブのステータスをポーリングし、完了したら結果を返します。"""
        start_time = time.time()
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                logger.error(f"ジョブ {job_id} がタイムアウトしました（{timeout}秒）")
                if self.session_id:
                    try:
                        # サーバーにタイムアウトイベントを記録 (失敗しても継続)
                        await self.call_tool("add_session_history",
                                             session_id=self.session_id,
                                             event_type="job_timeout",
                                             event_data=json.dumps({"job_id": job_id, "timeout_duration": timeout}))
                    except Exception as hist_err:
                        logger.warning(f"セッション履歴へのジョブタイムアウト記録に失敗しました: {hist_err}")
                raise TimeoutError(f"ジョブ {job_id} がタイムアウトしました（{timeout}秒）")

            try:
                logger.info(f"ジョブ {job_id} のステータス確認中...")
                # get_job_status ツールを呼び出す
                job_status_result = await self.call_tool("get_job_status", job_id=job_id)

                # get_job_status はジョブ情報全体を返す想定
                if not isinstance(job_status_result, dict):
                     logger.warning(f"get_job_status が予期しない形式を返しました: {job_status_result}。リトライします...")
                     await asyncio.sleep(self.poll_interval)
                     continue

                status = job_status_result.get("status")

                if status == 'completed':
                    logger.info(f"ジョブ {job_id} が正常に完了しました")
                    return job_status_result # ジョブ情報全体を返す
                elif status == 'failed':
                    error_msg = job_status_result.get("error", "不明なエラー")
                    error_details = job_status_result.get("error_details", {}) # サーバーで追加される可能性
                    traceback_str = job_status_result.get("traceback", "") # サーバーで追加される可能性

                    logger.error(f"ジョブ {job_id} が失敗しました: {error_msg}")
                    if error_details: logger.error(f"  詳細: {error_details}")
                    if traceback_str: logger.error(f"  トレースバック:\n{traceback_str}")

                    if self.session_id:
                         try:
                             # 失敗イベントを記録 (失敗しても継続)
                             await self.call_tool("add_session_history",
                                                  session_id=self.session_id,
                                                  event_type="job_failed",
                                                  event_data=json.dumps({
                                                      "job_id": job_id,
                                                      "error": error_msg,
                                                      "error_details": error_details # 詳細も記録
                                                  }))
                         except Exception as hist_err:
                            logger.warning(f"セッション履歴へのジョブ失敗記録に失敗しました: {hist_err}")
                    # ジョブ失敗を示す情報を伴う例外を発生させる
                    raise RuntimeError(f"サーバーでジョブ {job_id} が失敗しました: {error_msg}")
                elif status in ['pending', 'running', 'queued']:
                    logger.info(f"ジョブ {job_id} のステータス: {status}。{self.poll_interval}秒待機...")
                    await asyncio.sleep(self.poll_interval)
                else:
                    logger.warning(f"予期しないジョブステータス '{status}' (ジョブ {job_id})。リトライします...")
                    await asyncio.sleep(self.poll_interval)

            except LLMError as e: # call_tool が発生させる可能性
                logger.error(f"ジョブ {job_id} のポーリング中にネットワーク/タイムアウトエラー: {e}。リトライします...")
                await asyncio.sleep(self.poll_interval * 2) # エラー時は少し長めに待つ
            except Exception as e:
                # RuntimeError や TimeoutError 以外
                logger.error(f"ジョブ {job_id} のポーリング中に予期せぬエラーが発生しました: {e}", exc_info=True)
                # 予期せぬエラーはリトライせずにループを抜けるか？ -> ここではリトライする
                await asyncio.sleep(self.poll_interval * 2)

    # --- 履歴追加 (状態更新機能付き) ---
    async def add_history_event(self,
                              event_type: str,
                              data: Dict[str, Any],
                              cycle_state_update: Optional[Dict[str, Any]] = None
                              ) -> Optional[Dict[str, Any]]: # 更新後のセッション情報全体を返す可能性
        """セッション履歴にイベントを追加し、必要なら cycle_state も更新します。"""
        if not self.session_id:
            logger.warning("アクティブなセッションがないため、履歴イベントを追加できません")
            return None
        try:
            # event_data を JSON シリアライズ可能にする (NumpyEncoder を使用)
            try:
                serialized_data = json.dumps(data, cls=NumpyEncoder)
            except TypeError as e:
                logger.warning(f"イベントデータ ({event_type}) のJSONシリアライズに失敗: {e}。文字列に変換します。")
                # オブジェクトを安全に文字列化する試み
                try:
                    serialized_data = json.dumps(str(data))
                except Exception as str_e:
                     logger.error(f"イベントデータの文字列変換も失敗: {str_e}")
                     serialized_data = json.dumps({"error": "Data serialization failed"})

            # cycle_state_update は辞書のまま渡す (ツール側が処理)
            serialized_update = cycle_state_update

            # add_session_history ツールを呼び出し
            # サーバー側が更新後のセッション情報全体を返すことを期待
            response = await self.call_tool(
                "add_session_history",
                session_id=self.session_id,
                event_type=event_type,
                event_data=serialized_data, # シリアライズ済みデータ
                cycle_state_update=serialized_update # 辞書を渡す
            )

            # 応答から更新後のセッション情報を取得
            # サーバーの add_session_history は更新後のセッション情報全体 (dict) を返すと想定
            if isinstance(response, dict):
                 updated_session_info = response
                 logger.info(f"セッション {self.session_id} に履歴イベント '{event_type}' を追加しました" + (" (状態更新あり)" if cycle_state_update else ""))
                 # 更新後の cycle_state でクライアント側キャッシュを更新
                 if 'cycle_state' in updated_session_info:
                     self.cycle_state = updated_session_info['cycle_state']
                     logger.debug(f"クライアント側の cycle_state を更新: {self.cycle_state}")
                 else:
                     logger.warning("サーバー応答に 'cycle_state' が含まれていません。")
                 return updated_session_info
            else:
                 # 応答形式が予期しない場合
                 logger.error(f"add_session_history 応答が予期しない形式です: {type(response)}")
                 # ここでエラーを raise するか、None を返すか？ -> Noneを返す
                 return None

        except Exception as e:
            logger.error(f"セッション {self.session_id} への履歴イベント '{event_type}' の追加/状態更新に失敗しました: {e}", exc_info=True)
            return None # エラー時は None を返す

    # --- セッション管理ツールラッパー ---
    async def start_session(self, base_algorithm: str, goal: Optional[str] = None) -> Dict[str, Any]:
        """MCPサーバーで新しい改善セッションを開始します。"""
        logger.info(f"新規セッション開始リクエスト: アルゴリズム={base_algorithm}")
        try:
            # start_session ツールを呼び出す (結果はセッション情報全体と想定)
            session_info = await self.call_tool(
                "start_session",
                base_algorithm=base_algorithm,
                goal=goal or f"{base_algorithm} のパフォーマンス改善" # デフォルトゴール
            )
            if not isinstance(session_info, dict) or "session_id" not in session_info:
                raise ValueError(f"start_session 応答が不正です: {session_info}")

            self.session_id = session_info["session_id"]
            self.session_data = session_info # 全体を保持
            self.cycle_state = session_info.get('cycle_state', {}) # 初期状態を設定
            self.current_detector_name = base_algorithm
            logger.info(f"新規セッション開始成功: ID={self.session_id}")
            return session_info
        except Exception as e:
            logger.error(f"セッション開始に失敗しました: {e}", exc_info=True)
            raise # エラーを再発生させる

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """指定されたセッションの情報をサーバーから取得します。"""
        logger.info(f"セッション情報取得リクエスト: ID={session_id}")
        try:
            # get_session_info ツールを呼び出す
            session_info = await self.call_tool("get_session_info", session_id=session_id)
            if isinstance(session_info, dict):
                 return session_info
            else:
                 logger.warning(f"get_session_info 応答が不正です: {session_info}")
                 return None
        except Exception as e:
            logger.error(f"セッション情報取得に失敗しました ({session_id}): {e}", exc_info=True)
            return None # エラー時は None を返す

    # --- コード管理ツールラッパー ---
    async def get_detector_code(self, detector_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """検出器のコードをサーバーから取得します。"""
        logger.info(f"コード取得リクエスト: 検出器={detector_name}, バージョン={version or 'latest'}")
        try:
            # get_code ツールを呼び出す (ジョブIDが返る)
            job_start_result = await self.call_tool("get_code", detector_name=detector_name, version=version, session_id=self.session_id) # session_idも渡す
            job_id = job_start_result.get("job_id")
            if not job_id: raise ValueError("get_code がジョブIDを返しませんでした。")

            # ジョブ完了待機
            job_result = await self.poll_job_status(job_id, timeout=60) # コード取得は短いはず
            if job_result.get('status') == 'completed':
                code_content = job_result.get('result') # 結果はコード文字列そのものと想定
                if isinstance(code_content, str):
                     # 取得したバージョン情報も返せると良い (サーバー側ツールに依存)
                     # ここでは取得できたコードのみ返す
                     return {"code": code_content, "version": version or "latest"}
                else:
                     raise ValueError(f"get_code ジョブ結果が文字列ではありません: {type(code_content)}")
            else:
                 error_msg = job_result.get('error', '不明なエラー')
                 raise RuntimeError(f"get_code ジョブが失敗しました: {error_msg}")
        except Exception as e:
            logger.error(f"コード取得 ({detector_name}, {version or 'latest'}) に失敗しました: {e}", exc_info=True)
            raise # エラーを再発生させる

    async def update_detector_code( # save_code から名称変更
        self,
        detector_name: str,
        code: str,
        session_id: str,
        parent_version: Optional[str] = None,
        changes_summary: Optional[str] = None
    ) -> Dict[str, Any]: # 戻り値は保存結果情報 (バージョンタグ等含む)
        """改善されたコードをサーバーに保存します。"""
        logger.info(f"コード保存リクエスト: 検出器={detector_name}, 親バージョン={parent_version}")
        try:
            # save_code ツールを呼び出す (ジョブIDが返る)
            job_start_result = await self.call_tool(
                "save_code",
                detector_name=detector_name,
                code=code,
                version=None, # サーバー側で自動生成させる
                parent_version=parent_version, # 親バージョンを渡す
                changes_summary=changes_summary or "コード改善", # 変更概要を渡す
                session_id=session_id
            )
            job_id = job_start_result.get("job_id")
            if not job_id: raise ValueError("save_code がジョブIDを返しませんでした。")

            # ジョブ完了待機
            job_result = await self.poll_job_status(job_id, timeout=60) # 保存も短いはず
            if job_result.get('status') == 'completed':
                save_details = job_result.get('result') # 結果は保存情報辞書と想定
                if isinstance(save_details, dict) and "version" in save_details:
                     logger.info(f"コード保存成功。バージョン: {save_details['version']}")
                     return save_details # 保存情報 (バージョンタグ等) を返す
                else:
                     raise ValueError(f"save_code ジョブ結果が不正です: {save_details}")
            else:
                 error_msg = job_result.get('error', '不明なエラー')
                 raise RuntimeError(f"save_code ジョブが失敗しました: {error_msg}")
        except Exception as e:
            logger.error(f"コード保存 ({detector_name}) に失敗しました: {e}", exc_info=True)
            raise

    # --- LLMツールラッパー ---
    async def improve_code(self, prompt: str, session_id: str) -> Dict[str, Any]:
        """LLMを使用してコードを改善します。"""
        logger.info("コード改善リクエストを送信します...")
        try:
            job_start_result = await self.call_tool("improve_code", prompt=prompt, session_id=session_id)
            job_id = job_start_result.get("job_id")
            if not job_id: raise ValueError("improve_code がジョブIDを返しませんでした。")

            logger.info(f"コード改善ジョブ ({job_id}) を開始しました。完了を待ちます...")
            job_result = await self.poll_job_status(job_id, timeout=self.job_timeout) # LLMは時間がかかる
            logger.info(f"コード改善ジョブ完了: {job_result.get('status')}")
            return job_result # ジョブ結果全体を返す
        except Exception as e:
            logger.error(f"コード改善リクエストまたはジョブ待機に失敗しました: {e}", exc_info=True)
            # 失敗を示す辞書を返す
            return {"status": "failed", "error": f"コード改善エラー: {e}"}

    # 他のLLMツールラッパーも同様に追加... (analyze_evaluation_results, generate_hypotheses, etc.)

    # --- 評価ツールラッパー ---
    async def evaluate_detector(self,
                              detector_name: str,
                              code_version: Optional[str] = None,
                              dataset_name: Optional[str] = None,
                              audio_path: Optional[str] = None,
                              ref_path: Optional[str] = None,
                              session_id: Optional[str] = None
                              ) -> Dict[str, Any]: # ジョブ結果全体を返すように変更
        """検出器の評価をサーバーで実行します。"""
        logger.info(f"評価リクエスト: 検出器={detector_name}, バージョン={code_version or 'latest'}, データセット={dataset_name or '個別指定'}")
        if not session_id: session_id = self.session_id
        if not session_id: raise ValueError("評価実行にはセッションIDが必要です。")
        if not dataset_name and not (audio_path and ref_path):
            raise ValueError("評価には dataset_name または audio_path/ref_path が必要です。")

        try:
            # run_evaluation ツールを呼び出す
            job_start_result = await self.call_tool(
                "run_evaluation",
                detector_name=detector_name,
                code_version=code_version, # 評価対象のバージョンを指定
                dataset_name=dataset_name,
                audio_path=audio_path,
                ref_path=ref_path,
                # evaluator_config はサーバー側で読み込む想定？ or ここで渡す？ -> サーバー側で読み込む想定とする
                session_id=session_id
            )
            job_id = job_start_result.get("job_id")
            if not job_id: raise ValueError("run_evaluation がジョブIDを返しませんでした。")

            logger.info(f"評価ジョブ ({job_id}) を開始しました。完了を待ちます...")
            job_result = await self.poll_job_status(job_id, timeout=self.job_timeout) # 評価は時間がかかる可能性
            logger.info(f"評価ジョブ完了: {job_result.get('status')}")
            return job_result # ジョブ結果全体を返す
        except Exception as e:
            logger.error(f"評価リクエストまたはジョブ待機に失敗しました ({detector_name}): {e}", exc_info=True)
            return {"status": "failed", "error": f"評価エラー: {e}"}

    # グリッドサーチやパラメータ最適化のラッパーも同様に追加...
    # async def run_grid_search(...)
    # async def optimize_parameters(...)

    # --- 可視化ツールラッパー ---
    async def visualize_improvement_trajectory(self, session_id: str) -> Dict[str, Any]:
        """改善軌跡の可視化をサーバーに依頼します。"""
        logger.info(f"改善軌跡の可視化リクエスト: SessionID={session_id}")
        if not session_id: raise ValueError("可視化にはセッションIDが必要です。")
        try:
            job_start_result = await self.call_tool("visualize_improvement_trajectory", session_id=session_id)
            job_id = job_start_result.get("job_id")
            if not job_id: raise ValueError("visualize_improvement_trajectory がジョブIDを返しませんでした。")

            logger.info(f"可視化ジョブ ({job_id}) を開始しました。完了を待ちます...")
            job_result = await self.poll_job_status(job_id, timeout=120) # 可視化は比較的早いはず
            logger.info(f"可視化ジョブ完了: {job_result.get('status')}")
            return job_result # ジョブ結果全体を返す
        except Exception as e:
            logger.error(f"改善軌跡の可視化リクエストまたはジョブ待機に失敗しました ({session_id}): {e}", exc_info=True)
            return {"status": "failed", "error": f"可視化エラー: {e}"}


    # --- 改善サイクル実行 (修正) ---
    async def run_improvement_cycle(
        self,
        detector_name: str,
        goal: Optional[str] = None,
        iterations: Optional[int] = None,
        session_id: Optional[str] = None, # 再開用セッションID
        dataset_name: Optional[str] = None, # 評価用データセット名
        audio_path: Optional[str] = None,   # または個別ファイルパス
        ref_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        指定された検出器に対する改善サイクルを実行します。(クライアント駆動方式)

        Args:
            detector_name: 改善対象の検出器名。
            goal: 改善の目標（オプション）。
            iterations: 最大改善回数（Noneの場合、設定から取得）。
            session_id: 再開するセッションID（Noneの場合、新規）。
            dataset_name: 評価に使用するデータセット名 (config.yamlで定義)。
            audio_path: 評価に使用する単一音声ファイルパス (dataset_nameと排他)。
            ref_path: 評価に使用する単一参照ファイルパス (dataset_nameと排他)。

        Returns:
            Dict[str, Any]: 改善結果の概要。
        """
        start_time = time.time() # ループ全体の開始時刻
        self.current_detector_name = detector_name
        effective_iterations = iterations if iterations is not None else self.max_iterations

        # データセット指定の検証
        if not dataset_name and not (audio_path and ref_path):
            raise ValueError("dataset_name または (audio_path と ref_path) が必要です。")
        if dataset_name and (audio_path or ref_path):
             raise ValueError("dataset_name と audio_path/ref_path は同時に指定できません。")

        # --- 1. セッションの開始または再開 --- #
        if session_id:
            logger.info(f"既存セッション {session_id} を再開します")
            try:
                 session_info = await self.get_session_info(session_id)
                 if not session_info:
                     logger.error(f"セッション {session_id} がサーバーで見つかりませんでした。新規セッションを開始します。")
                     session_id = None
                 elif session_info.get("base_algorithm") != detector_name:
                     logger.error(f"セッション {session_id} は検出器 '{session_info.get('base_algorithm')}' 用です。'{detector_name}' 用ではありません。")
                     return {"status": "error", "message": "Session detector mismatch"}
                 else:
                     self.session_id = session_id
                     self.session_data = session_info # 取得した情報で更新
                     self.cycle_state = session_info.get('cycle_state', {}) # 状態をロード
                     best_f = self.cycle_state.get('best_f_measure')
                     best_f_str = f"{best_f:.4f}" if isinstance(best_f, float) else 'N/A'
                     current_iter = self.cycle_state.get('current_iteration', 0)
                     logger.info(f"サーバーから状態をロードしました: イテレーション {current_iter}, ベストF値 {best_f_str}")
            except Exception as e:
                 logger.error(f"既存セッション {session_id} の情報取得に失敗: {e}. 新規セッションとして開始します。", exc_info=True)
                 session_id = None

        if not self.session_id:
            logger.info("新規改善セッションを開始します")
            try:
                session_start_result = await self.start_session(detector_name, goal)
                # self.session_id, self.session_data, self.cycle_state は start_session内で設定される
                if not self.session_id: raise ValueError("サーバーから有効なセッションIDを取得できませんでした。")
                # 開始イベントを記録
                await self.add_history_event("cycle_started", {"goal": goal, "max_iterations": effective_iterations})
            except Exception as e:
                logger.error(f"セッションの開始に失敗しました: {e}", exc_info=True)
                return {"status": "error", "message": f"Failed to start session: {e}"}

        logger.info(f"セッション {self.session_id} を使用して改善ループを開始します。最大イテレーション: {effective_iterations}")

        # --- 2. メインループ --- #
        final_status = 'unknown'
        try:
            while self.cycle_state.get('current_iteration', 0) < effective_iterations:
                iteration = self.cycle_state.get('current_iteration', 0)
                logger.info(f"--- イテレーション {iteration + 1}/{effective_iterations} --- Session: {self.session_id}")
                iteration_start_time = time.time()

                # イテレーション開始をサーバーに記録
                await self.add_history_event("iteration_started", {"iteration": iteration + 1})

                # --- 2.1 評価 --- #
                code_version_to_evaluate = self.cycle_state.get('current_code_version') # Noneなら最新版
                logger.info(f"評価対象バージョン: {code_version_to_evaluate or 'latest'}")
                evaluation_results = None
                current_metrics = None
                current_f_measure = None
                evaluated_code_version = code_version_to_evaluate

                try:
                    eval_job_result = await self.evaluate_detector(
                        detector_name=detector_name,
                        code_version=code_version_to_evaluate,
                        dataset_name=dataset_name,
                        audio_path=audio_path,
                        ref_path=ref_path,
                        session_id=self.session_id
                    )

                    if eval_job_result.get('status') != 'completed':
                         raise EvaluationError(f"評価ジョブが失敗または完了しませんでした: Status={eval_job_result.get('status')}, Error={eval_job_result.get('error')}")

                    evaluation_results = eval_job_result.get('result', {}).get('evaluation_results') # 'evaluation_results' キーに格納されていると想定
                    evaluated_code_version = eval_job_result.get('result', {}).get('evaluated_version', code_version_to_evaluate) # 実際に評価されたバージョンタグ

                    if not evaluation_results:
                         raise EvaluationError("評価ジョブは完了しましたが、結果を取得できませんでした。")

                    detector_metrics = evaluation_results.get(detector_name, {})
                    current_metrics = detector_metrics.get('overall_metrics', {})
                    current_f_measure = current_metrics.get("note", {}).get("f_measure")

                    if current_f_measure is None:
                        logger.warning("評価結果から F-measure (Note) を取得できませんでした。0.0として扱います。")
                        current_f_measure = 0.0

                    self.last_evaluation_results = evaluation_results
                    logger.info(f"評価完了。 Version: {evaluated_code_version}, F-measure (Note): {current_f_measure:.4f}")

                    # 評価結果と状態更新をサーバーに記録
                    eval_event_data = {
                        "iteration": iteration + 1,
                        "code_version": evaluated_code_version,
                        "f_measure": current_f_measure,
                        "metrics": current_metrics, # サーバー側でJSONにできる形式である必要
                        "job_id": eval_job_result.get('job_id')
                    }
                    state_updates = {
                        "latest_f_measure": current_f_measure,
                        "latest_metrics": current_metrics,
                        "evaluated_code_version": evaluated_code_version
                    }
                    # 評価完了イベントを記録 (cycle_state 更新はしない)
                    await self.add_history_event("evaluation_complete", eval_event_data)
                    # クライアント側状態のみ更新
                    self.cycle_state.update(state_updates)

                except Exception as e:
                    logger.error(f"イテレーション {iteration + 1} の評価フェーズでエラー: {e}", exc_info=True)
                    await self.add_history_event("iteration_failed", {"iteration": iteration + 1, "phase": "evaluation", "error": str(e)}, cycle_state_update={'status': 'failed'})
                    self.cycle_state['status'] = 'failed'
                    final_status = 'failed'
                    break # 評価失敗は致命的なのでループ中断

                # --- 2.2 ベースライン設定と改善/停滞判定 --- #
                try:
                    is_first_eval = (iteration == 0 and not self.cycle_state.get('is_resumed')) or self.cycle_state.get('best_f_measure') is None
                    check_updates = {} # このステップでの cycle_state 更新内容
                    is_stagnating = False # 初期化

                    if is_first_eval:
                         check_updates["baseline_f_measure"] = current_f_measure
                         check_updates["best_f_measure"] = current_f_measure
                         check_updates["best_code_version"] = evaluated_code_version
                         check_updates["stagnation_count"] = 0
                         logger.info(f"ベースライン F-measure を設定: {current_f_measure:.4f} (Version: {evaluated_code_version})")
                         # ベースライン設定イベント記録 (状態更新も送信)
                         await self.add_history_event(
                             "baseline_set",
                             {"f_measure": current_f_measure, "version": evaluated_code_version},
                             cycle_state_update=check_updates
                         )
                    else:
                         best_f_measure = self.cycle_state.get('best_f_measure', 0.0)
                         improvement = current_f_measure - best_f_measure
                         stagnation_count = self.cycle_state.get('stagnation_count', 0)

                         if improvement < self.improvement_threshold:
                             stagnation_count += 1
                             check_updates["stagnation_count"] = stagnation_count
                             logger.info(f"パフォーマンス変化 ({improvement:.4f}) が閾値 ({self.improvement_threshold}) 以下。停滞カウント: {stagnation_count}")
                             stagnation_limit = self.config.get('improvement', {}).get('stagnation_limit', 3)
                             if stagnation_count >= stagnation_limit:
                                 is_stagnating = True
                                 logger.warning(f"停滞を検出 (カウント >= {stagnation_limit})。")
                         else:
                             logger.info(f"新しいベストパフォーマンス！ F-measure: {current_f_measure:.4f} (+{improvement:.4f})" + f" (Version: {evaluated_code_version})")
                             stagnation_count = 0
                             check_updates["stagnation_count"] = stagnation_count
                             check_updates["best_f_measure"] = current_f_measure
                             check_updates["best_code_version"] = evaluated_code_version
                             # 新ベストイベント記録 (状態更新も送信)
                             await self.add_history_event(
                                 "new_best_found",
                                 {"f_measure": current_f_measure, "version": evaluated_code_version, "improvement": improvement},
                                 cycle_state_update=check_updates
                             )

                         # 停滞チェック結果を記録 (新ベストでない場合 or 常に記録する場合)
                         # new_best_found で更新した場合は、ここでは stagnation_count のみ更新 or イベント送信不要
                         if not check_updates.get("best_f_measure"): # 新ベストでなければ停滞カウントのみ更新
                             await self.add_history_event(
                                 "stagnation_check",
                                 {"iteration": iteration + 1, "improvement": improvement, "stagnation_count": stagnation_count, "is_stagnating": is_stagnating},
                                 cycle_state_update={"stagnation_count": stagnation_count}
                             )
                         # もし new_best_found と stagnation_check の両方で状態を更新する場合、
                         # add_history_event が最新の cycle_state を返すことを利用して、
                         # check_updates を毎回マージする必要がある。

                    # クライアント側の状態を更新
                    self.cycle_state.update(check_updates)

                except Exception as e:
                     logger.error(f"イテレーション {iteration + 1} の状態更新フェーズでエラー: {e}", exc_info=True)
                     await self.add_history_event("iteration_failed", {"iteration": iteration + 1, "phase": "state_update", "error": str(e)}, cycle_state_update={'status': 'failed'})
                     self.cycle_state['status'] = 'failed'
                     final_status = 'failed'
                     break

                # --- 2.3 戦略決定 --- #
                next_action = None
                try:
                    logger.info("次の探索戦略を決定します...")
                    # suggest_exploration_strategy ツール呼び出し
                    # このツールは結果を直接返す (ジョブIDではない) と想定
                    strategy_result = await self.call_tool(
                        "suggest_exploration_strategy",
                        session_id=self.session_id,
                        current_performance=current_f_measure # 現在のパフォーマンスを渡す
                    )
                    if isinstance(strategy_result, dict):
                         next_action = strategy_result.get('strategy_suggestion', {})
                         reason = strategy_result.get('reason', 'N/A')
                         logger.info(f"提案された戦略: {next_action.get('action', 'N/A')}, 理由: {reason}")
                         self.cycle_state['last_strategy'] = next_action # ローカルキャッシュ更新
                         # イベント記録 (サーバー側ツール内で記録されるか、ここで明示的に記録)
                         # await self.add_history_event("strategy_suggested", {"strategy": next_action, "reason": reason}) # サーバー側で記録済みと仮定
                    else:
                         logger.warning(f"戦略提案ツールが失敗または予期せぬ結果: {strategy_result}")
                         next_action = {'action': 'improve_code', 'reason': 'Strategy suggestion failed, defaulting'}
                         self.cycle_state['last_strategy'] = next_action
                         await self.add_history_event("strategy_suggestion_failed", {"reason": "Tool failed or returned unexpected format"})

                except Exception as e:
                    logger.error(f"イテレーション {iteration + 1} の戦略決定フェーズでエラー: {e}", exc_info=True)
                    await self.add_history_event("iteration_failed", {"iteration": iteration + 1, "phase": "strategy_suggestion", "error": str(e)}, cycle_state_update={'status': 'failed'})
                    self.cycle_state['status'] = 'failed'
                    final_status = 'failed'
                    break

                # --- 2.4 アクション実行 --- #
                next_code_version = self.cycle_state.get('current_code_version') # 次の評価対象バージョン
                action_type = next_action.get('action')
                logger.info(f"実行アクション: {action_type}")

                try:
                    action_taken = False
                    if action_type == "improve_code" or action_type == "improve_code_focused":
                        # コード改善
                        logger.info("コード改善を実行...")
                        code_to_improve_version = evaluated_code_version # 直前評価バージョンを改善
                        code_result = await self.get_detector_code(detector_name, code_to_improve_version)
                        current_code = code_result.get('code')
                        if not current_code: raise ValueError(f"改善対象コード取得失敗 (Ver: {code_to_improve_version})")

                        # プロンプト作成 (improve_code_focused の場合、仮説情報を使う)
                        hypothesis = next_action.get("hypothesis") if action_type == "improve_code_focused" else None
                        # TODO: _create_improvement_prompt などのヘルパーを復活させるか、ここでプロンプトを構築
                        # history_summary = self._generate_improvement_history_summary(...) # 履歴が必要
                        improve_prompt = f"Improve code for '{detector_name}'. Focus: {next_action.get('focus', 'performance')}. F={current_f_measure:.4f}\n```python\n{current_code}\n```"
                        if hypothesis: improve_prompt += f"\nHypothesis: {hypothesis.get('description')}"

                        # improve_code ツール呼び出し (ジョブ監視含む)
                        improve_job_result = await self.improve_code(prompt=improve_prompt, session_id=self.session_id)

                        if improve_job_result.get('status') == 'completed':
                            improved_code = improve_job_result.get('result', {}).get('improved_code')
                            changes_summary = improve_job_result.get('result', {}).get('changes_summary', 'LLM improvement')
                            if improved_code:
                                 logger.info("LLM改善提案受信。保存...")
                                 # save_code ツール呼び出し
                                 save_result = await self.update_detector_code( # update_detector_code を使用
                                     detector_name=detector_name, code=improved_code, session_id=self.session_id,
                                     parent_version=code_to_improve_version, changes_summary=changes_summary
                                 )
                                 new_version_tag = save_result.get("version")
                                 if new_version_tag:
                                     next_code_version = new_version_tag # 次の評価対象
                                     logger.info(f"新バージョン '{new_version_tag}' 保存完了。")
                                     await self.add_history_event("action_taken", {"action": action_type, "new_version": new_version_tag})
                                     action_taken = True
                                 else:
                                     logger.warning("コード保存失敗。バージョンタグ取得できず。")
                                     await self.add_history_event("action_failed", {"action": action_type, "reason": "Failed to save code"})
                            else:
                                 logger.warning("LLMコード改善失敗またはコード返却なし。")
                                 await self.add_history_event("action_failed", {"action": action_type, "reason": "LLM returned no valid code"})
                        else:
                            error_msg = improve_job_result.get('error', '不明なエラー')
                            logger.warning(f"コード改善ジョブ失敗: {error_msg}")
                            await self.add_history_event("action_failed", {"action": action_type, "reason": f"Improvement job failed: {error_msg}"})

                    elif action_type == "optimize_parameters":
                        # パラメータ最適化
                        logger.info("パラメータ最適化を実行...")
                        target_version_for_opt = self.cycle_state.get("best_code_version", evaluated_code_version) # ベストコードを対象
                        # optimize_parameters ツール呼び出し (ジョブID取得)
                        opt_job_start_result = await self.call_tool(
                            "optimize_parameters",
                            detector_name=detector_name, code_version=target_version_for_opt, session_id=self.session_id,
                            dataset_name=dataset_name, audio_path=audio_path, ref_path=ref_path # 評価用データ指定
                        )
                        if opt_job_start_result and opt_job_start_result.get('job_id'):
                             job_id = opt_job_start_result['job_id']
                             logger.info(f"パラメータ最適化ジョブ ({job_id}) 開始。完了待機...")
                             await self.add_history_event("action_started", {"action": action_type, "job_id": job_id})
                             # ジョブ完了ポーリング
                             opt_result_final = await self.poll_job_status(job_id, timeout=self.config.get('improvement', {}).get('optimization_timeout', 1800))
                             logger.info(f"パラメータ最適化ジョブ完了: {opt_result_final.get('status')}")
                             if opt_result_final.get('status') == 'completed':
                                 result_data = opt_result_final.get('result', {})
                                 best_params = result_data.get('grid_search_results', {}).get('best_params')
                                 if best_params:
                                     logger.info(f"最適パラメータ発見: {best_params}")
                                     # TODO: パラメータ適用バージョン作成ツール呼び出し？ or 次回 improve_code のプロンプトに含める？
                                     await self.add_history_event("params_optimized", {"best_params": best_params, "metrics": result_data.get('grid_search_results', {}).get('best_metrics')})
                                 else:
                                     logger.info("改善するパラメータは見つからず。")
                                     await self.add_history_event("action_completed_no_change", {"action": action_type})
                                 action_taken = True
                             else:
                                 error_msg = opt_result_final.get('error', '不明')
                                 logger.warning(f"パラメータ最適化ジョブ失敗: {error_msg}")
                                 await self.add_history_event("action_failed", {"action": action_type, "reason": f"Optimization job failed: {error_msg}"})
                        else:
                             logger.warning("パラメータ最適化ジョブ開始失敗。")
                             await self.add_history_event("action_failed", {"action": action_type, "reason": "Failed to start optimization job"})

                    elif action_type == "generate_hypothesis":
                         # 仮説生成
                         logger.info("仮説生成を実行...")
                         # generate_hypotheses ツール呼び出し (ジョブID取得)
                         hypo_job_start_result = await self.call_tool("generate_hypotheses", session_id=self.session_id)
                         if hypo_job_start_result and hypo_job_start_result.get('job_id'):
                              job_id = hypo_job_start_result['job_id']
                              logger.info(f"仮説生成ジョブ ({job_id}) 開始。完了待機...")
                              await self.add_history_event("action_started", {"action": action_type, "job_id": job_id})
                              # ジョブ完了ポーリング
                              hypo_result_final = await self.poll_job_status(job_id, timeout=self.job_timeout)
                              logger.info(f"仮説生成ジョブ完了: {hypo_result_final.get('status')}")
                              if hypo_result_final.get('status') == 'completed':
                                   hypotheses = hypo_result_final.get('result', {}).get('hypotheses', [])
                                   logger.info(f"生成された仮説数: {len(hypotheses)}")
                                   # TODO: 生成された仮説を次の improve_code_focused で使えるように self.cycle_state に保存？
                                   # await self.add_history_event("hypotheses_generated", {"hypotheses": hypotheses}, cycle_state_update=...)
                                   await self.add_history_event("hypotheses_generated", {"hypotheses": hypotheses}) # 状態更新なし
                                   action_taken = True
                              else:
                                   error_msg = hypo_result_final.get('error', '不明')
                                   logger.warning(f"仮説生成ジョブ失敗: {error_msg}")
                                   await self.add_history_event("action_failed", {"action": action_type, "reason": f"Hypothesis generation job failed: {error_msg}"})
                         else:
                              logger.warning("仮説生成ジョブ開始失敗。")
                              await self.add_history_event("action_failed", {"action": action_type, "reason": "Failed to start hypothesis generation job"})

                    elif action_type == "rollback":
                         # ロールバック
                         target_version = next_action.get('params', {}).get('version_tag')
                         if target_version:
                             logger.info(f"バージョン '{target_version}' にロールバック。")
                             try:
                                 await self.get_detector_code(detector_name, target_version) # 存在確認
                                 next_code_version = target_version # 次の評価対象を変更
                                 logger.info(f"次イテレーションでバージョン '{target_version}' を評価。")
                                 await self.add_history_event("action_taken", {"action": action_type, "version": target_version})
                                 action_taken = True
                             except Exception as ve:
                                 logger.warning(f"ロールバック対象 '{target_version}' が無効: {ve}。スキップ。")
                                 await self.add_history_event("action_skipped", {"action": action_type, "reason": f"Version '{target_version}' not found"})
                         else:
                             logger.warning("ロールバック対象バージョン指定なし。スキップ。")
                             await self.add_history_event("action_skipped", {"action": action_type, "reason": "No target version"})

                    elif action_type == "finish":
                         # 終了
                         logger.info("戦略に基づきループ終了。")
                         final_status = 'completed'
                         await self.add_history_event("loop_finished", {"reason": "Strategy decided to finish"}, cycle_state_update={'status': 'completed'})
                         action_taken = True
                         break # ループ中断

                    else:
                         # 未知のアクション
                         logger.warning(f"未定義アクション: '{action_type}'。スキップ。")
                         await self.add_history_event("action_skipped", {"action": action_type, "reason": "Unknown action"})
                         action_taken = True # 試みたとみなす

                except Exception as e:
                    logger.error(f"イテレーション {iteration + 1} のアクション実行 ({action_type}) フェーズでエラー: {e}", exc_info=True)
                    await self.add_history_event("iteration_failed", {"iteration": iteration + 1, "phase": f"action_execution ({action_type})", "error": str(e)}, cycle_state_update={'status': 'failed'})
                    self.cycle_state['status'] = 'failed'
                    final_status = 'failed'
                    break

                # --- 2.5 イテレーション完了 --- #
                iteration_duration = time.time() - iteration_start_time
                next_iteration_number = iteration + 1

                logger.info(f"イテレーション {iteration + 1} 完了。所要時間: {iteration_duration:.2f} 秒")
                # イテレーション完了をサーバーに記録 (cycle_state更新も含む)
                await self.add_history_event(
                    "iteration_completed",
                    {"iteration": iteration + 1, "duration": iteration_duration},
                    cycle_state_update={
                        'current_iteration': next_iteration_number,
                        'current_code_version': next_code_version # 次に評価するバージョンを更新
                    }
                )
                # self.cycle_state は add_history_event 内で更新される

                # セッションタイムアウトチェック
                if time.time() - start_time > self.session_timeout:
                    logger.warning(f"セッションタイムアウト ({self.session_timeout}秒) 超過。ループ終了。")
                    await self.add_history_event("loop_finished", {"reason": "Session timeout"}, cycle_state_update={'status': 'timeout'})
                    self.cycle_state['status'] = 'timeout' # クライアント側も更新
                    final_status = 'timeout'
                    break

            # --- ループ終了後 --- #
            if final_status == 'unknown': # ループが break せずに正常に完了した場合
                 final_status = 'completed'
                 logger.info(f"最大イテレーション ({effective_iterations}) 到達。改善ループ終了。")
                 self.cycle_state['status'] = 'completed'
                 await self.add_history_event("loop_finished", {"reason": "Max iterations reached"}, cycle_state_update={'status': 'completed'})

            total_duration = time.time() - start_time
            logger.info(f"=== 自動改善サイクル終了: {detector_name} ({final_status}) ===")
            # ... (ログ出力部分は変更なし) ...

        except Exception as e:
            logger.error(f"改善ループの実行中に予期せぬエラー: {e}", exc_info=True)
            final_status = 'error'
            if self.session_id and hasattr(self, 'cycle_state'):
                 self.cycle_state['status'] = 'error'; self.cycle_state['error_message'] = str(e)
                 try: await self.add_history_event("loop_error", {"error": str(e)}, cycle_state_update={'status': 'error'})
                 except Exception as log_e: logger.error(f"ループエラー記録失敗: {log_e}")
            else: logger.error("セッションIDまたはcycle_state未定義のためループエラー記録不可。")

        # --- 3. 最終結果の準備と返却 --- #
        iterations_completed = self.cycle_state.get('current_iteration', 0)
        baseline_f = self.cycle_state.get('baseline_f_measure', 0.0)
        best_f = self.cycle_state.get('best_f_measure', baseline_f)
        best_code_version = self.cycle_state.get('best_code_version')
        improvement_percentage = ((best_f - baseline_f) / baseline_f * 100) if baseline_f > 0 else 0.0

        # (オプション) 最終的な可視化を実行
        visualization_path = ""
        if final_status in ['completed', 'timeout']:
             try:
                  viz_result = await self.visualize_improvement_trajectory(self.session_id)
                  if viz_result and viz_result.get('status') == 'completed':
                       visualization_path = viz_result.get('result', {}).get('chart_path', "")
                       if visualization_path: logger.info(f"改善軌跡を可視化: {visualization_path}")
             except Exception as viz_e:
                  logger.warning(f"改善軌跡の可視化に失敗: {viz_e}")

        return {
            "status": final_status,
            "session_id": self.session_id,
            "detector_name": detector_name,
            "iterations_completed": iterations_completed,
            "baseline_f_measure": baseline_f,
            "best_f_measure": best_f,
            "improvement_percentage": improvement_percentage,
            "best_code_version": best_code_version,
            "total_duration_seconds": total_duration,
            "visualization_path": visualization_path,
            "final_cycle_state": self.cycle_state # 最終状態全体も返す
        }

    # --- プライベートヘルパーメソッド (変更なし or 軽微な修正) ---
    # _visualize_improvement_trajectory, _parse_hypothesis, _create_*_prompt, _extract_code_and_changes など
    # _restore_cycle_state, _validate_session はサーバーからのロードに置き換わるため不要になる可能性
    # _generate_parameter_grid, _run_grid_search, apply_parameter_changes は optimize_parameters ツール内で実行されるため削除可能

# --- Main Execution ---
async def main():
    """
    メイン実行関数
    """
    import argparse
    parser = argparse.ArgumentParser(description="MCP対応アルゴリズム自動改善クライアント")
    parser.add_argument("--detector", type=str, required=True, help="改善対象の検出器名")
    parser.add_argument("--goal", type=str, default=None, help="改善目標の説明")
    parser.add_argument("--iterations", type=int, default=None, help="改善サイクルの最大回数")
    parser.add_argument("--session-id", type=str, default=None, help="再開するセッションID")
    # 評価データ指定方法
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset", type=str, help="評価に使用するデータセット名 (config.yamlで定義)")
    data_group.add_argument("--audio-path", type=str, help="評価に使用する単一音声ファイルパス")
    parser.add_argument("--ref-path", type=str, help="--audio-path と併用する参照ファイルパス")
    # サーバー・設定
    parser.add_argument("--server-url", type=str, default=None, help="MCPサーバーのURL")
    parser.add_argument("--config", type=str, default=None, help="クライアント設定ファイルパス (現在は未使用)")
    args = parser.parse_args()

    # audio-path を使う場合は ref-path も必須にするバリデーション
    if args.audio_path and not args.ref_path:
        parser.error("--audio-path を指定する場合は --ref-path も必須です。")

    # 自動改善ツールの初期化
    improver = AutoImprover(config_path=args.config, server_url=args.server_url)

    try:
        # 改善サイクルの実行 (引数を渡す)
        result = await improver.run_improvement_cycle(
            detector_name=args.detector,
            goal=args.goal,
            iterations=args.iterations,
            session_id=args.session_id,
            dataset_name=args.dataset,
            audio_path=args.audio_path,
            ref_path=args.ref_path
        )

        # 結果の表示
        print("\n=== 改善サイクル結果 ===")
        print(f"ステータス: {result.get('status', '不明')}")
        print(f"セッションID: {result.get('session_id', 'N/A')}")
        print(f"検出器: {result.get('detector_name', 'N/A')}")
        print(f"完了イテレーション数: {result.get('iterations_completed', 'N/A')}")
        bf = result.get('baseline_f_measure')
        print(f"ベースラインF値: {bf:.4f}" if isinstance(bf, float) else 'N/A')
        ef = result.get('best_f_measure')
        print(f"最終ベストF値: {ef:.4f}" if isinstance(ef, float) else 'N/A')
        ip = result.get('improvement_percentage')
        print(f"改善率: {ip:.2f}%" if isinstance(ip, float) else 'N/A')
        print(f"ベストコードバージョン: {result.get('best_code_version', 'N/A')}")
        td = result.get('total_duration_seconds')
        print(f"総所要時間: {td:.2f}秒" if isinstance(td, float) else 'N/A')
        vp = result.get('visualization_path')
        if vp: print(f"可視化ファイル: {vp}")

        return 0 if result.get('status') in ['completed', 'timeout'] else 1 # 完了かタイムアウトなら成功
    except Exception as e:
        logger.error(f"実行中にエラーが発生しました: {e}", exc_info=True)
        print(f"エラー: {e}")
        return 1

if __name__ == "__main__":
    import asyncio
    # Windows での asyncio イベントループエラー対策
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.exit(asyncio.run(main()))