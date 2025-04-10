#!/usr/bin/env python3
"""
MCP サーバー拡張ツール

このスクリプトは既存の MCP サーバーに可視化と科学的発見の自動化機能を追加します。
既存のMCPサーバーと統合するために設計されています。

使用方法:
1. このファイルを既存のMCPサーバーのディレクトリに配置します。

2. MCPサーバーから以下のようにインポートします:
   ```python
   try:
       import mcp_server_extensions
       has_extensions = True
   except ImportError:
       has_extensions = False
   ```

3. サーバーの初期化後に拡張機能を登録します:
   ```python
   # MCPサーバーの初期化
   mcp = MCPServer()
   # 標準ツールの登録...
   
   # 拡張機能の登録（ある場合）
   if has_extensions:
       mcp_server_extensions.register_tools(mcp, sessions, async_jobs)
   ```

この拡張モジュールは以下の機能を追加します:
- コード変更影響の可視化
- パラメータ空間の性能ヒートマップ
- 科学的成果物（仮説、論文、実験ノート）の生成

必要条件:
- src/visualization モジュール
- src/science_automation モジュール
- 既存のMCPサーバー
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, TYPE_CHECKING, Coroutine
import time

# mcp と sessions の型ヒントのために TYPE_CHECKING を使用
if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP # 仮の型として使用
    # sessions と async_jobs の具体的な型が不明なため Any を使用
    # SessionManager = Any # 不要になる
    # AsyncJobManager = Any # 不要になる
    ConfigDict = Dict[str, Any]
    DBPath = Path
    AsyncJobFunc = Callable[..., Coroutine[Any, Any, Dict[str, Any]]]
    HistoryAddFunc = Callable[..., None]

logger = logging.getLogger('mcp_server_extensions')

# def register_tools(mcp: 'FastMCP', sessions: 'SessionManager', async_jobs: 'AsyncJobManager', config: 'ConfigDict'):
def register_extension_tools(
    mcp: 'FastMCP',
    config: 'ConfigDict',
    db_path: 'DBPath', # db_path を追加
    start_async_job_func: 'AsyncJobFunc', # ヘルパー関数を受け取る
    add_history_sync_func: 'HistoryAddFunc' # ヘルパー関数を受け取る
):
    """MCPサーバーに拡張ツールを登録する
    
    Parameters
    ----------
    mcp : FastMCP
        MCPサーバーインスタンス
    config : ConfigDict
        サーバー設定
    db_path : DBPath
        データベースファイルのパス
    start_async_job_func : AsyncJobFunc
        非同期ジョブを開始するための関数
    add_history_sync_func : HistoryAddFunc
        セッション履歴を同期的に追加するための関数
    """
    logger.info("拡張MCP機能を登録しています: 可視化と科学的発見の自動化")
    
    # get_timestamp は内部で time.time() を使うか、config等から渡す想定に変更？
    # 一旦 time.time() を直接使う
    get_timestamp = time.time 
    
    # # 非同期ジョブ管理関数 (削除: 引数で受け取る start_async_job_func を使用)
    # def start_async_job(task_func, *args, **kwargs):
    #     """非同期ジョブを開始する"""
    #     # ... (省略)
    
    @mcp.tool()
    async def visualize_code_impact(session_id: str, iteration: int) -> dict:
        """
        特定のイテレーションのコード変更影響を可視化します。
        
        Args:
            session_id: 改善セッションID
            iteration: イテレーション番号
        """
        # セッション情報の取得方法を変更 (db_utils を利用？)
        # ここでは簡略化のため、直接DBアクセスはせず、必要な情報がない場合はエラーとする
        # TODO: セッション情報を DB から取得するロジックを実装
        # 例: session_data = db_utils.get_session(db_path, session_id)
        # if not session_data: return {"error": "Session not found"}

        # 履歴情報もDBから取得する必要がある
        # history = db_utils.get_session_history(db_path, session_id)
        # original_code, improved_code = ... # 履歴から抽出
        
        # === 仮実装: 必要な情報を直接取得できないためエラーを返す ===
        logger.warning("visualize_code_impact: DBからのセッション情報取得が未実装です。")
        # return {"error": "DBからのセッション情報取得が未実装です。"}
        # === ここまで ===
        
        # --- 実際の処理（上記仮実装を削除した場合）---
        # original_code と improved_code をDBから取得した前提
        original_code = "# dummy original code"
        improved_code = "# dummy improved code"
        # --- ここまで ---

        # 可視化を実行 (渡された関数を使用)
        job_id = await start_async_job_func(
            visualize_code_impact_task, 
            "visualize_code_impact", # ツール名
            session_id, # セッションID
            original_code, improved_code, iteration,
            config, # config をタスク関数に渡す
            add_history_sync_func, # 履歴追加関数もタスクに渡す
            db_path # DBパスもタスクに渡す
        )
        
        return {"job_id": job_id, "status": "pending"}

    @mcp.tool()
    async def generate_performance_heatmap(session_id: str, param_x: str, param_y: str, metric: str) -> dict:
        """
        パラメータ空間の性能ヒートマップを生成します。
        
        Args:
            session_id: 改善セッションID
            param_x: X軸のパラメータ名
            param_y: Y軸のパラメータ名
            metric: 可視化するメトリック (例: "note_f_measure")
        """
        # セッション情報とグリッドサーチ結果をDBから取得する必要がある
        # TODO: 必要な情報を DB から取得するロジックを実装
        # session_data = db_utils.get_session(db_path, session_id)
        # grid_search_results = db_utils.get_latest_grid_search_result(db_path, session_id)
        # if not grid_search_results: return {"error": "Grid search results not found"}

        # === 仮実装 ===
        logger.warning("generate_performance_heatmap: DBからのグリッドサーチ結果取得が未実装です。")
        # return {"error": "DBからのグリッドサーチ結果取得が未実装です。"}
        grid_search_results = {} # ダミー
        # === ここまで ===
        
        # ヒートマップ生成を実行
        job_id = await start_async_job_func(
            generate_heatmap_task, 
            "generate_performance_heatmap", # ツール名
            session_id,
            grid_search_results, param_x, param_y, metric,
            config,
            add_history_sync_func,
            db_path
        )
        
        return {"job_id": job_id, "status": "pending"}

    @mcp.tool()
    async def generate_scientific_outputs(session_id: str) -> dict:
        """
        改善プロセスから科学的成果物（仮説、論文、実験ノート）を生成します。
        
        Args:
            session_id: 改善セッションID
        """
        # セッション情報をDBから取得する必要がある
        # TODO: DBからセッション情報を取得するロジック
        # session_data = db_utils.get_session(db_path, session_id)
        # if not session_data: return {"error": "Session not found"}

        # === 仮実装 ===
        logger.warning("generate_scientific_outputs: DBからのセッション情報取得が未実装です。")
        # === ここまで ===
        
        # 科学的成果物生成を実行
        job_id = await start_async_job_func(
             generate_scientific_outputs_task, 
             "generate_scientific_outputs", # ツール名
             session_id, 
             config,
             add_history_sync_func,
             db_path
         )
        
        return {"job_id": job_id, "status": "pending"}
    
    # --- 非同期タスク関数 --- 
    # タスク関数も add_history_sync_func, db_path を受け取るように修正

    async def visualize_code_impact_task(session_id, original_code, improved_code, iteration, config, add_history_sync_func, db_path):
        """コード影響可視化の実行タスク"""
        get_timestamp = time.time
        try:
            # 設定から可視化ディレクトリを取得 (config[paths] を参照する想定)
            vis_dir = Path(config.get('paths', {}).get('visualizations_dir', './visualizations'))
            # if not vis_dir:
            #     logger.error("設定に paths.visualizations_dir が見つかりません")
            #     vis_dir = Path(tempfile.gettempdir()) / f"mirex_vis_{session_id}"
            # else:
            #     vis_dir = Path(vis_dir)
            
            output_dir = vis_dir / session_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # プロジェクトルートを sys.path に追加 (config[paths][project_root] を想定？)
            project_root = Path(config.get('paths', {}).get('project_root', '.'))
            if str(project_root) not in sys.path:
                 sys.path.insert(0, str(project_root))
                 logger.debug(f"sys.path に {project_root} を追加しました")
            
            # 可視化モジュールのインポート
            try:
                from src.visualization.code_impact_graph import CodeChangeAnalyzer
            except ImportError as e:
                logger.error(f"可視化モジュール(code_impact_graph)のインポートに失敗: {e}")
                raise
            
            analyzer = CodeChangeAnalyzer()
            # analyze_code_change は同期的な処理と仮定
            # もし非同期なら await analyzer.analyze_code_change(...)
            analyzer.analyze_code_change(original_code, improved_code) 
            
            output_path = output_dir / f"code_impact_iteration_{iteration}.png"
            # visualize_impact も同期的と仮定
            analyzer.visualize_impact(str(output_path), f"Code Change Impact - Iteration {iteration}")
            
            # セッション履歴にイベントを追加 (渡された関数を使用)
            try:
                 add_history_sync_func(session_id, 'visualization_created', {
                     'type': 'code_impact_graph',
                     'iteration': iteration,
                     'path': str(output_path) # 文字列で保存
                 })
                 logger.debug(f"セッション {session_id} に visualization_created イベントを追加")
            except Exception as e_session:
                 logger.warning(f"セッション履歴への追加中にエラー: {e_session}")
            
            # await asyncio.sleep(1) # 仮の非同期処理
            return {"status": "completed", "output_path": str(output_path)}
        except Exception as e:
            logger.error(f"コード影響可視化タスク失敗: {e}", exc_info=True)
            # エラーを履歴に追加
            try:
                add_history_sync_func(session_id, 'task_error', {
                    'task_name': 'visualize_code_impact_task',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
            except Exception as e_hist:
                logger.error(f"Failed to log task error to history: {e_hist}")
            return {"status": "failed", "error": str(e)} # ジョブ管理システムにエラーを返す

    async def generate_heatmap_task(session_id, grid_search_results, param_x, param_y, metric, config, add_history_sync_func, db_path):
        """性能ヒートマップ生成の実行タスク"""
        get_timestamp = time.time
        try:
            vis_dir = Path(config.get('paths', {}).get('visualizations_dir', './visualizations'))
            output_dir = vis_dir / session_id
            output_dir.mkdir(parents=True, exist_ok=True)

            project_root = Path(config.get('paths', {}).get('project_root', '.'))
            if str(project_root) not in sys.path:
                 sys.path.insert(0, str(project_root))

            try:
                from src.visualization.grid_search_plots import plot_parameter_heatmap
            except ImportError as e:
                logger.error(f"可視化モジュール(grid_search_plots)のインポートに失敗: {e}")
                raise

            # plot_parameter_heatmap が期待するデータ形式に grid_search_results を変換する必要があるかもしれない
            # ここでは grid_search_results['results'] が DataFrame またはそれに準ずる形式と仮定
            results_data = grid_search_results.get('results') 
            if results_data is None:
                 raise ValueError("グリッドサーチ結果が見つかりません (results キー)")
            
            # 必要なら DataFrame に変換
            # import pandas as pd
            # if not isinstance(results_data, pd.DataFrame):
            #     results_data = pd.DataFrame(results_data)

            output_path = output_dir / f"heatmap_{param_x}_vs_{param_y}_for_{metric}.png"
            
            # plot_parameter_heatmap を呼び出す (同期的と仮定)
            plot_parameter_heatmap(
                results_data=results_data,
                param_x=param_x,
                param_y=param_y,
                metric=metric,
                save_path=str(output_path)
            )

            # セッション履歴にイベントを追加
            try:
                add_history_sync_func(session_id, 'visualization_created', {
                    'type': 'performance_heatmap',
                    'param_x': param_x,
                    'param_y': param_y,
                    'metric': metric,
                    'path': str(output_path)
                })
                logger.debug(f"セッション {session_id} に visualization_created イベントを追加")
            except Exception as e_session:
                logger.warning(f"セッション履歴への追加中にエラー: {e_session}")

            # await asyncio.sleep(1) # 仮の非同期処理
            return {"status": "completed", "output_path": str(output_path)}
        except Exception as e:
            logger.error(f"性能ヒートマップ生成タスク失敗: {e}", exc_info=True)
            try:
                add_history_sync_func(session_id, 'task_error', {
                    'task_name': 'generate_heatmap_task',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
            except Exception as e_hist:
                logger.error(f"Failed to log task error to history: {e_hist}")
            return {"status": "failed", "error": str(e)}

    async def generate_scientific_outputs_task(session_id, config, add_history_sync_func, db_path):
        """科学的成果物生成の実行タスク"""
        get_timestamp = time.time
        try:
            # 設定から出力ディレクトリを取得
            output_base_dir = Path(config.get('paths', {}).get('scientific_output_dir', './scientific_output'))
            session_output_dir = output_base_dir / session_id
            session_output_dir.mkdir(parents=True, exist_ok=True)

            project_root = Path(config.get('paths', {}).get('project_root', '.'))
            if str(project_root) not in sys.path:
                 sys.path.insert(0, str(project_root))

            # 科学的自動化モジュールのインポート
            try:
                from src.science_automation.reporting import generate_experiment_report, generate_hypothesis_summary
                from src.science_automation.paper_generation import generate_paper_outline # 仮の関数
            except ImportError as e:
                logger.error(f"科学的自動化モジュールのインポートに失敗: {e}")
                raise

            # DBから必要な情報を取得 (セッションデータ、履歴、評価結果など)
            # TODO: DBアクセスロジックを実装
            # session_data = db_utils.get_session(db_path, session_id)
            # history = db_utils.get_session_history(db_path, session_id)
            # evaluation_results = db_utils.get_evaluation_results(db_path, session_id)
            # hypotheses = [event['data'] for event in history if event['type'] == 'hypothesis_generated']
            session_data = {'base_algorithm': 'DummyDetector'} # ダミー
            hypotheses = [{'text': 'Dummy hypothesis'}] # ダミー
            evaluation_results = [] # ダミー

            # 成果物生成
            report_path = session_output_dir / "experiment_report.md"
            hypothesis_path = session_output_dir / "hypothesis_summary.md"
            paper_outline_path = session_output_dir / "paper_outline.md"

            # generate_experiment_report (同期的と仮定)
            generate_experiment_report(session_data, evaluation_results, str(report_path))
            # generate_hypothesis_summary (同期的と仮定)
            generate_hypothesis_summary(hypotheses, str(hypothesis_path))
            # generate_paper_outline (同期的と仮定)
            generate_paper_outline(session_data, evaluation_results, hypotheses, str(paper_outline_path))

            output_paths = {
                "experiment_report": str(report_path),
                "hypothesis_summary": str(hypothesis_path),
                "paper_outline": str(paper_outline_path)
            }

            # セッション履歴にイベントを追加
            try:
                add_history_sync_func(session_id, 'scientific_outputs_generated', {
                    'paths': output_paths
                })
                logger.debug(f"セッション {session_id} に scientific_outputs_generated イベントを追加")
            except Exception as e_session:
                logger.warning(f"セッション履歴への追加中にエラー: {e_session}")

            # await asyncio.sleep(1) # 仮の非同期処理
            return {"status": "completed", "output_paths": output_paths}
        except Exception as e:
            logger.error(f"科学的成果物生成タスク失敗: {e}", exc_info=True)
            try:
                add_history_sync_func(session_id, 'task_error', {
                    'task_name': 'generate_scientific_outputs_task',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
            except Exception as e_hist:
                logger.error(f"Failed to log task error to history: {e_hist}")
            return {"status": "failed", "error": str(e)}
    
    logger.info("拡張MCP機能を登録しました: 可視化と科学的発見の自動化")
    
    # 登録したツールを返す
    return {
        "visualize_code_impact": visualize_code_impact,
        "generate_performance_heatmap": generate_performance_heatmap,
        "generate_scientific_outputs": generate_scientific_outputs
    } 