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
from typing import Dict, List, Any, Callable, Optional, TYPE_CHECKING
import time

# mcp と sessions の型ヒントのために TYPE_CHECKING を使用
if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP # 仮の型として使用
    # sessions と async_jobs の具体的な型が不明なため Any を使用
    SessionManager = Any
    AsyncJobManager = Any 
    ConfigDict = Dict[str, Any]

logger = logging.getLogger('mcp_server_extensions')

def register_tools(mcp: 'FastMCP', sessions: 'SessionManager', async_jobs: 'AsyncJobManager', config: 'ConfigDict'):
    """MCPサーバーに拡張ツールを登録する
    
    Parameters
    ----------
    mcp : MCPServer
        MCPサーバーインスタンス
    sessions : SessionManager
        セッション管理オブジェクト
    async_jobs : AsyncJobManager
        非同期ジョブ管理オブジェクト
    config : ConfigDict
        サーバー設定（書き込み可能パスなどを含む）
    
    Returns
    -------
    dict
        登録されたツールの辞書
    """
    logger.info("拡張MCP機能を登録しています: 可視化と科学的発見の自動化")
    
    # get_timestampの取得 (mcp_server.pyから直接importできれば最適)
    get_timestamp = getattr(sessions, "get_timestamp", lambda: getattr(async_jobs, "get_timestamp", lambda: time.time())())
    
    # 非同期ジョブ管理関数
    def start_async_job(task_func, *args, **kwargs):
        """非同期ジョブを開始する"""
        # async_jobsが辞書の場合（mcp_server.pyからの{"start_job": start_async_job}形式）
        if isinstance(async_jobs, dict) and "start_job" in async_jobs:
            return async_jobs["start_job"](task_func, *args, **kwargs)
        # async_jobsがオブジェクトでstart_job属性を持つ場合
        elif hasattr(async_jobs, "start_job"):
            return async_jobs.start_job(task_func, *args, **kwargs)
        # 従来のインタフェース
        else:
            return async_jobs.start_job(task_func, *args, **kwargs)
    
    @mcp.tool()
    async def visualize_code_impact(session_id: str, iteration: int) -> dict:
        """
        特定のイテレーションのコード変更影響を可視化します。
        
        Args:
            session_id: 改善セッションID
            iteration: イテレーション番号
        """
        # セッション情報を取得
        session = sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        # イテレーションの改善前後のコードを取得
        original_code = None
        improved_code = None
        
        for event in session.get('history', []):
            if event.get('type') == 'iteration_start' and event.get('data', {}).get('iteration') == iteration:
                # このイテレーションのコード情報を探す
                idx = session['history'].index(event)
                for i in range(idx, len(session['history'])):
                    e = session['history'][i]
                    if e.get('type') == 'code_before_improvement':
                        original_code = e.get('data', {}).get('code')
                    elif e.get('type') == 'code_after_improvement':
                        improved_code = e.get('data', {}).get('code')
        
        if not original_code or not improved_code:
            return {"error": "Code not found for this iteration"}
        
        # 可視化を実行
        job_id = start_async_job(
            visualize_code_impact_task, 
            session_id, original_code, improved_code, iteration,
            config # config をタスク関数に渡す
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
        session = sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        # グリッドサーチの結果を取得
        grid_search_results = None
        for event in session.get('history', []):
            if event.get('type') == 'grid_search_complete':
                grid_search_results = event.get('data', {})
        
        if not grid_search_results:
            return {"error": "No grid search results found for this session"}
        
        # ヒートマップ生成を実行
        job_id = start_async_job(
            generate_heatmap_task, 
            session_id, grid_search_results, param_x, param_y, metric,
            config # config をタスク関数に渡す
        )
        
        return {"job_id": job_id, "status": "pending"}

    @mcp.tool()
    async def generate_scientific_outputs(session_id: str) -> dict:
        """
        改善プロセスから科学的成果物（仮説、論文、実験ノート）を生成します。
        
        Args:
            session_id: 改善セッションID
        """
        session = sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        # 科学的成果物生成を実行
        job_id = start_async_job(
             generate_scientific_outputs_task, 
             session_id, 
             config # config をタスク関数に渡す
         )
        
        return {"job_id": job_id, "status": "pending"}
    
    # 非同期タスク関数
    def visualize_code_impact_task(session_id, original_code, improved_code, iteration, config):
        """コード影響可視化の実行タスク"""
        try:
            # 設定から可視化ディレクトリを取得
            vis_dir = config.get('visualizations_dir')
            if not vis_dir:
                logger.error("設定に visualizations_dir が見つかりません")
                # フォールバックとして一時ディレクトリを使用
                vis_dir = Path(tempfile.gettempdir()) / f"mirex_vis_{session_id}"
            else:
                # Pathオブジェクトに変換
                vis_dir = Path(vis_dir)
            
            output_dir = vis_dir / session_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # プロジェクトルートを sys.path に追加 (必要に応じて)
            project_root = config.get('workspace_dir') # workspace_dir をプロジェクトルートと仮定
            if project_root and str(project_root) not in sys.path:
                 sys.path.insert(0, str(project_root))
                 logger.debug(f"sys.path に {project_root} を追加しました")
            
            # 可視化モジュールのインポート
            # 注意: モジュールの場所が workspace_dir/src であることを前提とする
            try:
                from src.visualization.code_impact_graph import CodeChangeAnalyzer
            except ImportError as e:
                logger.error(f"可視化モジュール(code_impact_graph)のインポートに失敗: {e}")
                raise
            
            analyzer = CodeChangeAnalyzer()
            analyzer.analyze_code_change(original_code, improved_code)
            
            output_path = output_dir / f"code_impact_iteration_{iteration}.png"
            analyzer.visualize_impact(str(output_path), f"Code Change Impact - Iteration {iteration}")
            
            # セッション履歴にイベントを追加 (sessions オブジェクトに save_session があるか不明なため try-except)
            try:
                 session = sessions.get(session_id)
                 if session and 'history' in session:
                     session['history'].append({
                         'type': 'visualization_created',
                         'data': {
                             'type': 'code_impact_graph',
                             'iteration': iteration,
                             'path': str(output_path) # 文字列で保存
                         },
                         'timestamp': get_timestamp()
                     })
                     # sessions.save_session(session_id, session) # save_session の存在が不明なためコメントアウト
                 logger.debug(f"セッション {session_id} に visualization_created イベントを追加 (保存は未実行)")
            except Exception as e_session:
                 logger.warning(f"セッション履歴への追加中にエラー: {e_session}")
            
            return {"status": "success", "output_path": str(output_path)}
        except Exception as e:
            logger.error(f"コード影響可視化に失敗: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def generate_heatmap_task(session_id, grid_search_results, param_x, param_y, metric, config):
        """ヒートマップ生成の実行タスク"""
        try:
            # 設定から可視化ディレクトリを取得
            vis_dir = config.get('visualizations_dir')
            if not vis_dir:
                logger.error("設定に visualizations_dir が見つかりません")
                vis_dir = Path(tempfile.gettempdir()) / f"mirex_vis_{session_id}"
            else:
                vis_dir = Path(vis_dir)
            
            output_dir = vis_dir / session_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # プロジェクトルートを sys.path に追加
            project_root = config.get('workspace_dir')
            if project_root and str(project_root) not in sys.path:
                 sys.path.insert(0, str(project_root))
                 logger.debug(f"sys.path に {project_root} を追加しました")
            
            # 可視化モジュールのインポート
            try:
                from src.visualization.performance_heatmap import PerformanceHeatmap
            except ImportError as e:
                logger.error(f"可視化モジュール(performance_heatmap)のインポートに失敗: {e}")
                raise
            
            # grid_search_results から 'all_results' を安全に取得
            all_results_data = grid_search_results.get('result', {}).get('all_results') if grid_search_results.get('result') else None
            if not all_results_data:
                 # 'result' がない場合、直接 'all_results' を試す (古い形式かもしれないため)
                 all_results_data = grid_search_results.get('all_results', [])
                 if not all_results_data:
                      logger.warning("ヒートマップ生成のためのグリッドサーチ結果 ('all_results') が見つかりません。")
                      return {"status": "error", "message": "Grid search result data ('all_results') not found or empty."} 

            heatmap = PerformanceHeatmap(all_results_data)
            
            output_path = output_dir / f"heatmap_{param_x}_{param_y}_{metric}.png"
            heatmap.create_heatmap(param_x, param_y, metric, str(output_path))
            
            # セッション履歴にイベントを追加 (sessions オブジェクトに save_session があるか不明なため try-except)
            try:
                 session = sessions.get(session_id)
                 if session and 'history' in session:
                     session['history'].append({
                         'type': 'visualization_created',
                         'data': {
                             'type': 'performance_heatmap',
                             'params': [param_x, param_y],
                             'metric': metric,
                             'path': str(output_path) # 文字列で保存
                         },
                         'timestamp': get_timestamp()
                     })
                     # sessions.save_session(session_id, session) # save_session の存在が不明なためコメントアウト
                 logger.debug(f"セッション {session_id} に visualization_created イベントを追加 (保存は未実行)")
            except Exception as e_session:
                 logger.warning(f"セッション履歴への追加中にエラー: {e_session}")
            
            return {"status": "success", "output_path": str(output_path)}
        except Exception as e:
            logger.error(f"ヒートマップ生成に失敗: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def generate_scientific_outputs_task(session_id, config):
        """科学的成果物生成の実行タスク"""
        try:
            # 設定から科学的成果物出力ディレクトリを取得
            sci_output_dir = config.get('scientific_output_dir')
            if not sci_output_dir:
                 logger.error("設定に scientific_output_dir が見つかりません")
                 sci_output_dir = Path(tempfile.gettempdir()) / f"mirex_sci_{session_id}"
            else:
                 sci_output_dir = Path(sci_output_dir)

            output_dir = sci_output_dir / session_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # プロジェクトルートを sys.path に追加
            project_root = config.get('workspace_dir')
            if project_root and str(project_root) not in sys.path:
                 sys.path.insert(0, str(project_root))
                 logger.debug(f"sys.path に {project_root} を追加しました")
                 
            # 科学的発見モジュールのインポート
            try:
                 from src.science_automation.hypothesis_generator import HypothesisGenerator
                 from src.science_automation.paper_generator import PaperGenerator
                 from src.science_automation.research_notebook import ResearchNotebook
            except ImportError as e:
                 logger.error(f"科学的発見モジュールのインポートに失敗: {e}")
                 raise
            
            # セッションデータの取得
            session_data = sessions.get(session_id, {})
            if not session_data:
                 logger.warning(f"科学的成果物生成のためのセッションデータが見つかりません: {session_id}")
                 return {"status": "error", "message": f"Session data not found for {session_id}"}
            
            output_paths = {}
            
            # 1. 仮説生成
            hypothesis_generator = HypothesisGenerator(session_data.get('history', []))
            hypotheses = hypothesis_generator.generate_hypotheses()
            
            hypothesis_path = output_dir / "hypotheses.json"
            with open(hypothesis_path, 'w', encoding='utf-8') as f:
                json.dump(hypotheses, f, indent=2, ensure_ascii=False)
            
            output_paths['hypotheses'] = str(hypothesis_path)
            logger.info(f"仮説を生成しました: {hypothesis_path}")
            
            # 2. 論文ドラフト生成
            paper_generator = PaperGenerator()
            paper_path = paper_generator.generate_paper(
                session_id=session_id,
                session_data=session_data,
                best_version_info=session_data.get('best_version_info', {}),
                hypothesis_data=hypotheses,
                output_path=str(output_dir / "paper_draft.md") # Path を文字列に変換
            )
            
            output_paths['paper'] = paper_path # generate_paper がパス文字列を返すと仮定
            logger.info(f"論文ドラフトを生成しました: {paper_path}")
            
            # 3. 実験ノート生成
            notebook_generator = ResearchNotebook()
            notebook_path = notebook_generator.generate_notebook(
                session_id=session_id,
                session_data=session_data,
                output_path=str(output_dir / "experiment_notebook.md") # Path を文字列に変換
            )
            
            output_paths['notebook'] = notebook_path # generate_notebook がパス文字列を返すと仮定
            logger.info(f"実験ノートを生成しました: {notebook_path}")
            
            # セッション履歴にイベントを追加 (sessions オブジェクトに save_session があるか不明なため try-except)
            try:
                 session = sessions.get(session_id)
                 if session and 'history' in session:
                     session['history'].append({
                         'type': 'scientific_output_generated',
                         'data': {
                             'output_dir': str(output_dir),
                             'files': output_paths
                         },
                         'timestamp': get_timestamp()
                     })
                     # sessions.save_session(session_id, session) # save_session の存在が不明なためコメントアウト
                 logger.debug(f"セッション {session_id} に scientific_output_generated イベントを追加 (保存は未実行)")
            except Exception as e_session:
                 logger.warning(f"セッション履歴への追加中にエラー: {e_session}")
            
            return {
                "status": "success", 
                "output_dir": str(output_dir),
                "files": output_paths
            }
        except Exception as e:
            logger.error(f"科学的成果物生成に失敗: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    logger.info("拡張MCP機能を登録しました: 可視化と科学的発見の自動化")
    
    # 登録したツールを返す
    return {
        "visualize_code_impact": visualize_code_impact,
        "generate_performance_heatmap": generate_performance_heatmap,
        "generate_scientific_outputs": generate_scientific_outputs
    } 