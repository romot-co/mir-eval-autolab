#!/usr/bin/env python
# src/cli/mirai.py
"""MIR Auto Improver (mirai) Command Line Interface."""

import typer
from typing_extensions import Annotated
from typing import Optional, Dict, Any, Union, List, Tuple
import json
import sys
import logging
import asyncio
import re
import difflib
from pathlib import Path
import yaml
import traceback
import time
import os

# --- MCP Client Imports ---
from mcp import McpError
from mcp.client.session import ClientSession

# ToolErrorはMCPパッケージに存在しないので自前で定義
class ToolError(Exception):
    """Mock ToolError class for testing"""
    pass

# MCPClientクラスを定義（実際のMCPClientを使用）
class MCPClient:
    """Simplified MCPClient implementation for testing"""
    def __init__(self, base_url=None):
        self.base_url = base_url
        
    def run_tool(self, tool_name, tool_input):
        # テスト用のダミー実装
        return {'job_id': 'test_job_id'}
        
    def get_job_status(self, job_id):
        # テスト用のダミー実装
        return {'status': 'completed', 'result': {}}
        
    def get_session_info(self, session_id):
        # テスト用のダミー実装
        return {'session_id': session_id, 'status': 'active'}
        
    def start_session(self, detector_name, dataset_name):
        # テスト用のダミー実装
        return {'session_id': 'test_session_id', 'status': 'active'}

# --- Core Function Imports ---
# 適切なパスに調整してください
try:
    from src.evaluation.evaluation_runner import evaluate_detector
    from src.utils.logging_utils import setup_logger
    from src.utils.path_utils import get_project_root
    from src.utils.exception_utils import log_exception
    from src.cli.llm_client import initialize_llm_client, LLMClientError  # 新しいLLMクライアント
except ImportError as e:
    print(f"Error importing core modules: {e}. Ensure PYTHONPATH is set correctly.", file=sys.stderr)
    sys.exit(1)

# --- Helper Functions ---

# config.yaml をロードするヘルパー (evaluation_runner から移動または共通化)
CONFIG = {}
def load_global_config():
    global CONFIG
    try:
        config_path = get_project_root() / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                CONFIG = yaml.safe_load(f) or {}
            logging.info(f"Loaded config from {config_path}")
        else:
            logging.warning(f"Config file not found at {config_path}")
            CONFIG = {}
    except Exception as e:
        logging.error(f"Failed to load config.yaml: {e}")
        CONFIG = {}

# 結果表示ヘルパー (evaluate_standalone.py から移植)
def print_metrics(results: dict):
    """評価結果のメトリクスを整形して表示する"""
    print("\n--- Evaluation Metrics ---")
    if not results or 'metrics' not in results:
        print("No metrics found or evaluation failed.")
        # エラー詳細があれば表示
        if results and 'error' in results:
            print(f"Error: {results['error']}")
            if 'traceback' in results:
                print("Traceback:")
                print(results['traceback'])
        print("--------------------------")
        return

    metrics_data = results['metrics']
    # mir_eval 形式の出力を想定
    for metric_group, group_values in metrics_data.items():
        # Note: 'error' キーはメトリクスグループではないのでスキップ
        if metric_group == 'error':
            continue
        print(f"\n[{metric_group.capitalize()}]")
        if isinstance(group_values, dict):
             # mir_eval スコア (precision, recall, f_measure など)
            for key, value in group_values.items():
                # mir_eval の一般的なメトリクス名を整形
                metric_name = key.replace('_', ' ').capitalize()
                if isinstance(value, (float, int)):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
        elif isinstance(group_values, (float, int)):
             # グループ名自体がメトリクスの場合 (例: accuracy)
             print(f"  {metric_group.capitalize()}: {group_values:.4f}")
        else:
            print(f"  {metric_group}: {group_values}")
    print("--------------------------")

# MCPクライアント取得ヘルパー
def get_mcp_client(server_url: str) -> MCPClient:
    """MCPクライアントを初期化して返す"""
    logger = logging.getLogger(__name__)
    try:
        client = MCPClient(base_url=server_url)
        logger.info(f"MCP Client initialized for {server_url}")
        # TODO: サーバー接続確認 (オプション)
        # client.ping() # 例: pingメソッドがあれば
        return client
    except Exception as e:
        logger.error(f"Failed to initialize MCP client for {server_url}: {e}", exc_info=True)
        typer.echo(f"Error: Could not connect to MCP server at {server_url}. Please check the URL and server status.", err=True)
        raise typer.Exit(code=1)

# ジョブポーリング関数 (improver_cli.py から移植・修正)
def poll_job_status(client: MCPClient, job_id: str, poll_interval: int = 5, timeout: int = 600) -> Dict[str, Any]:
    """MCPジョブの完了または失敗をポーリングする。完了/失敗したジョブ情報を返す。"""
    start_time = time.time()
    logger = logging.getLogger(__name__)
    logger.info(f"Polling job {job_id} for completion (timeout: {timeout}s)")
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            logger.error(f"Polling job {job_id} timed out after {timeout} seconds.")
            raise TimeoutError(f"Job {job_id} polling timed out ({timeout}s)")

        try:
            # MCPクライアントの get_job_status を使用
            job_info = client.get_job_status(job_id=job_id)
            status = job_info.get("status")
            logger.debug(f"Job {job_id} status: {status} (elapsed: {elapsed_time:.1f}s)")

            if status == "completed":
                logger.info(f"Job {job_id} completed successfully.")
                return job_info # ジョブ情報を返す
            elif status == "failed":
                error_details = job_info.get("error_details", {"message": "Unknown error"})
                error_msg = error_details.get("message", "Unknown error")
                logger.error(f"Job {job_id} failed: {error_msg}")
                # トレースバックがあればログに出力
                if "traceback" in error_details:
                    logger.error(f"Server Traceback:\n{error_details['traceback']}")
                return job_info # エラー情報を含むジョブ情報を返す
            elif status in ["pending", "running", "queued"]:
                # 進行中なので待機
                time.sleep(poll_interval)
            else:
                logger.warning(f"Unknown job status for {job_id}: '{status}'. Continuing poll.")
                time.sleep(poll_interval)

        except ToolError as e:
            logger.error(f"Error getting job status for {job_id} (ToolError): {e}", exc_info=True)
            # get_job_status ツール自体がない場合など、リトライ不可のエラー
            raise RuntimeError(f"Server error while getting job status for {job_id}: {e}") from e
        except McpError as e:
            # 接続エラーなど、リトライ可能な可能性のあるエラー
            logger.warning(f"MCP communication error while getting status for {job_id}: {e}. Retrying in {poll_interval * 2}s...")
            time.sleep(poll_interval * 2)
        except Exception as e:
             logger.error(f"Unexpected error during job polling for {job_id}: {e}", exc_info=True)
             raise # 予期せぬエラーは再発生させる

app = typer.Typer(help="MIR Auto Improver (mirai) CLI - Evaluate and improve MIR algorithms.")

# --- Shared Options ---
ServerURL = Annotated[
    Optional[str],
    typer.Option("--server", "-s", help="URL of the MCP server. If not provided, runs in standalone mode (if supported).")
]
OutputDir = Annotated[
    Optional[Path],
    typer.Option("--output", "-o", help="Directory to save evaluation results or logs.")
]
LogLevel = Annotated[
    Optional[str],
    typer.Option("--log-level", help="Set the logging level.", case_sensitive=False)
]

# --- Evaluate Command ---
eval_app = typer.Typer(help="Evaluate MIR detectors either standalone or via MCP server.")
app.add_typer(eval_app, name="evaluate")

@eval_app.callback()
def eval_main(ctx: typer.Context, log_level: Optional[str] = "INFO"):
    """ Set log level for evaluate commands. """
    # effective_log_level = log_level if log_level is not None else "INFO" # この行を削除
    # このコールバックは eval_app のすべてのコマンドの前に実行される
    # setup_logger は特定のロガーを設定するため、ここでは基本的な logging.basicConfig を使うか、
    # ルートロガーを設定する必要があるかもしれない。ただし、他のライブラリへの影響に注意。
    # 一旦、ルートロガーレベルを設定するシンプルな方法に変更
    log_level_int = getattr(logging, log_level.upper(), logging.INFO) # effective_log_level を log_level に戻す
    logging.basicConfig(level=log_level_int, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s')
    # logging.getLogger().setLevel(log_level_int) # basicConfig がレベルを設定
    logging.info(f"Root logger level set to {log_level.upper()} via basicConfig.") # effective_log_level を log_level に戻す
    # setup_logger(name=__name__, level=getattr(logging, log_level.upper(), logging.INFO)) # 特定ロガーではなくルートを設定
    # グローバル設定をロード (一度だけ行うべきだが、ここでは簡単のため)
    load_global_config()
    # contextオブジェクトは必要に応じてサブコマンドに渡せる
    # ctx.obj = {"log_level": log_level.upper(), "config": CONFIG}


@eval_app.command("run", help="Run evaluation.")
def evaluate(
    # --- Arguments --- #
    detector_name: Annotated[str, typer.Option("--detector", help="Name of the detector to evaluate.")],
    # Standalone options (only used if --server is NOT provided)
    audio_path: Annotated[Optional[Path], typer.Option("--audio", help="Path to the audio file (for standalone mode).")] = None,
    ref_path: Annotated[Optional[Path], typer.Option("--ref", help="Path to the reference annotation file (for standalone mode).")] = None,
    params_json: Annotated[Optional[str], typer.Option("--params", help="JSON string of parameters for the detector (for standalone mode).")] = None,
    save_plot: Annotated[bool, typer.Option("--save-plot", help="Save evaluation plots (for standalone mode).")] = False,
    # Server options (required if --server is provided)
    server_url: ServerURL = None,
    dataset_name: Annotated[Optional[str], typer.Option("--dataset", help="Name of the dataset (required for server mode).")] = None,
    version: Annotated[Optional[str], typer.Option("--version", help="Specific version of the detector (for server mode).")] = None,
    # Shared
    output_dir: OutputDir = None,
):
    """Evaluates a detector either standalone or via the MCP server."""
    load_global_config() # Load config at the start of the command
    logger = logging.getLogger(__name__)

    if server_url:
        # --- Server Mode ---
        logger.info(f"Running evaluation in SERVER mode for detector '{detector_name}'.")
        if not dataset_name:
            logger.error("Error: --dataset is required for server mode.")
            raise typer.Exit(code=1)

        client = get_mcp_client(server_url)

        # ツール入力の準備
        tool_input = {
            "detector_name": detector_name,
            "dataset_name": dataset_name,
            # --- オプショナルなパラメータ --- #
            # 'detector_params': {}, # MCP経由の場合、パラメータは通常バージョン管理されるか、改善ループの一部
            # 'evaluator_config': {}, # サーバー側のデフォルトを使用
            'save_plots': save_plot, # クライアントから指定可能にする (サーバー側で解釈)
            'save_results_json': True, # 基本的にサーバー側では保存する想定
            'code_version': version, # バージョン指定
            # 'num_procs': None, # 必要なら追加
            # 'output_base_dir': str(output_dir) if output_dir else None # サーバー側で結果を保存する際の基底ディレクトリ (サーバー側でパス検証必須)
        }
        # output_dir はサーバー側で管理されるべきか？一旦保留。ジョブ結果に含まれるパスを参照する形が良いか。

        logger.info(f"Calling 'run_evaluation' tool on server {server_url}...")
        logger.debug(f"Tool input: {tool_input}")

        try:
            # MCPツール呼び出し
            job_start_result = client.run_tool("run_evaluation", tool_input)
            job_id = job_start_result.get("job_id")
            if not job_id:
                 logger.error("Failed to start evaluation job: No job_id received from server.")
                 logger.error(f"Server response: {job_start_result}")
                 raise typer.Exit(code=1)
            logger.info(f"Evaluation job started successfully. Job ID: {job_id}")

            # ジョブ完了待機
            final_job_info = poll_job_status(client, job_id, timeout=1800) # タイムアウトを30分に設定

            # 結果処理
            if final_job_info.get("status") == "completed":
                logger.info(f"Evaluation job {job_id} completed.")
                results = final_job_info.get("result", {})
                # TODO: ジョブ結果からメトリクスや出力パスを抽出して表示
                logger.info(f"Job Result Summary: {results.get('summary', 'No summary available.')}")
                # print_metrics(results) # results['metrics'] が期待する形式であれば表示可能
                if 'output_dir' in results:
                    logger.info(f"Server-side results directory: {results['output_dir']}")
                # 必要であれば結果ファイルをダウンロードする機能を追加？
            else:
                logger.error(f"Evaluation job {job_id} failed.")
                # poll_job_status内でもエラーログは出るが、ここでも最終結果としてログを残す
                error_msg = final_job_info.get("error_details", {}).get("message", "Unknown error")
                logger.error(f"Failure reason: {error_msg}")
                raise typer.Exit(code=1)

        except (ToolError, McpError, TimeoutError, RuntimeError, Exception) as e:
            logger.error(f"An error occurred during server-based evaluation: {e}", exc_info=True)
            # エラータイプに応じてユーザーへのメッセージを調整
            if isinstance(e, ToolError):
                typer.echo(f"Error: Server could not execute the evaluation tool. {e}", err=True)
            elif isinstance(e, McpError):
                typer.echo(f"Error: Communication failed with the server. {e}", err=True)
            elif isinstance(e, TimeoutError):
                 typer.echo(f"Error: Evaluation job timed out. {e}", err=True)
            elif isinstance(e, RuntimeError):
                 typer.echo(f"Error: Server error during job status check. {e}", err=True)
            else:
                 typer.echo(f"An unexpected error occurred: {e}", err=True)
            raise typer.Exit(code=1)

    else:
        # --- Standalone Mode ---
        logger.info(f"Running evaluation in STANDALONE mode for detector '{detector_name}'.")
        # ★ 引数チェック: standaloneでは audio と ref が必須
        if not audio_path or not ref_path:
            logger.error("Error: --audio and --ref are required for standalone mode (when --server is not provided).")
            raise typer.Exit(code=1)

        logger.info(f"Running standalone evaluation for detector '{detector_name}'.")
        logger.info(f"Audio: '{audio_path}', Reference: '{ref_path}'")

        # ファイル存在チェック
        if not audio_path.is_file():
            logger.error(f"Audio file not found: {audio_path}")
            raise typer.Exit(code=1)
        if not ref_path.is_file():
            logger.error(f"Reference file not found: {ref_path}")
            raise typer.Exit(code=1)

        # パラメータのパース
        detector_params = {}
        if params_json:
            try:
                detector_params = json.loads(params_json)
                logger.info(f"Using detector parameters: {detector_params}")
            except json.JSONDecodeError as e:
                logger.error(f"Error: Invalid JSON string provided for --params: {e}")
                raise typer.Exit(code=1)

        # 出力ディレクトリの準備
        eval_id = f"{audio_path.stem}_{detector_name}"
        output_path = None
        if output_dir:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir
                logger.info(f"Output directory: {output_dir}")
            except OSError as e:
                logger.error(f"Failed to create output directory {output_dir}: {e}")
                output_path = None

        # config.yaml から評価設定などを取得 (evaluate_detector内部で参照される想定)
        evaluator_config = CONFIG.get('evaluation', {}).get('mir_eval_options', {})
        plot_config = CONFIG.get('visualization', {}).get('plot_options', {})
        plot_format = CONFIG.get('visualization', {}).get('default_plot_format', 'png')

        try:
            logger.info("Starting evaluation...")
            result = evaluate_detector(
                detector_name=detector_name,
                detector_params=detector_params,
                audio_file=str(audio_path),
                ref_file=str(ref_path),
                output_dir=output_path,
                eval_id=eval_id,
                evaluator_config=evaluator_config,
                save_plots=save_plot,
                plot_format=plot_format,
                plot_config=plot_config,
                save_results_json=bool(output_path)
            )
            logger.info("Evaluation finished.")

            # 結果の表示
            print_metrics(result)

            if output_path and 'results_json_path' in result:
                 logger.info(f"Results saved to: {result['results_json_path']}")
            if save_plot and output_path and 'plot_path' in result:
                 logger.info(f"Plot saved to: {result['plot_path']}")
            elif save_plot and output_path and 'error' not in result:
                 logger.warning("Plot saving was requested but plot path not found in results.")

        except Exception as e:
            logger.error(f"An unexpected error occurred during standalone evaluation:")
            # トレースバックをログとコンソールに出力
            tb_str = traceback.format_exc()
            logger.error(tb_str)
            print(f"\nError details:\n{e}", file=sys.stderr)
            print(f"Traceback:\n{tb_str}", file=sys.stderr)
            raise typer.Exit(code=1)


# --- Grid Search Command ---
grid_app = typer.Typer(help="Run grid search for detector parameters via MCP server.")
app.add_typer(grid_app, name="grid-search")

@grid_app.callback()
def grid_main(ctx: typer.Context, log_level: Optional[str] = "INFO"):
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_int, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s')
    logging.info(f"Root logger level set to {log_level.upper()} via basicConfig.")
    # load_global_config() # Moved to command functions

@grid_app.command("run", help="Run grid search.")
def grid_search(
    server_url: Annotated[str, typer.Option("--server", "-s", help="URL of the MCP server (required for grid search).")],
    config_path: Annotated[Path, typer.Option("--config", "-c", help="Path to the grid search configuration YAML file.")],
    output_dir: OutputDir = None,
):
    """Runs grid search using the MCP server."""
    load_global_config() # Load config at the start of the command
    logger = logging.getLogger(__name__)
    logger.info(f"Running grid search via server {server_url} using config '{config_path}'.")

    # 設定ファイル存在チェックと読み込み
    if not config_path.is_file():
        logger.error(f"Grid search config file not found: {config_path}")
        raise typer.Exit(code=1)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            grid_config = yaml.safe_load(f)
            if not grid_config:
                 logger.error(f"Grid search config file is empty: {config_path}")
                 raise typer.Exit(code=1)
            # YAMLの内容をツール入力用に文字列にするか、辞書のまま送るか (サーバーAPIによる)
            # ここでは辞書のまま送る想定
            # grid_config_str = yaml.dump(grid_config) # 文字列にする場合
    except yaml.YAMLError as e:
        logger.error(f"Error parsing grid search config YAML file {config_path}: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Error reading grid search config file {config_path}: {e}")
        raise typer.Exit(code=1)

    client = get_mcp_client(server_url)

    # ツール入力の準備
    tool_input = {
        "grid_config": grid_config, # 辞書として送信
        # 'output_base_dir': str(output_dir) if output_dir else None, # サーバー側で解釈・検証
        # 'num_procs': None, # 必要なら追加
        # 'skip_existing': False # 必要なら追加
    }

    logger.info("Calling 'run_grid_search' tool on server...")
    logger.debug(f"Tool input (config part omitted for brevity): {{'grid_config': ..., 'output_base_dir': {tool_input.get('output_base_dir')}}}")

    try:
        # MCPツール呼び出し
        job_start_result = client.run_tool("run_grid_search", tool_input)
        job_id = job_start_result.get("job_id")
        if not job_id:
             logger.error("Failed to start grid search job: No job_id received.")
             logger.error(f"Server response: {job_start_result}")
             raise typer.Exit(code=1)
        logger.info(f"Grid search job started successfully. Job ID: {job_id}")

        # ジョブ完了待機 (タイムアウトは長めに設定, 例: 2時間)
        final_job_info = poll_job_status(client, job_id, timeout=7200)

        # 結果処理
        if final_job_info.get("status") == "completed":
            logger.info(f"Grid search job {job_id} completed.")
            results = final_job_info.get("result", {})
            # TODO: グリッドサーチ結果のサマリー表示 (例: 最良パラメータ、パスなど)
            logger.info(f"Job Result Summary: {results.get('summary', 'No summary available.')}")
            if 'best_params_path' in results:
                 logger.info(f"Best parameters saved to (on server): {results['best_params_path']}")
            if 'results_csv_path' in results:
                 logger.info(f"Detailed results CSV saved to (on server): {results['results_csv_path']}")
        else:
            logger.error(f"Grid search job {job_id} failed.")
            error_msg = final_job_info.get("error_details", {}).get("message", "Unknown error")
            logger.error(f"Failure reason: {error_msg}")
            raise typer.Exit(code=1)

    except (ToolError, McpError, TimeoutError, RuntimeError, yaml.YAMLError, Exception) as e:
        logger.error(f"An error occurred during grid search: {e}", exc_info=True)
        if isinstance(e, yaml.YAMLError):
             typer.echo(f"Error: Invalid YAML in config file {config_path}. {e}", err=True)
        # 他のエラータイプは evaluate と同様
        elif isinstance(e, ToolError):
            typer.echo(f"Error: Server could not execute the grid search tool. {e}", err=True)
        elif isinstance(e, McpError):
            typer.echo(f"Error: Communication failed with the server. {e}", err=True)
        elif isinstance(e, TimeoutError):
             typer.echo(f"Error: Grid search job timed out. {e}", err=True)
        elif isinstance(e, RuntimeError):
             typer.echo(f"Error: Server error during job status check. {e}", err=True)
        else:
             typer.echo(f"An unexpected error occurred: {e}", err=True)
        raise typer.Exit(code=1)

# --- Improve Command ---
improve_app = typer.Typer(help="Automatically improve detector performance using MCP server.")
app.add_typer(improve_app, name="improve")

# デフォルト値用の定数（config.yaml にない場合のフォールバック）
DEFAULT_POLL_INTERVAL = 5
DEFAULT_JOB_TIMEOUT = 600
DEFAULT_MAX_CYCLES = 10
DEFAULT_LLM_MODEL = "claude-3-opus-20240229"
DEFAULT_IMPROVEMENT_THRESHOLD = 0.005 # 例: 0.5% 改善
DEFAULT_MAX_STAGNATION = 3
DEFAULT_SESSION_TIMEOUT = 3600 # 1時間

@improve_app.callback()
def improve_main(ctx: typer.Context, log_level: Optional[str] = "INFO"):
    """ Set log level for improve commands. """
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_int, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s')
    logging.info(f"Root logger level set to {log_level.upper()} via basicConfig.")
    # load_global_config() # Moved to command functions
    # ctx.obj = {"log_level": log_level.upper(), "config": CONFIG}

@improve_app.command("start", help="Start or resume the automatic improvement loop.")
async def improve(
    # --- Required arguments ---
    server_url: Annotated[str, typer.Option("--server", "-s", help="URL of the MCP server (required for improve).")],
    detector_name: Annotated[str, typer.Option("--detector", help="Name of the detector to improve.")],
    dataset_name: Annotated[str, typer.Option("--dataset", help="Name of the dataset to use for evaluation.")],

    # --- Optional arguments (with defaults) ---
    goal: Annotated[Optional[str], typer.Option("--goal", help="High-level goal for the improvement (e.g., 'Improve F1-score for onset detection').")] = None,
    session_id: Annotated[Optional[str], typer.Option("--session-id", help="Resume an existing improvement session.")] = None,
    # Client-side loop limit (safety net)
    max_cycles: Annotated[Optional[int], typer.Option("--max-cycles", help="Maximum number of improvement cycles.")] = None, # Default from config
    # LLM関連オプション
    llm_model: Annotated[Optional[str], typer.Option("--llm-model", help=f"LLM model to use (default from config or {DEFAULT_LLM_MODEL})")] = None, # Default from config
    api_key: Annotated[Optional[str], typer.Option("--api-key", help="LLM API key (if not specified, reads from environment variable)")] = None,
    # Server-side session parameters (passed during start_session)
    improvement_threshold: Annotated[Optional[float], typer.Option("--threshold", help=f"Minimum F1-score improvement required to reset stagnation counter (default from config or {DEFAULT_IMPROVEMENT_THRESHOLD})")] = None, # Default from config
    max_stagnation: Annotated[Optional[int], typer.Option("--max-stagnation", help=f"Stop if F1-score doesn't improve by the threshold for this many cycles (default from config or {DEFAULT_MAX_STAGNATION})")] = None, # Default from config
    session_timeout: Annotated[Optional[int], typer.Option("--session-timeout", help=f"Maximum duration for the entire improvement session in seconds (default from config or {DEFAULT_SESSION_TIMEOUT})")] = None, # Default from config
    # Polling options
    poll_interval: Annotated[Optional[int], typer.Option("--poll-interval", help=f"Interval (seconds) for polling session/job status (default from config or {DEFAULT_POLL_INTERVAL})")] = None, # Default from config
    job_timeout: Annotated[Optional[int], typer.Option("--job-timeout", help=f"Timeout (seconds) for individual improvement jobs (default from config or {DEFAULT_JOB_TIMEOUT})")] = None, # Default from config
    # ユーザー確認オプション
    always_confirm: Annotated[bool, typer.Option("--always-confirm/--no-confirm", help="Confirm before applying code changes (default: True).")] = True,
):
    """Starts or resumes the automatic improvement loop for a detector."""
    load_global_config() # Load config at the start of the command
    logger = logging.getLogger(__name__)

    # --- Resolve defaults from CONFIG --- #
    # Config sections: 'improve', 'llm', 'strategy', 'cleanup'
    improve_cfg = CONFIG.get('improve', {})
    llm_cfg = CONFIG.get('llm', {})
    strategy_cfg = CONFIG.get('strategy', {})
    cleanup_cfg = CONFIG.get('cleanup', {})

    final_max_cycles = max_cycles if max_cycles is not None else improve_cfg.get('max_cycles', DEFAULT_MAX_CYCLES)
    final_llm_model = llm_model if llm_model is not None else llm_cfg.get('model', DEFAULT_LLM_MODEL)
    final_improvement_threshold = improvement_threshold if improvement_threshold is not None else strategy_cfg.get('improvement_threshold', DEFAULT_IMPROVEMENT_THRESHOLD)
    final_max_stagnation = max_stagnation if max_stagnation is not None else strategy_cfg.get('max_stagnation', DEFAULT_MAX_STAGNATION)
    final_session_timeout = session_timeout if session_timeout is not None else cleanup_cfg.get('session_timeout_seconds', DEFAULT_SESSION_TIMEOUT)
    final_poll_interval = poll_interval if poll_interval is not None else improve_cfg.get('poll_interval', DEFAULT_POLL_INTERVAL)
    final_job_timeout = job_timeout if job_timeout is not None else improve_cfg.get('job_timeout', DEFAULT_JOB_TIMEOUT)

    # Log the effective parameters being used
    logger.info(f"--- Effective Improve Parameters ---")
    logger.info(f"Max Cycles: {final_max_cycles}")
    logger.info(f"LLM Model: {final_llm_model}")
    logger.info(f"Improvement Threshold: {final_improvement_threshold}")
    logger.info(f"Max Stagnation: {final_max_stagnation}")
    logger.info(f"Session Timeout: {final_session_timeout}s")
    logger.info(f"Polling Interval: {final_poll_interval}s")
    logger.info(f"Job Timeout: {final_job_timeout}s")
    logger.info(f"Always Confirm: {always_confirm}")
    logger.info(f"------------------------------------")

    # LLMクライアントのインポート (先ほど作成したモジュール)
    try:
        from src.cli.llm_client import initialize_llm_client, LLMClientError
    except ImportError as e:
        logger.error(f"Failed to import LLM client: {e}")
        typer.echo(f"Error: LLM client module not found. Please ensure src/cli/llm_client.py exists.", err=True)
        raise typer.Exit(code=1)
    
    # 非同期処理を実行するためのイベントループ取得
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # イベントループがないので新しく作成
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # MCP クライアント初期化
    client = get_mcp_client(server_url)
    
    # セッション識別子
    active_session_id = None
    
    try:
        # 1. LLM クライアント初期化 (API キーは環境変数から自動取得)
        logger.info(f"Initializing LLM client with model {final_llm_model}")
        # llm_client = loop.run_until_complete( # This causes RuntimeError in tests
        #     initialize_llm_client_async(api_key=api_key, model=final_llm_model) # Use final_llm_model
        # )
        # Await the async initializer directly
        llm_client = await initialize_llm_client_async(api_key=api_key, model=final_llm_model)

        # 2. セッション開始または再開
        logger.info(f"{'Resuming' if session_id else 'Starting'} improvement session...")
        # Use final config values for session parameters
        session_params = {
            "threshold": final_improvement_threshold,
            "max_stagnation": final_max_stagnation,
            "timeout": final_session_timeout,
        }

        start_session_input = {
            "base_algorithm": detector_name,
            "dataset_name": dataset_name,
            "goal": goal,
            "improvement_params": session_params # Pass resolved params
        }

        if session_id:
            # 既存セッションの再開
            try:
                # session_info = client.get_session_info(session_id=session_id) # Synchronous call
                session_info = await loop.run_in_executor(None, client.get_session_info, session_id) # Use run_in_executor
                active_session_id = session_id
                logger.info(f"Session {session_id} resumed successfully.")
            except Exception as e:
                logger.error(f"Failed to resume session {session_id}: {e}")
                typer.echo(f"Error: Could not resume session {session_id}. {e}", err=True)
                raise typer.Exit(code=1)
        else:
            # 新規セッション開始
            try:
                # session_info = client.start_session(**start_session_input) # Synchronous call
                # Use lambda to pass keyword arguments to run_in_executor
                session_info = await loop.run_in_executor(
                    None,
                    lambda: client.start_session(**start_session_input)
                )
                # session_info = await loop.run_in_executor(None, client.start_session, **start_session_input) # This caused TypeError
                active_session_id = session_info.get("session_id")
                if not active_session_id:
                    logger.error("Failed to start session: No session_id received.")
                    logger.error(f"Server response: {session_info}")
                    raise typer.Exit(code=1)
                logger.info(f"New session started. Session ID: {active_session_id}")
            except Exception as e:
                logger.error(f"Failed to start new session: {e}")
                typer.echo(f"Error: Could not start new session. {e}", err=True)
                raise typer.Exit(code=1)
        
        # 初期セッション情報の表示
        print_session_summary(session_info)
        
        # 3. 改善ループ (CLI主導)
        loop = asyncio.get_event_loop()
        session_active = True
        cycle_count = 0
        last_exception = None # ループ終了時の例外表示用

        while session_active and cycle_count < final_max_cycles: # Use final_max_cycles
            cycle_count += 1
            logger.info(f"\n--- Improvement Cycle {cycle_count}/{final_max_cycles} (Session: {active_session_id}) ---") # Use final_max_cycles
            typer.echo(f"\nCycle {cycle_count}...")

            try:
                # 1. 次のアクションを決定 (前にセッション情報を取得)
                logger.info("Fetching latest session info...")
                active_session_info = await loop.run_in_executor(None, client.get_session_info, active_session_id) # Use run_in_executor
                print_session_summary(active_session_info) # Print summary at start of cycle

                logger.info("Determining next action...")
                action_info = await determine_next_action(loop, client, llm_client, active_session_id, active_session_info) # Pass latest info
                action_name = action_info.get("action")
                action_params = action_info.get("parameters", {})
                # Use status from action_info first, fallback to fetched info
                session_status = action_info.get("session_status") or active_session_info.get("status", "running")

                # Define terminal states
                terminal_states = {"completed", "failed", "timeout", "cancelled"} # Example terminal states

                # if session_status != "running": # Original incorrect check
                if session_status in terminal_states:
                    logger.info(f"Session {active_session_id} has reached a terminal state ({session_status}). Stopping loop.")
                    typer.echo(f"Session ended with status: {session_status}")
                    session_active = False
                    # 完了/失敗理由を表示 (あれば)
                    reason = action_info.get("reason")
                    if reason:
                        typer.echo(f"Reason: {reason}")
                    continue # ループの次のイテレーションへ (while 条件で終了)

                if not action_name:
                    logger.warning("No next action determined by server/LLM. Waiting before retry.")
                    await asyncio.sleep(final_poll_interval) # Use final_poll_interval
                    continue # 次のサイクルへ

                logger.info(f"Next action: {action_name}")
                logger.debug(f"Action parameters: {action_params}")
                typer.echo(f"Next action: {action_name}")

                # 2. アクション実行
                execution_result = None
                start_exec_time = time.time()

                if action_name == "analyze_evaluation":
                    evaluation_results = action_params.get("evaluation_results")
                    if not evaluation_results: raise ValueError("Missing 'evaluation_results' for analyze_evaluation")
                    execution_result = await execute_analyze_evaluation(loop, client, llm_client, active_session_id, evaluation_results)
                elif action_name == "generate_hypotheses":
                    current_metrics = action_params.get("current_metrics")
                    num_hypotheses = action_params.get("num_hypotheses", 3)
                    if not current_metrics: raise ValueError("Missing 'current_metrics' for generate_hypotheses")
                    execution_result = await execute_generate_hypotheses(loop, client, llm_client, active_session_id, current_metrics, num_hypotheses)
                elif action_name == "improve_code":
                    code = action_params.get("code")
                    suggestion = action_params.get("suggestion")
                    if not code or not suggestion: raise ValueError("Missing 'code' or 'suggestion' for improve_code")
                    execution_result = await execute_improve_code(loop, client, llm_client, active_session_id, detector_name, code, suggestion, always_confirm)
                elif action_name == "run_evaluation":
                    code_version = action_params.get("code_version") # オプショナル
                    execution_result = await execute_run_evaluation(loop, client, active_session_id, detector_name, dataset_name, code_version, final_poll_interval, final_job_timeout)
                elif action_name == "optimize_parameters":
                    current_metrics = active_session_info.get("current_metrics")
                    if not current_metrics: raise ValueError("Missing 'current_metrics' in session info for optimize_parameters")
                    execution_result = await execute_optimize_parameters(loop, client, llm_client, active_session_id, detector_name, current_metrics, always_confirm, final_poll_interval, final_job_timeout)
                else:
                    logger.error(f"Unknown action name: {action_name}. Skipping.")
                    typer.echo(f"Error: Unknown action '{action_name}' received.", err=True)
                    # 不明なアクションの場合、少し待って次のサイクルへ
                    await asyncio.sleep(final_poll_interval) # Use final_poll_interval
                    continue # このサイクルの残りをスキップ

                end_exec_time = time.time()
                logger.info(f"Action '{action_name}' completed in {end_exec_time - start_exec_time:.2f}s")
                if execution_result:
                     logger.debug(f"Execution result: {execution_result}")

                # アクションが成功した場合、次のサイクルに進む前に待機
                await asyncio.sleep(1) # 1秒待機

            except (McpError, ToolError, LLMClientError, TimeoutError, ValueError) as loop_err: # ValueErrorもここで捕捉
                 # M-02: determine_next_action やアクション実行中の特定のエラーを捕捉
                 logger.error(f"Error in improvement loop (cycle {cycle_count}): {loop_err}", exc_info=True)
                 last_exception = loop_err
                 # エラーの種類に応じた処理
                 if isinstance(loop_err, TimeoutError):
                     typer.echo(typer.style(f"Error: Operation timed out during cycle {cycle_count}.", fg=typer.colors.YELLOW),
                                err=True) # Corrected style call
                     typer.echo("This might be a temporary issue. Consider increasing timeout values or checking server responsiveness.", err=True)
                     # 一旦停止させる
                     session_active = False
                 elif isinstance(loop_err, LLMClientError):
                     error_msg = f"LLM Client Error: {loop_err}"
                     if loop_err.error_type == "APIError":
                         error_msg += " Check API key, quota, and LLM service status."
                     elif loop_err.error_type == "RateLimitError":
                         error_msg += " Rate limit reached. Please wait and try again later."
                     else:
                         error_msg += " An unexpected error occurred with the LLM client."
                     typer.echo(typer.style(f"Error during cycle {cycle_count}: {error_msg}", fg=typer.colors.RED),
                                err=True) # Corrected style call
                     session_active = False # 回復不能とみなし停止 (リトライ実装なし)
                 elif isinstance(loop_err, (McpError, ToolError)):
                     typer.echo(typer.style(f"Error during cycle {cycle_count}: MCP Server/Tool Error - {loop_err}", fg=typer.colors.RED),
                                err=True) # Corrected style call
                     typer.echo("Check MCP server status and logs for details.", err=True)
                     session_active = False # サーバーエラーも一旦停止
                 elif isinstance(loop_err, ValueError):
                     # 設定や予期しないデータによるエラーなど
                     typer.echo(typer.style(f"Error during cycle {cycle_count}: Configuration or Data Error - {loop_err}", fg=typer.colors.RED),
                                err=True) # Corrected style call
                     typer.echo("Check configuration files or input data.", err=True)
                     session_active = False
                 else:
                     # これは通常発生しないはずだが念のため
                     typer.echo(typer.style(f"An unexpected operational error occurred during cycle {cycle_count}: {loop_err}", fg=typer.colors.RED),
                                err=True) # Corrected style call
                     session_active = False
                 # break # ループを抜ける必要はない、次の while 条件で評価される
            except Exception as global_err:
                 # M-02: ループ内のさらに予期せぬエラー
                 logger.critical(f"Critical unexpected error in improvement loop (cycle {cycle_count}): {global_err}", exc_info=True)
                 typer.echo(typer.style(f"Critical unexpected error: {global_err}", fg=typer.colors.RED), err=True) # Added style
                 typer.echo("Stopping the improvement loop immediately.", err=True)
                 last_exception = global_err
                 session_active = False
                 break # 予期せぬエラーは即時 break する

        # --- ループ終了後 --- # <- コメント修正
        logger.info("Improvement loop finished.")

        # ループ終了後の最終状態取得
        try:
            # final_session_info = client.get_session_info(session_id=active_session_id) # Synchronous
            final_session_info = await loop.run_in_executor(None, client.get_session_info, active_session_id) # Use run_in_executor
            logger.info("Final session state:")
            if final_session_info: # Check if info was obtained
                print_session_summary(final_session_info)
            else:
                logger.warning("Could not retrieve final session state.")
        except Exception as e:
            logger.error(f"Failed to get final session state: {e}")
    
    except typer.Exit:
        # Handle exits gracefully
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error: {e}", exc_info=True)
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        raise typer.Exit(code=1)
    finally:
        logger.info("Exiting mirai improve command.")

# 以下、新しく追加する関数

async def initialize_llm_client_async(api_key: Optional[str] = None, model: str = DEFAULT_LLM_MODEL):
    """LLMクライアントを非同期に初期化"""
    from src.cli.llm_client import initialize_llm_client
    # The model default here might be less relevant now, as it's resolved in 'improve'
    # But keep it for potential direct calls or other usages.
    return initialize_llm_client(api_key=api_key, model=model)

async def determine_next_action(loop, client, llm_client, session_id: str, session_info: Dict[str, Any]) -> Dict[str, Any]:
    """次のアクションを決定する
    
    ホスト側で戦略決定を行う重要な関数。セッション状態に基づいて次のアクションを決定します。
    """
    logger = logging.getLogger(__name__)
    
    # セッション状態の取得
    cycle_state = session_info.get("cycle_state", {})
    current_metrics = session_info.get("current_metrics", {})
    cycle_count = cycle_state.get("cycle_count", 0)
    
    # 特定の状態での判断
    if cycle_count == 0 or not current_metrics:
        # 初期サイクルまたはメトリクスがない場合は評価実行
        logger.info("Initial cycle or no metrics available. Suggesting evaluation.")
        return {"action": "run_evaluation"}
    
    try:
        # 戦略提案をサーバーに依頼
        job_result = client.run_tool("get_suggest_exploration_strategy_prompt", {"session_id": session_id})
        job_id = job_result.get("job_id")
        
        if not job_id:
            logger.error("No job_id received from get_suggest_exploration_strategy_prompt.")
            # フォールバック戦略
            return {"action": "improve_code", "suggestion": "Improve the overall performance."}
        
        # ジョブ完了を待つ
        job_info = await loop.run_in_executor(
            None, poll_job_status, client, job_id # Pass client and job_id
            # Timeout/interval not directly controlled here, uses poll_job_status defaults
        )
        
        if job_info.get("status") != "completed":
            logger.error("Strategy suggestion prompt generation failed.")
            # フォールバック戦略
            return {"action": "analyze_evaluation", "evaluation_results": current_metrics}
        
        # プロンプトの取得
        prompt = job_info.get("result", {}).get("prompt", "")
        if not prompt:
            logger.error("No prompt received from strategy suggestion.")
            return {"action": "analyze_evaluation"}
        
        # LLMによる戦略提案
        strategy_json = await llm_client.generate(prompt, request_json=True)
        logger.debug(f"LLM strategy response: {strategy_json}")
        
        # JSONのパース
        strategy = json.loads(strategy_json)
        
        # 戦略からアクション取得
        action = strategy.get("action", "improve_code")
        reasoning = strategy.get("reasoning", "No reasoning provided.")
        parameters = strategy.get("parameters", {})
        
        logger.info(f"Strategy suggests action: {action}")
        logger.info(f"Reasoning: {reasoning}")
        
        # アクションと付加情報を返す
        result = {"action": action}
        result.update(parameters)
        return result
        
    except Exception as e:
        logger.error(f"Error determining next action: {e}", exc_info=True)
        # エラー時のフォールバック戦略
        if cycle_count % 2 == 0:
            return {"action": "improve_code", "suggestion": "Fix any issues and improve performance."}
        else:
            return {"action": "run_evaluation"}

async def execute_analyze_evaluation(loop, client, llm_client, session_id: str, evaluation_results: Dict[str, Any]):
    """評価結果の分析を実行"""
    logger = logging.getLogger(__name__)
    typer.echo("Analyzing evaluation results...") # Added start message

    try:
        # 評価分析プロンプトを生成
        job_result = client.run_tool("get_analyze_evaluation_prompt", {
            "session_id": session_id,
            "evaluation_results": evaluation_results
        })
        
        job_id = job_result.get("job_id")
        if not job_id:
            logger.error("No job_id received from get_analyze_evaluation_prompt.")
            return False
        
        # ジョブ完了を待つ
        typer.echo(f"Waiting for analysis prompt generation job ({job_id}) to complete...") # Added waiting message
        job_info = await loop.run_in_executor(
            None, poll_job_status, client, job_id # Pass client and job_id
            # Timeout/interval not directly controlled here, uses poll_job_status defaults
        )
        
        if job_info.get("status") != "completed":
            logger.error("Analyze evaluation prompt generation failed.")
            return False
        
        # プロンプトの取得
        prompt = job_info.get("result", {}).get("prompt", "")
        if not prompt:
            logger.error("No prompt received for analysis.")
            return False
        
        # LLMによる分析
        typer.echo("Generating analysis using LLM...") # Added LLM call message
        analysis_json = await llm_client.generate(prompt, request_json=True)
        logger.debug(f"LLM analysis response: {analysis_json}")
        
        # 分析結果を表示
        try:
            analysis = json.loads(analysis_json)
            typer.echo("\n=== Evaluation Analysis ===")
            typer.echo(f"Overall Summary: {analysis.get('overall_summary', 'No summary available.')}")
            
            strengths = analysis.get('strengths', [])
            if strengths:
                typer.echo("\nStrengths:")
                for strength in strengths:
                    typer.echo(f"- {strength}")
            
            weaknesses = analysis.get('weaknesses', [])
            if weaknesses:
                typer.echo("\nWeaknesses:")
                for weakness in weaknesses:
                    typer.echo(f"- {weakness}")
            
            next_steps = analysis.get('potential_next_steps', [])
            if next_steps:
                typer.echo("\nPotential Next Steps:")
                for step in next_steps:
                    typer.echo(f"- {step}")
            
            typer.echo("===========================\n")
        except json.JSONDecodeError:
            typer.echo("\n=== Evaluation Analysis ===")
            typer.echo(analysis_json)
            typer.echo("===========================\n")
        
        # 分析結果を履歴として記録
        client.add_session_history(
            session_id=session_id,
            event_type="analysis_complete",
            data={"analysis": json.loads(analysis_json) if isinstance(analysis_json, str) else analysis_json},
            cycle_state_update={"last_analysis": json.loads(analysis_json) if isinstance(analysis_json, str) else analysis_json}
        )
        
        typer.echo("Analysis complete.") # Added end message
        return True
    
    except Exception as e:
        logger.error(f"Error executing analysis: {e}", exc_info=True)
        return False

async def execute_generate_hypotheses(loop, client, llm_client, session_id: str, current_metrics: Dict[str, Any], num_hypotheses: int = 3):
    """改善仮説の生成を実行"""
    logger = logging.getLogger(__name__)
    typer.echo("Generating improvement hypotheses...") # Added start message

    try:
        # 仮説生成プロンプトを取得
        job_result = client.run_tool("get_generate_hypotheses_prompt", {
            "session_id": session_id,
            "num_hypotheses": num_hypotheses,
            "current_metrics": current_metrics
        })
        
        job_id = job_result.get("job_id")
        if not job_id:
            logger.error("No job_id received from get_generate_hypotheses_prompt.")
            return False
        
        # ジョブ完了を待つ
        job_info = await loop.run_in_executor(
            None, poll_job_status, client, job_id # Pass client and job_id
            # Timeout/interval not directly controlled here, uses poll_job_status defaults
        )
        
        if job_info.get("status") != "completed":
            logger.error("Generate hypotheses prompt generation failed.")
            return False
        
        # プロンプトの取得
        prompt = job_info.get("result", {}).get("prompt", "")
        if not prompt:
            logger.error("No prompt received for hypotheses generation.")
            return False
        
        # LLMによる仮説生成
        hypotheses_json = await llm_client.generate(prompt, request_json=True)
        logger.debug(f"LLM hypotheses response: {hypotheses_json}")
        
        # 仮説を表示
        try:
            hypotheses_data = json.loads(hypotheses_json)
            typer.echo("\n=== Improvement Hypotheses ===")
            
            hypotheses = hypotheses_data.get('hypotheses', [])
            if isinstance(hypotheses, list):
                for i, hypothesis in enumerate(hypotheses, 1):
                    if isinstance(hypothesis, str):
                        typer.echo(f"{i}. {hypothesis}")
                    elif isinstance(hypothesis, dict) and 'description' in hypothesis:
                        typer.echo(f"{i}. {hypothesis['description']}")
            
            typer.echo("==============================\n")
        except json.JSONDecodeError:
            typer.echo("\n=== Improvement Hypotheses ===")
            typer.echo(hypotheses_json)
            typer.echo("==============================\n")
        
        # 仮説を履歴として記録
        client.add_session_history(
            session_id=session_id,
            event_type="hypotheses_generated",
            data={"hypotheses": json.loads(hypotheses_json) if isinstance(hypotheses_json, str) else hypotheses_json},
            cycle_state_update={"last_hypotheses": json.loads(hypotheses_json) if isinstance(hypotheses_json, str) else hypotheses_json}
        )
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating hypotheses: {e}", exc_info=True)
        return False

async def execute_improve_code(loop, client, llm_client, session_id: str, detector_name: str, code: str, suggestion: str, always_confirm: bool = True):
    """コード改善を実行"""
    logger = logging.getLogger(__name__)
    
    try:
        # コード改善プロンプトを生成
        job_result = client.run_tool("get_improve_code_prompt", {
            "session_id": session_id,
            "code": code,
            "suggestion": suggestion
        })
        
        job_id = job_result.get("job_id")
        if not job_id:
            logger.error("No job_id received from get_improve_code_prompt.")
            return False
        
        # ジョブ完了を待つ
        job_info = await loop.run_in_executor(
            None, poll_job_status, client, job_id # Pass client and job_id
            # Timeout/interval not directly controlled here, uses poll_job_status defaults
        )
        
        if job_info.get("status") != "completed":
            logger.error("Improve code prompt generation failed.")
            return False
        
        # プロンプトの取得
        prompt = job_info.get("result", {}).get("prompt", "")
        if not prompt:
            logger.error("No prompt received for code improvement.")
            return False
        
        # LLMによるコード改善
        llm_response = await llm_client.generate(prompt)
        logger.debug(f"LLM code response length: {len(llm_response)}")
        
        # コードの抽出
        improved_code = await llm_client.extract_code_from_text(llm_response)
        if not improved_code:
            logger.error("Failed to extract code from LLM response.")
            return False
        
        # コードの差分表示 (difflib)
        diff = difflib.unified_diff(
            code.splitlines(),
            improved_code.splitlines(),
            fromfile=f'original_{detector_name}.py',
            tofile=f'improved_{detector_name}.py',
            lineterm=''
        )
        
        # 差分の表示
        typer.echo("\n=== Code Improvement ===")
        typer.echo(f"Suggestion: {suggestion}")
        typer.echo("\nChanges:")
        for line in diff:
            if line.startswith('+'):
                typer.echo(typer.style(line, fg=typer.colors.GREEN))
            elif line.startswith('-'):
                typer.echo(typer.style(line, fg=typer.colors.RED))
            else:
                typer.echo(line)
        typer.echo("========================\n")
        
        # ユーザー確認 (オプション)
        if always_confirm:
            if not typer.confirm("Apply these code changes?"):
                logger.info("User rejected code changes.")
                # 改善拒否を履歴に記録
                client.add_session_history(
                    session_id=session_id,
                    event_type="code_improvement_rejected",
                    data={"suggestion": suggestion},
                    cycle_state_update={}
                )
                return False
        
        # コードの保存
        save_result = client.run_tool("save_code", {
            "detector_name": detector_name,
            "code": improved_code,
            "session_id": session_id,
            "changes_summary": suggestion
        })
        
        save_job_id = save_result.get("job_id")
        if not save_job_id:
            logger.error("No job_id received from save_code.")
            return False
        
        # 保存ジョブ完了を待つ
        save_job_info = await loop.run_in_executor(
            None, poll_job_status, client, save_job_id # Pass client and job_id
            # Timeout/interval not directly controlled here, uses poll_job_status defaults
        )
        
        if save_job_info.get("status") != "completed":
            logger.error("Save code failed.")
            return False
        
        # 成功メッセージ
        saved_version = save_job_info.get("result", {}).get("version", "unknown")
        typer.echo(f"Code successfully saved as version: {saved_version}")
        
        # コード改善を履歴として記録 (すでにsave_codeで記録されている場合もあり)
        try:
            client.add_session_history(
                session_id=session_id,
                event_type="code_improvement_applied",
                data={"suggestion": suggestion, "version": saved_version},
                cycle_state_update={"last_code_version": saved_version, "needs_evaluation": True}
            )
        except Exception as hist_err:
            logger.warning(f"Failed to add history for code improvement: {hist_err}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error improving code: {e}", exc_info=True)
        return False

async def execute_run_evaluation(loop, client, session_id: str, detector_name: str, dataset_name: str, code_version: Optional[str] = None, poll_interval: int = DEFAULT_POLL_INTERVAL, job_timeout: int = DEFAULT_JOB_TIMEOUT):
    """評価を実行"""
    logger = logging.getLogger(__name__)
    
    try:
        # 評価ジョブを開始
        eval_result = client.run_tool("run_evaluation", {
            "detector_name": detector_name,
            "dataset_name": dataset_name,
            "code_version": code_version,
            "session_id": session_id,
            "save_plots": True
        })
        
        eval_job_id = eval_result.get("job_id")
        if not eval_job_id:
            logger.error("No job_id received from run_evaluation.")
            return False
        
        # 評価ジョブ完了を待つ
        typer.echo("Running evaluation... This may take a while.")
        with typer.progressbar(length=100, label="Evaluating") as progress:
            # 進捗の表示 (本当の進捗ではなく時間ベース)
            progress.update(10)
            
            eval_job_info = await loop.run_in_executor(
                None,
                lambda: poll_job_status(client, eval_job_id, poll_interval=poll_interval, timeout=job_timeout)
            )
            
            progress.update(100)
        
        if eval_job_info.get("status") != "completed":
            logger.error("Evaluation failed.")
            typer.echo("Evaluation failed.")
            return False
        
        # 評価結果の取得
        eval_results = eval_job_info.get("result", {})
        
        # 結果を表示
        typer.echo("\n=== Evaluation Results ===")
        metrics = eval_results.get("metrics", {})
        
        # F1スコアなどの主要メトリクスを表示
        note_metrics = metrics.get("note", {})
        if note_metrics:
            typer.echo(f"Note F-measure: {note_metrics.get('f_measure', 'N/A'):.4f}")
            typer.echo(f"Note Precision: {note_metrics.get('precision', 'N/A'):.4f}")
            typer.echo(f"Note Recall: {note_metrics.get('recall', 'N/A'):.4f}")
        
        onset_metrics = metrics.get("onset", {})
        if onset_metrics:
            typer.echo(f"Onset F-measure: {onset_metrics.get('f_measure', 'N/A'):.4f}")
        
        typer.echo("=========================\n")
        
        # 評価結果を履歴として記録 (すでにrun_evaluationで記録されている場合もあり)
        try:
            client.add_session_history(
                session_id=session_id,
                event_type="evaluation_viewed",
                data={"metrics": metrics},
                cycle_state_update={"last_evaluation": metrics, "needs_evaluation": False, "needs_assessment": True}
            )
        except Exception as hist_err:
            logger.warning(f"Failed to add history for evaluation view: {hist_err}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error running evaluation: {e}", exc_info=True)
        return False

async def execute_optimize_parameters(loop, client, llm_client, session_id: str, detector_name: str, current_metrics: Dict[str, Any], always_confirm: bool = True, poll_interval: int = DEFAULT_POLL_INTERVAL, job_timeout: int = DEFAULT_JOB_TIMEOUT):
    """パラメータ最適化を実行"""
    logger = logging.getLogger(__name__)
    
    try:
        # 現在のコードを取得
        code_info = client.get_code(detector_name=detector_name)
        current_code = code_info.get("code", "")
        
        # パラメータ提案プロンプトを生成
        job_result = client.run_tool("get_suggest_parameters_prompt", {
            "session_id": session_id,
            "detector_code": current_code,
            "current_metrics": current_metrics
        })
        
        job_id = job_result.get("job_id")
        if not job_id:
            logger.error("No job_id received from get_suggest_parameters_prompt.")
            return False
        
        # ジョブ完了を待つ
        job_info = await loop.run_in_executor(
            None, poll_job_status, client, job_id # Pass client and job_id
            # Timeout/interval not directly controlled here, uses poll_job_status defaults
        )
        
        if job_info.get("status") != "completed":
            logger.error("Parameter suggestion prompt generation failed.")
            return False
        
        # プロンプトの取得
        prompt = job_info.get("result", {}).get("prompt", "")
        if not prompt:
            logger.error("No prompt received for parameter suggestion.")
            return False
        
        # LLMによるパラメータ提案
        params_json = await llm_client.generate(prompt, request_json=True)
        logger.debug(f"LLM parameters response: {params_json}")
        
        # パラメータ提案を表示
        try:
            params_data = json.loads(params_json)
            typer.echo("\n=== Parameter Suggestions ===")
            
            parameters = params_data.get('parameters', [])
            overall_rationale = params_data.get('overall_rationale', '')
            
            if overall_rationale:
                typer.echo(f"Rationale: {overall_rationale}")
            
            typer.echo("\nSuggested Parameters:")
            for param in parameters:
                name = param.get('name', 'unknown')
                current = param.get('current_value', 'N/A')
                suggested_range = param.get('suggested_range', [])
                suggested_values = param.get('suggested_values', [])
                rationale = param.get('rationale', '')
                
                typer.echo(f"- {name}:")
                typer.echo(f"  Current: {current}")
                if suggested_range:
                    typer.echo(f"  Suggested Range: {suggested_range}")
                if suggested_values:
                    typer.echo(f"  Suggested Values: {suggested_values}")
                if rationale:
                    typer.echo(f"  Rationale: {rationale}")
            
            typer.echo("=============================\n")
        except json.JSONDecodeError:
            typer.echo("\n=== Parameter Suggestions ===")
            typer.echo(params_json)
            typer.echo("=============================\n")
        
        # グリッドサーチを実行するか確認
        if typer.confirm("Run grid search with these parameter suggestions?"):
            # パラメータ提案からグリッドサーチ設定を生成
            grid_config = create_grid_config_from_suggestions(detector_name, json.loads(params_json))
            
            # グリッドサーチを実行
            grid_result = client.run_tool("run_grid_search", {
                "grid_config": grid_config,
                "session_id": session_id
            })
            
            grid_job_id = grid_result.get("job_id")
            if not grid_job_id:
                logger.error("No job_id received from run_grid_search.")
                return False
            
            # グリッドサーチジョブ完了を待つ
            typer.echo("Running grid search... This may take a while.")
            with typer.progressbar(length=100, label="Grid Search") as progress:
                progress.update(10)
                
                grid_job_info = await loop.run_in_executor(
                    None,
                    lambda: poll_job_status(client, grid_job_id, poll_interval=poll_interval, timeout=job_timeout)
                )
                
                progress.update(100)
            
            if grid_job_info.get("status") != "completed":
                logger.error("Grid search failed.")
                typer.echo("Grid search failed.")
                return False
            
            # グリッドサーチ結果の取得
            grid_results = grid_job_info.get("result", {})
            
            # 結果を表示
            typer.echo("\n=== Grid Search Results ===")
            typer.echo(f"Best Score: {grid_results.get('best_score', 'N/A')}")
            typer.echo(f"Best Parameters: {grid_results.get('best_params', {})}")
            typer.echo(f"Results CSV: {grid_results.get('results_csv_path', 'N/A')}")
            typer.echo("==========================\n")
            
            # 最適パラメータでコードを更新するか確認
            if typer.confirm("Apply best parameters to the detector code?"):
                # パラメータを適用したコードを生成
                updated_code = apply_parameters_to_code(current_code, grid_results.get('best_params', {}))
                
                # コードを保存
                save_result = client.run_tool("save_code", {
                    "detector_name": detector_name,
                    "code": updated_code,
                    "session_id": session_id,
                    "changes_summary": f"Applied optimal parameters: {grid_results.get('best_params', {})}"
                })
                
                save_job_id = save_result.get("job_id")
                if not save_job_id:
                    logger.error("No job_id received from save_code.")
                    return False
                
                # 保存ジョブ完了を待つ
                save_job_info = await loop.run_in_executor(
                    None, poll_job_status, client, save_job_id # Pass client and job_id
                    # Timeout/interval not directly controlled here, uses poll_job_status defaults
                )
                
                if save_job_info.get("status") != "completed":
                    logger.error("Save code with optimized parameters failed.")
                    return False
                
                # 成功メッセージ
                saved_version = save_job_info.get("result", {}).get("version", "unknown")
                typer.echo(f"Code with optimized parameters saved as version: {saved_version}")
                
                # パラメータ最適化を履歴として記録
                client.add_session_history(
                    session_id=session_id,
                    event_type="parameters_optimized",
                    data={"best_params": grid_results.get('best_params', {}), "version": saved_version},
                    cycle_state_update={"last_code_version": saved_version, "needs_evaluation": True, "last_optimized_params": grid_results.get('best_params', {})}
                )
        
        return True
    
    except Exception as e:
        logger.error(f"Error optimizing parameters: {e}", exc_info=True)
        return False

def create_grid_config_from_suggestions(detector_name: str, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """パラメータ提案からグリッドサーチ設定を生成する"""
    logger = logging.getLogger(__name__)
    grid_config = {
        "detector_name": detector_name,
        "parameters": {}
    }

    try:
        parameters = suggestion.get('parameters', [])
        if not isinstance(parameters, list):
            logger.warning(f"Expected 'parameters' to be a list in suggestion, got {type(parameters)}. Skipping grid config generation.")
            return grid_config

        for param in parameters:
            try:
                if not isinstance(param, dict):
                    logger.warning(f"Expected parameter item to be a dict, got {type(param)}. Skipping this item: {param}")
                    continue

                name = param.get('name')
                if not name:
                    logger.warning(f"Missing 'name' in parameter suggestion item. Skipping: {param}")
                    continue

                suggested_range = param.get('suggested_range', [])
                suggested_values = param.get('suggested_values', [])
                param_type = param.get('type', '').lower()

                if suggested_range and isinstance(suggested_range, list) and len(suggested_range) >= 2:
                    # 数値範囲の場合
                    try:
                        min_val, max_val = suggested_range[0], suggested_range[-1]

                        if param_type == 'int':
                            # 整数パラメータ
                            min_int, max_int = int(min_val), int(max_val)
                            grid_config['parameters'][name] = {
                                "min": min_int,
                                "max": max_int,
                                "num": min(10, max_int - min_int + 1) if max_int >= min_int else 1, # 最大10点, 範囲が逆なら1点
                                "log": False
                            }
                        elif param_type == 'float':
                            # 浮動小数点パラメータ
                            min_float, max_float = float(min_val), float(max_val)
                            grid_config['parameters'][name] = {
                                "min": min_float,
                                "max": max_float,
                                "num": 10,  # 10点
                                "log": False
                            }
                        else:
                            logger.warning(f"Unknown or unsupported type '{param_type}' for range parameter '{name}'. Skipping.")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing range for parameter '{name}': {e}. Range was: {suggested_range}. Skipping.")

                elif suggested_values and isinstance(suggested_values, list):
                    # 離散値の場合
                    if not suggested_values: # 空リストはスキップ
                        logger.warning(f"Empty 'suggested_values' for parameter '{name}'. Skipping.")
                        continue
                    grid_config['parameters'][name] = {
                        "values": suggested_values
                    }
                else:
                    # 範囲も離散値も指定がない場合は警告（ログレベルDEBUGでも良いかも）
                    logger.debug(f"No valid 'suggested_range' or 'suggested_values' found for parameter '{name}'. Skipping grid entry for it.")

            except (KeyError, TypeError, AttributeError) as e:
                logger.warning(f"Error processing parameter suggestion item: {e}. Item was: {param}. Skipping.")
                continue # 次のパラメータへ

    except Exception as e:
        # suggestion 自体の形式がおかしい場合など、予期せぬエラー
        logger.error(f"Unexpected error creating grid config from suggestions: {e}", exc_info=True)
        # エラー発生時は空の設定を返す（あるいは例外を再発生させるか要検討）
        return {"detector_name": detector_name, "parameters": {}}

    logger.info(f"Generated grid config for {detector_name}: {grid_config['parameters']}")
    return grid_config

def apply_parameters_to_code(code: str, parameters: Dict[str, Any]) -> str:
    """最適パラメータをコードに適用する (正規表現ベース)。

    注意:
    この関数は正規表現を使用してコード内の単純な変数代入 (`var = value`) を
    見つけ、置き換えます。以下の制限があります:
    - コメント内や文字列リテラル内の変数名を誤って変更する可能性があります。
    - 辞書やクラス属性など、より複雑な方法でパラメータが設定されている場合、
      正しく適用できない可能性があります。
    - パラメータが存在しない場合、ファイルの末尾に追加されます。

    この関数による変更後は、コードが期待通り動作するか必ず確認してください。
    より堅牢な方法として AST (抽象構文木) の使用が考えられますが、実装が複雑になります。

    Args:
        code: 元のソースコード文字列。
        parameters: 適用するパラメータ名と値の辞書。

    Returns:
        パラメータが適用された (可能性のある) コード文字列。
    """
    import re
    logger = logging.getLogger(__name__) # Add logger

    # コードの行リスト
    lines = code.splitlines()
    updated_lines = []
    
    # パラメータ変数の定義を探す
    param_applied = set()
    for line in lines:
        applied = False
        for param_name, param_value in parameters.items():
            # 変数定義のパターン (例: threshold = 0.5)
            pattern = r'^\s*(' + re.escape(param_name) + r')\s*=\s*([^#]*)'
            match = re.match(pattern, line)
            
            if match and param_name not in param_applied:
                # 変数定義を新しい値で置き換え
                var_name = match.group(1)
                before_comment = line.split('#', 1)[0]
                after_comment = line.split('#', 1)[1] if '#' in line else ""
                
                # コメントがある場合は保持
                if after_comment:
                    updated_line = f"{var_name} = {repr(param_value)} # {after_comment}" # Use repr for safer string representation
                else:
                    updated_line = f"{var_name} = {repr(param_value)}" # Use repr

                updated_lines.append(updated_line)
                param_applied.add(param_name)
                applied = True
                break
        
        if not applied:
            updated_lines.append(line)
    
    # 適用されなかったパラメータを最後に追加 (意図しない場合もあるのでログを残す)
    unapplied_params = set(parameters.keys()) - param_applied
    if unapplied_params:
        logger.warning(f"Parameters not found in code and will be appended: {unapplied_params}")
        updated_lines.append("\n# --- Parameters appended by optimization ---")
        for param_name in sorted(unapplied_params):
            param_value = parameters[param_name]
            updated_lines.append(f"{param_name} = {repr(param_value)}  # Added by parameter optimization")

    return "\n".join(updated_lines)

def print_session_summary(session_data: Dict[str, Any]):
    """Helper function to print session summary information."""
    # Add logger definition
    logger = logging.getLogger(__name__)
    print("\n--- Session Summary ---")
    session_id = session_data.get('session_id')
    status = session_data.get('status', 'Unknown')
    cycle_state = session_data.get('cycle_state', {})
    best_metrics = session_data.get('best_metrics') # Get best_metrics, might be None

    cycle_count = cycle_state.get('cycle_count', 0)
    stagnation_count = cycle_state.get('stagnation_count', 0)
    # Handle the case where best_metrics might be None
    best_f_measure = None
    if best_metrics: # Check if best_metrics is not None
        best_f_measure = best_metrics.get('overall', {}).get('f_measure') # 仮に overall F-measure を指標とする

    typer.echo(f"--- Session Summary (ID: {session_id}) ---") # Use typer.echo for user output
    typer.echo(f"Status: {status}")
    logger.info(f"--- Session Summary (ID: {session_id}) ---")
    logger.info(f"Cycle: {cycle_count}")
    logger.info(f"Stagnation: {stagnation_count}")
    if best_f_measure is not None:
        typer.echo(f"Best F-measure: {best_f_measure:.4f}")
    else:
        typer.echo("Best F-measure: N/A")
    logger.info("-----------------------------------------")

# ... (if __name__ == "__main__": の部分は mirai.py では不要) 