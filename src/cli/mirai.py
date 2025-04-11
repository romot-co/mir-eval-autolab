#!/usr/bin/env python
# src/cli/mirai.py
"""MIR Auto Improver (mirai) Command Line Interface."""

import typer
from typing_extensions import Annotated
from typing import Optional, Dict, Any
import json
import sys
import logging
from pathlib import Path
import yaml
import traceback
import time

# --- MCP Client Imports ---
from mcp import client as MCPClient
from mcp import ToolError, MCPError

# --- Core Function Imports ---
# 適切なパスに調整してください
try:
    from src.evaluation.evaluation_runner import evaluate_detector
    from src.utils.logging_utils import setup_logger
    from src.utils.path_utils import get_project_root
    from src.utils.exception_utils import log_exception
    from src.utils.misc_utils import load_config_yaml
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
        except MCPError as e:
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
    str,
    typer.Option("--log-level", help="Set the logging level.", default="INFO", case_sensitive=False)
]

# --- Evaluate Command ---
eval_app = typer.Typer(help="Evaluate MIR detectors either standalone or via MCP server.")
app.add_typer(eval_app, name="evaluate")

@eval_app.callback()
def eval_main(ctx: typer.Context, log_level: LogLevel):
    """ Set log level for evaluate commands. """
    # このコールバックは eval_app のすべてのコマンドの前に実行される
    # setup_logger は特定のロガーを設定するため、ここでは基本的な logging.basicConfig を使うか、
    # ルートロガーを設定する必要があるかもしれない。ただし、他のライブラリへの影響に注意。
    # 一旦、ルートロガーレベルを設定するシンプルな方法に変更
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_int, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s')
    # logging.getLogger().setLevel(log_level_int) # basicConfig がレベルを設定
    logging.info(f"Root logger level set to {log_level.upper()} via basicConfig.")
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

        except (ToolError, MCPError, TimeoutError, RuntimeError, Exception) as e:
            logger.error(f"An error occurred during server-based evaluation: {e}", exc_info=True)
            # エラータイプに応じてユーザーへのメッセージを調整
            if isinstance(e, ToolError):
                typer.echo(f"Error: Server could not execute the evaluation tool. {e}", err=True)
            elif isinstance(e, MCPError):
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
def grid_main(ctx: typer.Context, log_level: LogLevel):
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_int, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s')
    logging.info(f"Root logger level set to {log_level.upper()} via basicConfig.")
    load_global_config()

@grid_app.command("run", help="Run grid search.")
def grid_search(
    server_url: Annotated[str, typer.Option("--server", "-s", help="URL of the MCP server (required for grid search).")],
    config_path: Annotated[Path, typer.Option("--config", "-c", help="Path to the grid search configuration YAML file.")],
    output_dir: OutputDir = None,
):
    """Runs grid search using the MCP server."""
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

    except (ToolError, MCPError, TimeoutError, RuntimeError, yaml.YAMLError, Exception) as e:
        logger.error(f"An error occurred during grid search: {e}", exc_info=True)
        if isinstance(e, yaml.YAMLError):
             typer.echo(f"Error: Invalid YAML in config file {config_path}. {e}", err=True)
        # 他のエラータイプは evaluate と同様
        elif isinstance(e, ToolError):
            typer.echo(f"Error: Server could not execute the grid search tool. {e}", err=True)
        elif isinstance(e, MCPError):
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

# デフォルト値 (設定ファイルで上書き可能にすべき)
DEFAULT_IMPROVEMENT_THRESHOLD = 0.005 # 0.5% improvement
DEFAULT_MAX_STAGNATION = 5
DEFAULT_SESSION_TIMEOUT = 3600 # 1 hour
DEFAULT_POLL_INTERVAL = 10 # seconds
DEFAULT_JOB_TIMEOUT = 1800 # 30 minutes per job

@improve_app.callback()
def improve_main(ctx: typer.Context, log_level: LogLevel):
    """ Set log level for improve commands. """
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_int, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s')
    logging.info(f"Root logger level set to {log_level.upper()} via basicConfig.")
    load_global_config()
    # ctx.obj = {"log_level": log_level.upper(), "config": CONFIG}

@improve_app.command("start", help="Start or resume the automatic improvement loop.")
def improve(
    # --- Required arguments ---
    server_url: Annotated[str, typer.Option("--server", "-s", help="URL of the MCP server (required for improve).")],
    detector_name: Annotated[str, typer.Option("--detector", help="Name of the detector to improve.")],
    dataset_name: Annotated[str, typer.Option("--dataset", help="Name of the dataset to use for evaluation.")],

    # --- Optional arguments (with defaults) ---
    goal: Annotated[Optional[str], typer.Option("--goal", help="High-level goal for the improvement (e.g., 'Improve F1-score for onset detection').")] = None,
    session_id: Annotated[Optional[str], typer.Option("--session-id", help="Resume an existing improvement session.")] = None,
    # Client-side loop limit (safety net)
    max_client_loops: Annotated[int, typer.Option("--max-loops", help="Maximum number of client-side polling loops (safety net).")] = 100,
    # Server-side session parameters (passed during start_session)
    max_cycles: Annotated[Optional[int], typer.Option("--max-cycles", help="Maximum number of improvement cycles (server-side limit).")] = None,
    improvement_threshold: Annotated[Optional[float], typer.Option("--threshold", help="Minimum F1-score improvement required to reset stagnation counter (server-side).")] = None,
    max_stagnation: Annotated[Optional[int], typer.Option("--max-stagnation", help="Stop if F1-score doesn't improve by the threshold for this many cycles (server-side).")] = None,
    session_timeout: Annotated[Optional[int], typer.Option("--session-timeout", help="Maximum duration for the entire improvement session in seconds (server-side).")] = None,
    # Polling options
    poll_interval: Annotated[int, typer.Option("--poll-interval", help="Interval (seconds) for polling session/job status.")] = DEFAULT_POLL_INTERVAL,
    job_timeout: Annotated[int, typer.Option("--job-timeout", help="Timeout (seconds) for waiting for individual improvement jobs.")] = DEFAULT_JOB_TIMEOUT,
):
    """Starts or resumes the automatic improvement loop via the MCP server."""
    logger = logging.getLogger(__name__)
    client = get_mcp_client(server_url)

    active_session_id = None

    try:
        # 1. Start or Resume Session
        logger.info(f"{'Resuming' if session_id else 'Starting'} improvement session...")
        start_session_input = {
            "detector_name": detector_name,
            "dataset_name": dataset_name,
            "goal": goal,
            "session_id": session_id, # None if starting new
            # Pass server-side limits if provided
            "session_config": {
                key: val for key, val in {
                    "max_cycles": max_cycles,
                    "improvement_threshold": improvement_threshold,
                    "max_stagnation": max_stagnation,
                    "session_timeout": session_timeout
                }.items() if val is not None
            }
        }
        try:
            session_info = client.start_session(**start_session_input)
            active_session_id = session_info.get("session_id")
            if not active_session_id:
                logger.error("Failed to start/resume session: No session_id received.")
                logger.error(f"Server response: {session_info}")
                raise typer.Exit(code=1)
            logger.info(f"Session {'resumed' if session_id else 'started'} successfully. Session ID: {active_session_id}")
            print_session_summary(session_info) # Display initial/resumed state

        except (ToolError, MCPError, Exception) as e:
            log_exception(logger, e, "Failed to start or resume session")
            typer.echo(f"Error communicating with server during session start: {e}", err=True)
            raise typer.Exit(code=1)

        # 2. Server-Driven Improvement Loop (Client-side polling)
        loop_count = 0
        while loop_count < max_client_loops:
            loop_count += 1
            logger.info(f"Client Loop #{loop_count}/{max_client_loops} - Checking session status ({active_session_id})...")

            try:
                # Get current session status from server
                current_session_info = client.get_session_info(session_id=active_session_id)
                session_status = current_session_info.get("status")
                logger.info(f"Session Status: {session_status}")
                print_session_summary(current_session_info) # Display current state

                # Check for terminal states
                if session_status in ["completed", "failed", "stopped", "timed_out"]:
                    logger.info(f"Improvement loop finished with status: {session_status}")
                    break # Exit the client polling loop

                # If session is active/running, trigger the next server-side step
                if session_status in ["active", "running", "needs_action", "needs_initial_evaluation"]: # Add other relevant states
                    logger.info(f"Triggering next improvement step on server for session {active_session_id}...")
                    try:
                        # Call the server-side tool to advance the cycle
                        # TOOL NAME NEEDS TO BE CONFIRMED/IMPLEMENTED ON SERVER
                        advance_tool_name = "advance_improvement_cycle" 
                        job_start_result = client.run_tool(advance_tool_name, {"session_id": active_session_id})
                        job_id = job_start_result.get("job_id")

                        if not job_id:
                            logger.error(f"Failed to start improvement job: No job_id received from server for tool '{advance_tool_name}'.")
                            logger.error(f"Server response: {job_start_result}")
                            # Consider this a session failure or retry?
                            typer.echo(f"Error: Server did not start the improvement job. Check server logs.", err=True)
                            # Attempt to mark session as failed on server?
                            # client.stop_session(session_id=active_session_id, status="failed", reason="ClientError: Failed to start advance job")
                            break # Exit loop on critical error

                        logger.info(f"Improvement job started (Tool: {advance_tool_name}). Job ID: {job_id}. Waiting for completion...")

                        # Wait for the job to complete
                        final_job_info = poll_job_status(client, job_id, poll_interval=poll_interval, timeout=job_timeout)

                        if final_job_info.get("status") == "failed":
                            error_details = final_job_info.get("error_details", {})
                            error_msg = error_details.get("message", "Improvement job failed")
                            logger.error(f"Improvement job {job_id} failed: {error_msg}")
                            typer.echo(f"Error: Improvement step failed on server. Check server logs (Session: {active_session_id}, Job: {job_id}).", err=True)
                            # The session status should ideally be updated to 'failed' by the server upon job failure.
                            # We break the client loop here.
                            break
                        elif final_job_info.get("status") == "completed":
                             logger.info(f"Improvement job {job_id} completed. Fetching updated session status...")
                             # The loop will continue and fetch the new status via get_session_info
                        else:
                             # Timeout or other unexpected status from poll_job_status
                             logger.error(f"Polling for job {job_id} ended with unexpected status: {final_job_info.get('status')}")
                             break # Exit loop

                    except (ToolError, MCPError, TimeoutError, Exception) as job_err:
                        log_exception(logger, job_err, f"Error during improvement step execution (Session: {active_session_id})")
                        typer.echo(f"Error during server communication or job execution: {job_err}", err=True)
                        # Attempt to stop session gracefully?
                        # client.stop_session(session_id=active_session_id, status="failed", reason=f"ClientError: {job_err}")
                        break # Exit loop on error
                else:
                    # Unknown or unexpected session status
                    logger.warning(f"Unexpected session status '{session_status}' for session {active_session_id}. Stopping client loop.")
                    break

            except (ToolError, MCPError, TimeoutError, Exception) as poll_err:
                log_exception(logger, poll_err, f"Error polling session status (Session: {active_session_id})")
                typer.echo(f"Error checking session status: {poll_err}", err=True)
                break # Exit loop on error

            # Wait before next poll
            logger.debug(f"Waiting {poll_interval}s before next status check...")
            time.sleep(poll_interval)

        # Loop finished (completed, failed, stopped, or max_client_loops reached)
        if loop_count >= max_client_loops:
            logger.warning(f"Reached maximum client polling loops ({max_client_loops}). Stopping client. Session {active_session_id} might still be running on the server.")

        logger.info("Fetching final session state...")
        final_session_info = client.get_session_info(session_id=active_session_id)
        print_session_summary(final_session_info)

    except typer.Exit:
        # Handle exits gracefully (e.g., from get_mcp_client or start_session error)
        raise
    except Exception as e:
        # Catch-all for unexpected errors in the main logic
        log_exception(logger, e, f"Unexpected error in improve command (Session: {active_session_id or 'N/A'})")
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        raise typer.Exit(code=1)
    finally:
        # Optional: Clean up resources if needed
        logger.info("Exiting mirai improve command.")

@improve_app.command("status", help="Get the status of an ongoing improvement session.")
def improve_status(
    server_url: Annotated[str, typer.Option("--server", "-s", help="URL of the MCP server.")],
    session_id: Annotated[str, typer.Argument(help="The ID of the session to check.")],
):
    """Gets the status of a specific improvement session."""
    logger = logging.getLogger(__name__)
    client = get_mcp_client(server_url)
    try:
        session_info = client.get_session_info(session_id=session_id)
        logger.info(f"Status for session {session_id}:")
        print_session_summary(session_info)
    except (ToolError, MCPError) as e:
        log_exception(logger, e, f"Failed to get status for session {session_id}")
        typer.echo(f"Error retrieving session status: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        log_exception(logger, e, f"Unexpected error getting session status {session_id}")
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        raise typer.Exit(code=1)

@improve_app.command("stop", help="Manually stop an ongoing improvement session.")
def improve_stop(
    server_url: Annotated[str, typer.Option("--server", "-s", help="URL of the MCP server.")],
    session_id: Annotated[str, typer.Argument(help="The ID of the session to stop.")],
):
    """Manually stops a specific improvement session on the server."""
    logger = logging.getLogger(__name__)
    client = get_mcp_client(server_url)
    try:
        logger.info(f"Attempting to stop session {session_id}...")
        # Assuming a 'stop_session' tool exists on the server
        stop_result = client.stop_session(session_id=session_id, status="stopped", reason="Manual stop requested by client")
        logger.info(f"Stop request sent for session {session_id}.")
        logger.debug(f"Server response: {stop_result}")
        typer.echo(f"Stop request sent for session {session_id}. Use 'improve status' to check final state.")
        # Optionally, poll get_session_info until status is 'stopped'?
    except (ToolError, MCPError) as e:
        log_exception(logger, e, f"Failed to send stop request for session {session_id}")
        typer.echo(f"Error sending stop request: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        log_exception(logger, e, f"Unexpected error stopping session {session_id}")
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        raise typer.Exit(code=1)

def print_session_summary(session_data: Dict[str, Any]):
    """Helper function to print session summary information."""
    # Add logger definition
    logger = logging.getLogger(__name__)
    print("\n--- Session Summary ---")
    session_id = session_data.get('session_id')
    status = session_data.get('status', 'Unknown')
    cycle_state = session_data.get('cycle_state', {})
    best_metrics = session_data.get('best_metrics', {})

    cycle_count = cycle_state.get('cycle_count', 0)
    stagnation_count = cycle_state.get('stagnation_count', 0)
    best_f_measure = best_metrics.get('overall', {}).get('f_measure') # 仮に overall F-measure を指標とする

    logger.info(f"--- Session Summary (ID: {session_id}) ---")
    logger.info(f"Status: {status}")
    logger.info(f"Cycle: {cycle_count}")
    logger.info(f"Stagnation: {stagnation_count}")
    if best_f_measure is not None:
        logger.info(f"Best F-measure: {best_f_measure:.4f}")
    else:
        logger.info("Best F-measure: N/A")
    logger.info("-----------------------------------------")

# ... (if __name__ == "__main__": の部分は mirai.py では不要) 