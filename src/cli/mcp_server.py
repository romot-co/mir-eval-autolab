#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import asyncio
import logging
import sys
import traceback
from contextlib import asynccontextmanager  # lifespan用
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine, Dict, Optional

from mcp.server.fastmcp import Context, FastMCP  # Contextもインポートしておく

# import uvicorn # Uvicorn削除
# from starlette.applications import Starlette # Starlette削除
# from starlette.routing import Mount # Mount削除

# --- Logging Basic Configuration (Root logger setup) ---
# 引数パース前に設定するため、デフォルトレベルで初期化
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("mcp_server_main")

# --- Import the logic modules --- #
try:
    from src.mcp_server_logic import (
        code_tools,
        core,
        db_utils,
        evaluation_tools,
        job_manager,
        llm_tools,
        schemas,
        session_manager,
    )
except ImportError as e:
    # ロガーが初期化されていることを期待
    logger.critical(f"必須モジュールのインポートに失敗しました: {e}", exc_info=True)
    logger.critical(
        "PYTHONPATHを確認し、依存関係が正しくインストールされているか確認してください。"
    )
    print(f"FATAL: Failed to import required modules: {e}", file=sys.stderr)
    print(
        "Check PYTHONPATH and ensure dependencies are installed correctly.",
        file=sys.stderr,
    )
    sys.exit(1)

# --- グローバル変数 (Lifespan と Tool 関数で共有) --- #
app_config: Optional[Dict[str, Any]] = None
app_db_path: Optional[Path] = None
background_tasks: list = []  # バックグラウンドタスクを管理するリスト


# --- Lifespan for Background Tasks (for FastMCP) ---
@asynccontextmanager
async def lifespan_manager(server: FastMCP):
    global app_config, app_db_path, background_tasks  # グローバル変数を参照
    logger.info("MCP Lifespan startup: Starting background tasks...")

    # Lifespan開始時には app_config と app_db_path が設定済みのはず
    if app_config is None or app_db_path is None:
        logger.error(
            "Lifespan startup: Config or DB path is None. This should not happen."
        )
        raise RuntimeError(
            "Configuration or DB path not initialized before lifespan startup."
        )

    # バックグラウンドタスクの開始
    try:
        num_workers = app_config.get("resource_limits", {}).get(
            "max_concurrent_jobs", 2
        )
        logger.info(f"Starting {num_workers} job workers...")
        job_worker_task = asyncio.create_task(
            job_manager.start_job_workers(num_workers, app_config, app_db_path)
        )
        background_tasks.append(job_worker_task)
        logger.info("Job workers scheduled.")

        logger.info("Starting cleanup thread...")
        cleanup_sessions_func = session_manager.cleanup_old_sessions
        cleanup_jobs_func = None  # job_manager.cleanup_old_jobs から None に変更
        cleanup_workspace_func = core.cleanup_workspace_files
        cleanup_task = asyncio.create_task(
            core.start_cleanup_task(
                app_config,
                cleanup_sessions_func,
                cleanup_jobs_func,  # None を渡す
                cleanup_workspace_func,
            )
        )
        background_tasks.append(cleanup_task)
        logger.info("Cleanup task scheduled (job cleanup disabled).")
        logger.info("MCP Lifespan startup complete.")
    except Exception as e:
        logger.critical(
            f"Failed to start background tasks during lifespan startup: {e}",
            exc_info=True,
        )
        raise  # エラーを再送出して起動を中止

    try:
        yield  # サーバー実行フェーズへ (コンテキストは渡さない)
    finally:
        # --- Shutdown Phase --- #
        logger.info("MCP Lifespan shutdown: Cleaning up background tasks...")
        cancelled_tasks = []
        for task in background_tasks:
            if task and not task.done():
                task.cancel()
                cancelled_tasks.append(task)
            elif task:
                logger.debug(f"Task {task.get_name()} already done.")
            else:
                logger.warning("Found None in background_tasks list during shutdown.")

        if cancelled_tasks:
            try:
                # キャンセルされたタスクが完了するのを待つ
                # gatherに空リストを渡すとエラーになるためチェック
                logger.info(
                    f"Waiting for {len(cancelled_tasks)} background tasks to cancel..."
                )
                await asyncio.gather(*cancelled_tasks, return_exceptions=True)
                logger.info("Cancelled background tasks have finished.")
            except asyncio.CancelledError:
                # gather 自体がキャンセルされる可能性は低いが念のため
                logger.warning(
                    "Gather operation was cancelled during background task shutdown."
                )
            except Exception as e:
                logger.error(
                    f"Error during background task cleanup (gather): {e}", exc_info=True
                )
        else:
            logger.info("No active background tasks needed cancellation.")

        # タスクリストをクリア
        background_tasks = []
        logger.info("MCP Lifespan shutdown complete.")


# --- Server Setup Function --- #
async def setup_server(args: argparse.Namespace) -> Optional[FastMCP]:
    """Performs asynchronous setup: config, DB, MCP instance, tools."""
    global app_config, app_db_path  # グローバル変数を設定

    # --- Configuration and Initialization --- #
    try:
        logger.info("Initializing config and DB...")
        app_config = core.load_config()
        server_config = app_config.get("server", {})
        core.log_config(app_config)
        await db_utils.init_database(app_config)
        app_db_path = Path(app_config["paths"]["db"]) / db_utils.DB_FILENAME
        logger.info("Config and DB initialized.")
    except Exception as e:
        logger.critical(f"サーバー初期化中に致命的なエラー: {e}", exc_info=True)
        print(f"FATAL: Server initialization failed: {e}", file=sys.stderr)
        return None  # 初期化失敗

    # --- MCP Server Instance with Lifespan --- #
    try:
        mcp = FastMCP(
            server_config.get("name", "MIREX Algorithm Improver"),
            dependencies=server_config.get("dependencies", []),
            lifespan=lifespan_manager,
        )
        logger.info(
            f"Created FastMCP server instance '{mcp.name}' with lifespan manager."
        )
    except Exception as e:
        logger.critical(f"FastMCP インスタンスの作成に失敗: {e}", exc_info=True)
        return None  # インスタンス作成失敗

    # --- Prepare Dependencies for Tool Registration --- #
    # config と db_path はグローバル変数として利用可能
    start_async_job_func = lambda task_coro_func, tool_name, session_id=None, *args, **kwargs: job_manager.start_async_job(
        app_config, app_db_path, task_coro_func, tool_name, session_id, *args, **kwargs
    )

    add_history_async_func: Callable[..., Awaitable[None]] = (
        lambda session_id, event_type, event_data, cycle_state_update=None: session_manager.add_session_history(
            app_config,
            app_db_path,
            session_id,
            event_type,
            event_data,
            cycle_state_update,
        )
    )

    # --- Register Tools from Modules --- #
    logger.info("Registering MCP tools...")
    try:
        evaluation_tools.register_evaluation_tools(
            mcp, app_config, start_async_job_func, add_history_async_func
        )
        code_tools.register_code_tools(
            mcp, app_config, start_async_job_func, app_db_path
        )
        session_manager.register_session_tools(mcp, app_config, app_db_path)
        llm_tools.register_llm_tools(
            mcp, app_config, start_async_job_func, add_history_async_func
        )
        logger.info("All required tools registered successfully.")
    except Exception as e:
        logger.critical(f"ツール登録中にエラーが発生しました: {e}", exc_info=True)
        return None  # ツール登録失敗

    return mcp  # 設定済みのMCPインスタンスを返す


# --- Main Execution Block --- #
if __name__ == "__main__":
    # --- Argument Parsing --- #
    parser = argparse.ArgumentParser(description="MIREX MCP Server")
    parser.add_argument(
        "--port",
        type=int,
        default=5002,
        help="Server port (Note: may not be used by mcp.run())",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (Note: may not be used by mcp.run())",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Application log level",
    )
    args = parser.parse_args()

    # --- Apply Log Level --- #
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level_int)  # ルートロガーに適用
    logger.info(f"Root logger level set to: {args.log_level.upper()}")

    # --- Run Setup --- #
    mcp_instance: Optional[FastMCP] = None
    exit_code = 0
    try:
        logger.info("Running server setup...")
        # 非同期セットアップを実行し、MCPインスタンスを取得
        mcp_instance = asyncio.run(setup_server(args))

        if mcp_instance is None:
            logger.critical("Server setup failed. Exiting.")
            exit_code = 1
        else:
            # --- Start Server using mcp.run() --- #
            logger.info(f"MCPサーバーを起動します (using mcp.run())...")
            # mcp.run() を直接呼び出し、イベントループ管理を任せる
            mcp_instance.run()
            logger.info("MCP Server has finished running.")
            # mcp.run() が正常終了した場合の終了コードは 0 のまま

    except KeyboardInterrupt:
        logger.info("サーバーが Ctrl+C により中断されました。")
        # lifespan の shutdown が適切に処理されることを期待
        exit_code = 0
    except Exception as e:
        logger.critical(
            f"サーバー実行中に予期せぬエラーが発生 (main level): {e}\n{traceback.format_exc()}"
        )
        print(
            f"FATAL: Unhandled exception during server execution: {e}", file=sys.stderr
        )
        traceback.print_exc(file=sys.stderr)
        exit_code = 1

    logger.info(f"Exiting with code {exit_code}")
    sys.exit(exit_code)
