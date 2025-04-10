#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import asyncio
import argparse
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Callable, Coroutine, Awaitable

from mcp.server.fastmcp import FastMCP # Image, Context はツール実装側で使用
import uvicorn
from fastapi import FastAPI # FastAPIを直接インポート

# --- Logging Basic Configuration (Root logger setup) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('mcp_server_main')

# --- Import the logic modules --- #
try:
    from src.mcp_server_logic import (
        core, db_utils, session_manager, job_manager, llm_tools,
        evaluation_tools, code_tools, improvement_loop, utils
    )
    # 必要なら json_utils 等も
except ImportError as e:
    logger.critical(f"必須モジュールのインポートに失敗しました: {e}", exc_info=True)
    logger.critical("PYTHONPATHを確認し、依存関係が正しくインストールされているか確認してください。")
    print(f"FATAL: Failed to import required modules: {e}", file=sys.stderr)
    print("Check PYTHONPATH and ensure dependencies are installed correctly.", file=sys.stderr)
    sys.exit(1)

# --- Main Application Logic --- #
async def main() -> int:
    # --- Configuration and Initialization ---
    try:
        # 1. Load Configuration
        config = core.load_config() # Default path: config.yaml
        core.log_config(config) # Log loaded config (masked)

        # 2. Initialize Database (非同期で実行)
        await db_utils.init_database(config)

        # 3. Ensure Workspace Directories Exist (core.load_config 内で処理されるため削除)
        # if not core.ensure_workspace_directories(config):
        #     raise RuntimeError("ワークスペースディレクトリの準備に失敗しました。")

    except Exception as e:
        logger.critical(f"サーバー初期化中に致命的なエラー: {e}", exc_info=True)
        print(f"FATAL: Server initialization failed: {e}", file=sys.stderr)
        return 1

    # --- FastAPI App Instance --- #
    # Get server config first
    server_config = config.get('server', {})
    # FastAPI アプリケーションを直接作成
    app = FastAPI(
        title=server_config.get('name', "MIREX Algorithm Improver"),
        description="Server for automatic MIREX algorithm improvement using LLMs and evaluation.",
        version="0.1.0"
    )

    # --- MCP Server Instance (Tools only) --- #
    mcp = FastMCP(
        server_config.get('name', "MIREX Algorithm Improver"),
        dependencies=server_config.get('dependencies', [])
    )

    # --- Prepare Dependencies for Tool Registration --- #
    db_path = Path(config['paths']['db_dir']) / db_utils.DB_FILENAME

    # Curry functions with necessary dependencies (config, db_path)
    start_async_job_func = lambda task_coro_func, tool_name, session_id=None, *args, **kwargs: \
        job_manager.start_async_job(config, db_path, task_coro_func, tool_name, session_id, *args, **kwargs)

    add_history_async_func: Callable[..., Awaitable[None]] = \
        lambda session_id, event_type, event_data, cycle_state_update=None: \
            session_manager.add_session_history(config, db_path, session_id, event_type, event_data, cycle_state_update)

    # --- Register Tools from Modules --- #
    logger.info("Registering MCP tools...")
    try:
        session_manager.register_session_tools(mcp, config, db_path)
        job_manager.register_job_tools(mcp, config, db_path)
        llm_tools.register_llm_tools(mcp, config, start_async_job_func, add_history_async_func)
        evaluation_tools.register_evaluation_tools(mcp, config, start_async_job_func, add_history_async_func)
        code_tools.register_code_tools(mcp, config, start_async_job_func, add_history_async_func)
        improvement_loop.register_loop_tools(mcp, config, start_async_job_func, add_history_async_func)
        # visualization_tools registration removed

        # Register extensions section removed

        logger.info("All required tools registered successfully.") # メッセージ修正
    except Exception as e:
        logger.critical(f"ツール登録中にエラー: {e}", exc_info=True)
        print(f"FATAL: Tool registration failed: {e}", file=sys.stderr)
        return 1

    # --- Setup Startup and Shutdown Events (using FastAPI instance) --- #
    @app.on_event("startup")
    async def startup_event():
        # Start Job Workers
        num_workers = config.get('resource_limits', {}).get('max_concurrent_jobs', 2)
        logger.info(f"Starting {num_workers} job workers...")
        db_path_startup = Path(config['paths']['db_dir']) / db_utils.DB_FILENAME
        asyncio.create_task(job_manager.start_job_workers(num_workers, config, db_path_startup))
        logger.info("Job workers scheduled to start.")

        # Start Cleanup Thread
        logger.info("Starting cleanup thread...")
        cleanup_sessions_func = session_manager.cleanup_old_sessions
        cleanup_jobs_func = job_manager.cleanup_old_jobs
        cleanup_workspace_func = core.cleanup_workspace_files # core の関数を直接参照
        asyncio.create_task(core.start_cleanup_task(
            config,
            cleanup_sessions_func, # async def cleanup_old_sessions
            cleanup_jobs_func,     # async def cleanup_old_jobs
            cleanup_workspace_func # async def cleanup_workspace_files
        ))
        logger.info("Cleanup task scheduled to start.")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("MCPサーバーをシャットダウンしています...")
        # Graceful shutdown: Wait for pending jobs? Close executor?
        # logger.info("Shutting down ThreadPoolExecutor...")
        # core.executor.shutdown(wait=True) # Wait for sync tasks to finish
        # Cancel running async job workers? (Uvicorn might handle this)
        # TODO: クリーンアップタスクのキャンセル処理を追加
        logger.info("Shutdown complete.")

    # --- Include MCP Router --- #
    # --- Mount MCP SSE App --- #
    # mcp.sse_app() が返すASGIアプリをルートパス("/")にマウントする
    # (必要に応じてパスは変更可能)
    if hasattr(mcp, 'sse_app'):
        # Ensure sse_app returns an ASGI app suitable for mounting
        sse_application = mcp.sse_app()
        app.mount("/", app=sse_application) # Mount at root path
        logger.info("Mounted MCP SSE application at path '/'.")
    else:
        logger.error("FastMCP instance does not have an 'sse_app' method. Tool endpoints will not be available.")

    # --- Parse Arguments and Start Server --- #
    parser = argparse.ArgumentParser(description="MIREX MCP Server")
    parser.add_argument("--port", type=int, default=server_config.get('port', 5002), help="Server port")
    parser.add_argument("--host", default=server_config.get('host', "0.0.0.0"), help="Server host")
    parser.add_argument("--log-level", default=server_config.get('log_level', "info").lower(),
                        choices=["debug", "info", "warning", "error", "critical"], help="Uvicorn log level")
    # Example: Add config file override argument
    # parser.add_argument("--config", type=str, help="Path to config file (overrides default)")
    args = parser.parse_args()

    # Apply log level from args to root logger (affects application logs)
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level_int)
    logger.info(f"Root logger level set to: {args.log_level.upper()}")

    logger.info(f"MCPサーバーを起動します (Host: {args.host}, Port: {args.port})..." )
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("サーバーが Ctrl+C により中断されました。")
        exit_code = 0 # 通常終了扱い
    except Exception as e:
        logger.critical(f"サーバー実行中に予期せぬエラーが発生: {e}", exc_info=True)
        print(f"FATAL: Unhandled exception during server execution: {e}", file=sys.stderr)
        exit_code = 1 # 異常終了

    # Restore original encoder if changed
    # json._default_encoder = _original_encoder

    sys.exit(exit_code)

if __name__ == "__main__":
    # Optional: Set global JSON encoder if needed outside MCP context (非推奨)
    # _original_encoder = json._default_encoder
    # json._default_encoder = NumpyEncoder() # <-- ここも変更

    exit_code = main()

    # Restore original encoder if changed
    # json._default_encoder = _original_encoder

    sys.exit(exit_code)