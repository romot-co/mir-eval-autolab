#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import asyncio
import argparse
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Callable, Coroutine

from mcp.server.fastmcp import FastMCP # Image, Context はツール実装側で使用
import uvicorn
from fastapi import FastAPI # FastAPIを直接インポート

# --- Logging Basic Configuration (Root logger setup) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('mcp_server_main')

# --- Import the logic modules --- #
try:
    from mcp_server_logic import (
        core, db_utils, session_manager, job_manager, llm_tools,
        evaluation_tools, code_tools, improvement_loop, visualization_tools, utils
    )
    # 必要なら json_utils 等も
    from mcp_server_logic.core import JsonNumpyEncoder # If needed globally
except ImportError as e:
    logger.critical(f"必須モジュールのインポートに失敗しました: {e}", exc_info=True)
    logger.critical("PYTHONPATHを確認し、依存関係が正しくインストールされているか確認してください。")
    print(f"FATAL: Failed to import required modules: {e}", file=sys.stderr)
    print("Check PYTHONPATH and ensure dependencies are installed correctly.", file=sys.stderr)
    sys.exit(1)

# --- Optional: Import Extensions --- #
try:
    import mcp_server_extensions
    has_extensions = True
    logger.info("MCP拡張機能をインポートしました")
except ImportError:
    has_extensions = False
    logger.warning("MCP拡張機能 (mcp_server_extensions.py) が見つかりません。拡張機能は無効になります。")


# --- Main Application Logic --- #
def main() -> int:
    # --- Configuration and Initialization ---
    try:
        # 1. Load Configuration
        config = core.load_config() # Default path: config.yaml
        core.log_config(config) # Log loaded config (masked)

        # 2. Initialize Database
        db_utils.init_database(config)

        # 3. Ensure Workspace Directories Exist
        if not core.ensure_workspace_directories(config):
            raise RuntimeError("ワークスペースディレクトリの準備に失敗しました。")

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
    db_path = Path(config['paths']['db']) / db_utils.DB_FILENAME

    # Curry functions with necessary dependencies (config, db_path)
    start_async_job_func = lambda task_coro, tool_name, session_id=None, *args, **kwargs: \
        core.start_async_job(config, task_coro, tool_name, session_id, *args, **kwargs)

    add_history_sync_func = lambda session_id, event_type, event_data, cycle_state_update=None: \
        session_manager._add_session_history_sync(db_path, session_id, event_type, event_data, cycle_state_update)

    # --- Register Tools from Modules --- #
    logger.info("Registering MCP tools...")
    try:
        session_manager.register_session_tools(mcp, config, db_path)
        job_manager.register_job_tools(mcp, config, db_path)
        llm_tools.register_llm_tools(mcp, config, start_async_job_func, add_history_sync_func)
        evaluation_tools.register_evaluation_tools(mcp, config, start_async_job_func, add_history_sync_func)
        code_tools.register_code_tools(mcp, config, start_async_job_func, add_history_sync_func)
        improvement_loop.register_loop_tools(mcp, config, start_async_job_func, add_history_sync_func)
        visualization_tools.register_visualization_tools(mcp, config, start_async_job_func, add_history_sync_func)

        # Register extensions if available
        if has_extensions:
            logger.info("Registering tools from mcp_server_extensions...")
            # Assuming register_extension_tools exists and accepts similar args
            if hasattr(mcp_server_extensions, 'register_extension_tools'):
                 mcp_server_extensions.register_extension_tools(
                     mcp, config, db_path, start_async_job_func, add_history_sync_func
                 )
                 logger.info("Extension tools registered.")
            else:
                 logger.warning("mcp_server_extensions found, but register_extension_tools function is missing.")

        logger.info("All tools registered successfully.")
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
        asyncio.create_task(core.start_job_workers(num_workers, config))
        logger.info("Job workers scheduled to start.")

        # Start Cleanup Thread
        logger.info("Starting cleanup thread...")
        cleanup_sessions_func = session_manager.cleanup_old_sessions
        cleanup_jobs_func = job_manager.cleanup_old_jobs
        core.start_cleanup_thread(config, cleanup_sessions_func, cleanup_jobs_func)
        logger.info("Cleanup thread scheduled to start.")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("MCPサーバーをシャットダウンしています...")
        # Graceful shutdown: Wait for pending jobs? Close executor?
        logger.info("Shutting down ThreadPoolExecutor...")
        core.executor.shutdown(wait=True) # Wait for sync tasks to finish
        # Cancel running async job workers? (Uvicorn might handle this)
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
        uvicorn.run(
            app, # FastAPI アプリケーションインスタンスを渡す
            host=args.host,
            port=args.port,
            log_level=args.log_level # Pass log level to uvicorn
            # reload=True # For development
        )
        logger.info("MCPサーバーが正常にシャットダウンしました。")
        return 0 # Success
    except Exception as e:
        logger.critical(f"サーバー起動エラー: {e}", exc_info=True)
        print(f"FATAL: Failed to start Uvicorn server: {e}", file=sys.stderr)
        return 1 # Failure

if __name__ == "__main__":
    # Optional: Set global JSON encoder if needed outside MCP context
    # _original_encoder = json._default_encoder
    # json._default_encoder = JsonNumpyEncoder()

    exit_code = main()

    # Restore original encoder if changed
    # json._default_encoder = _original_encoder

    sys.exit(exit_code)