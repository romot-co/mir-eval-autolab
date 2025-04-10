import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine, List, Tuple, Awaitable
import time
import re # Added for regex matching
import json # Added for json.loads
import asyncio # Added for run_in_executor

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP
from src.utils.path_utils import (
    # get_detector_path as get_detector_path_core, # Replaced by local helper
    ensure_dir,
    is_safe_path_component,
    get_detectors_src_dir, # Import specific getters
    get_improved_versions_dir,
    validate_path_within_allowed_dirs
)
from src.utils.exception_utils import FileError, ConfigError, MirexError

logger = logging.getLogger('mcp_server.code_tools')

# --- Helper Function --- #

def get_detector_path(
    detectors_dir: Path, # Passed in
    improved_versions_dir: Path, # Passed in
    detector_name: str,
    version: Optional[str] = None,
    # session_id: Optional[str] = None # Removed, not used
) -> Path:
    """
    検出器コードのパスを取得します。
    バージョン指定があれば improved_versions から探し、なければ最新版、それもなければ src から探します。
    指定バージョンが見つからない場合はエラー。最新版が見つからない場合はベースコードにフォールバック。

    Args:
        detectors_dir: ベースとなる検出器ソースディレクトリのパス
        improved_versions_dir: 改善版が保存されるディレクトリのパス
        detector_name: 検出器名
        version: 要求バージョン (例: 'v1') または None (最新版)

    Returns:
        検出器ファイルの絶対パス

    Raises:
        FileNotFoundError: ファイルが見つからない場合
        ValueError: 引数が不正な場合
    """
    if not is_safe_path_component(detector_name):
        raise ValueError(f"Invalid detector name format: {detector_name}")
    if version and not is_safe_path_component(version):
        raise ValueError(f"Invalid version format: {version}")

    src_detectors_dir = detectors_dir
    improved_dir = improved_versions_dir

    # Detector name might be class name or filename. Infer filename.
    if detector_name.endswith('.py'):
        filename_base = detector_name[:-3]
    else:
        # Convert CamelCase to snake_case (simple conversion)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', detector_name)
        filename_base = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    filename = f"{filename_base}.py"
    base_file = src_detectors_dir / filename
    logger.debug(f"Searching for detector '{detector_name}', base file: {base_file}")

    if version:
        # Specific version requested
        # Improved versions are typically stored directly under improved_versions_dir
        # Filename format: {base_filename}_{version}.py
        versioned_filename = f"{filename_base}_{version}.py"
        versioned_file = improved_dir / versioned_filename
        if versioned_file.is_file(): # Check if it's a file
            logger.debug(f"Found specified version: {versioned_file}")
            return versioned_file
        else:
            # If specific version is requested but not found, it's an error
            logger.error(f"Specified version '{version}' for detector '{detector_name}' not found at {versioned_file}")
            raise FileNotFoundError(f"Specified version '{version}' for detector '{detector_name}' not found.")
    else:
        # Latest version requested (version is None or empty)
        latest_version_file: Optional[Path] = None
        all_versions: List[Tuple[str, Path]] = [] # Store tuples (tag, path)

        if improved_dir.is_dir(): # Check if directory exists
            # Collect all improved versions for this detector
            # Format: {base_filename}_<tag>.py
            version_pattern = re.compile(rf"^{re.escape(filename_base)}_(.+)\.py$")
            for f in improved_dir.iterdir(): # Search directly in improved_dir
                 if f.is_file():
                     match = version_pattern.match(f.name)
                     if match:
                          tag = match.group(1)
                          # Basic tag validation (optional but good)
                          if is_safe_path_component(tag):
                               all_versions.append((tag, f)) # Store (tag, path)
                          else:
                               logger.warning(f"Skipping file with potentially unsafe version tag: {f.name}")

        if all_versions:
            # Sort versions to find the latest (using previous logic)
            def sort_key(version_tuple: Tuple[str, Path]):
                tag = version_tuple[0]
                # Try YYYYMMDD_HHMMSS
                match_ts = re.match(r"(\d{8})_(\d{6})", tag)
                if match_ts:
                    try:
                        ts_val = time.mktime(time.strptime(f"{match_ts.group(1)}{match_ts.group(2)}", "%Y%m%d%H%M%S"))
                        return (2, ts_val, tag)
                    except ValueError:
                        pass
                # Try v<unix_ts>
                match_vts = re.match(r"v(\d+)", tag)
                if match_vts:
                    try:
                        return (1, int(match_vts.group(1)), tag)
                    except ValueError:
                         pass
                # Fallback: lexicographical sort
                return (0, 0, tag)

            all_versions.sort(key=sort_key, reverse=True) # Sort descending
            latest_version_tag, latest_version_file = all_versions[0]
            logger.debug(f"Found latest improved version (tag: {latest_version_tag}): {latest_version_file}")
            return latest_version_file

        # No improved versions found, fall back to base code
        if base_file.is_file():
            logger.debug(f"No improved versions found for '{detector_name}'. Using base detector code: {base_file}")
            return base_file
        else:
            # Base code also doesn't exist
            logger.error(f"Detector code not found for '{detector_name}' (neither improved nor base version). Searched in {src_detectors_dir} and {improved_dir}")
            raise FileNotFoundError(f"Detector '{detector_name}' not found.")


# --- Synchronous Task Functions (run in executor) --- #

def _run_get_code(
    # Removed job_id, add_history_sync_func - managed by async wrapper
    detectors_dir: Path,
    improved_versions_dir: Path,
    allowed_base_dirs: List[Path], # For validation
    detector_name: str,
    version: Optional[str] = None,
    # session_id: Optional[str] = None # Managed by wrapper
) -> str:
    """検出器コード取得ジョブ (同期実処理)"""
    # logger.info(f"[Job {job_id}] Getting code for '{detector_name}' (Version: {version or 'latest'})... Session: {session_id}") # Logged in wrapper

    # Input Validation happens in get_detector_path and wrapper

    try:
        # Use the helper function with passed paths
        detector_path_unvalidated = get_detector_path(detectors_dir, improved_versions_dir, detector_name, version)

        # Validate the found path is within allowed directories
        detector_path = validate_path_within_allowed_dirs(
            detector_path_unvalidated,
            allowed_base_dirs,
            check_existence=True,
            check_is_file=True,
            allow_absolute=True # Paths returned by get_detector_path are absolute
        )

        logger.debug(f"Validated detector path: {detector_path}")
        with open(detector_path, "r", encoding="utf-8") as f:
            code_content = f.read()
        # logger.info(f"[Job {job_id}] Successfully retrieved code from {detector_path}") # Logged in wrapper
        return code_content
    except (FileNotFoundError, ValueError, ConfigError, FileError) as e:
        # logger.error(f"[Job {job_id}] Error getting code for '{detector_name}' (Version: {version or 'latest'}): {e}") # Logged in wrapper
        # Re-raise specific errors for wrapper to handle
        raise e
    except Exception as e:
        # logger.error(f"[Job {job_id}] Error getting code for '{detector_name}': {e}", exc_info=True) # Logged in wrapper
        # Wrap unexpected errors
        raise FileError(f"Unexpected error getting code for '{detector_name}': {e}") from e

def _run_save_code(
    # Removed job_id, add_history_sync_func - managed by async wrapper
    improved_versions_dir: Path,
    allowed_base_dirs: List[Path], # For validation
    detector_name: str,
    code: str,
    version_tag: Optional[str] = None,
    # session_id: Optional[str] = None # Managed by wrapper
) -> Dict[str, Any]:
    """コード保存ジョブ (同期実処理)"""
    # logger.info(f"[Job {job_id}] Saving code for '{detector_name}' (Version: {version_tag or 'auto'})... Session: {session_id}") # Logged in wrapper

    # --- Input Validation --- #
    if not is_safe_path_component(detector_name):
        raise ValueError(f"Invalid detector name format: {detector_name}")
    if version_tag and not is_safe_path_component(version_tag):
        raise ValueError(f"Invalid version tag format: {version_tag}")
    if not code or not isinstance(code, str):
        raise ValueError("Code content cannot be empty and must be a string.")
    # --- Validation End --- #

    final_version_tag = version_tag # Keep track of the final tag used

    # Generate version tag if not provided or empty
    if not final_version_tag:
        final_version_tag = time.strftime("%Y%m%d_%H%M%S") # Use YYYYMMDD_HHMMSS format
        logger.info(f"No version tag provided, generated timestamp tag: {final_version_tag}")
        # Re-validate the generated tag
        if not is_safe_path_component(final_version_tag):
            err_msg = f"Generated version tag is invalid: {final_version_tag}"
            logger.critical(f"{err_msg}") # Critical: should not happen
            raise ValueError(err_msg)

    # Infer base filename from detector name (consistent with get_detector_path)
    if detector_name.endswith('.py'):
        filename_base = detector_name[:-3]
    else:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', detector_name)
        filename_base = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    filename = f"{filename_base}_{final_version_tag}.py"
    # Target directory is directly improved_versions_dir
    improved_dir = improved_versions_dir
    file_path_unvalidated = improved_dir / filename

    try:
        # Ensure the improved_versions_dir itself exists (should be handled by getter, but double-check)
        ensure_dir(improved_dir, check_writable=True)

        # --- Final Path Validation --- #
        # Ensure the final save path is safe and within the allowed directory
        file_path = validate_path_within_allowed_dirs(
            file_path_unvalidated,
            allowed_base_dirs,
            check_existence=False, # File should not exist yet
            allow_absolute=True # Path is constructed as absolute
        )
        logger.debug(f"Validated save path: {file_path}")
        # --- Validation End --- #

        # Check if file already exists (optional, depends on desired behavior)
        if file_path.exists():
            # Option 1: Overwrite (current behavior implicitly)
            # Option 2: Raise error
            # Option 3: Generate a slightly different tag (e.g., add suffix)
            logger.warning(f"File already exists, overwriting: {file_path}")
            # raise FileExistsError(f"File with version tag '{final_version_tag}' already exists: {file_path}")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        # logger.info(f"[Job {job_id}] Successfully saved code to {file_path} (Version: {final_version_tag})") # Logged by wrapper

        result = {
            "detector_name": detector_name,
            "version": final_version_tag,
            "file_path": str(file_path), # Return the validated, absolute path as string
            "timestamp": time.time(),
            "status": "saved"
        }
        return result

    except (ValueError, ConfigError, FileError, FileExistsError) as e:
        # logger.error(f"[Job {job_id}] Error saving code to '{file_path_unvalidated}': {e}") # Logged by wrapper
        raise e # Re-raise specific errors for wrapper
    except Exception as e:
        # logger.error(f"[Job {job_id}] Error saving code to '{file_path_unvalidated}': {e}", exc_info=True) # Logged by wrapper
        raise FileError(f"Unexpected error saving code for '{detector_name}': {e}") from e

# --- Asynchronous Wrapper Functions --- #

async def _run_get_code_async(
    job_id: str,
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    detectors_dir: Path,
    improved_versions_dir: Path,
    allowed_base_dirs: List[Path],
    detector_name: str,
    version: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """非同期で _run_get_code を実行し、履歴を管理する"""
    logger.info(f"[Job {job_id}] Getting code for '{detector_name}' (Version: {version or 'latest'})... Session: {session_id}")
    start_time = time.monotonic()
    try:
        await add_history_async_func(session_id, "job_started", {
            "job_id": job_id,
            "tool_name": "get_code",
            "detector_name": detector_name,
            "version": version
        })

        loop = asyncio.get_running_loop()
        code_content = await loop.run_in_executor(
            None, # Use default executor
            _run_get_code,
            detectors_dir,
            improved_versions_dir,
            allowed_base_dirs,
            detector_name,
            version
        )

        duration = time.monotonic() - start_time
        logger.info(f"[Job {job_id}] Successfully retrieved code for '{detector_name}' (Version: {version or 'latest'}) in {duration:.2f}s.")
        await add_history_async_func(session_id, "job_completed", {
            "job_id": job_id,
            "result_summary": f"Code retrieved (length: {len(code_content)})",
            "duration_sec": duration
        })
        return code_content

    except (FileNotFoundError, ValueError, ConfigError, FileError) as e:
        duration = time.monotonic() - start_time
        logger.error(f"[Job {job_id}] Failed to get code for '{detector_name}' (Version: {version or 'latest'}) after {duration:.2f}s: {e}")
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": str(e), "duration_sec": duration})
        raise # Re-raise to be caught by job manager
    except Exception as e:
        duration = time.monotonic() - start_time
        logger.error(f"[Job {job_id}] Unexpected error getting code for '{detector_name}' after {duration:.2f}s: {e}", exc_info=True)
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": f"Unexpected error: {e}", "duration_sec": duration})
        raise FileError(f"Unexpected error in get_code job: {e}") from e

async def _run_save_code_async(
    job_id: str,
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    improved_versions_dir: Path,
    allowed_base_dirs: List[Path],
    detector_name: str,
    code: str,
    version_tag: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """非同期で _run_save_code を実行し、履歴を管理する"""
    logger.info(f"[Job {job_id}] Saving code for '{detector_name}' (Version tag: {version_tag or 'auto'})... Session: {session_id}")
    start_time = time.monotonic()
    try:
        await add_history_async_func(session_id, "job_started", {
            "job_id": job_id,
            "tool_name": "save_code",
            "detector_name": detector_name,
            "version_tag": version_tag,
            "code_length": len(code)
        })

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, # Use default executor
            _run_save_code,
            improved_versions_dir,
            allowed_base_dirs,
            detector_name,
            code,
            version_tag
        )

        duration = time.monotonic() - start_time
        logger.info(f"[Job {job_id}] Successfully saved code for '{detector_name}' as version '{result['version']}' in {duration:.2f}s. Path: {result['file_path']}")
        await add_history_async_func(session_id, "job_completed", {
            "job_id": job_id,
            "result": result,
            "duration_sec": duration
        })
        # Add specific code_saved event as well
        await add_history_async_func(session_id, "code_saved", {
            "detector": result['detector_name'],
            "version": result['version'],
            "path": result['file_path']
        })
        return result

    except (ValueError, ConfigError, FileError, FileExistsError) as e:
        duration = time.monotonic() - start_time
        logger.error(f"[Job {job_id}] Failed to save code for '{detector_name}' (Tag: {version_tag or 'auto'}) after {duration:.2f}s: {e}")
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": str(e), "duration_sec": duration})
        raise # Re-raise to be caught by job manager
    except Exception as e:
        duration = time.monotonic() - start_time
        logger.error(f"[Job {job_id}] Unexpected error saving code for '{detector_name}' after {duration:.2f}s: {e}", exc_info=True)
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": f"Unexpected error: {e}", "duration_sec": duration})
        raise FileError(f"Unexpected error in save_code job: {e}") from e

# --- Tool Registration --- #

def register_code_tools(
    mcp: FastMCP,
    config: Dict[str, Any],
    start_async_job_func: Callable[..., Coroutine[Any, Any, str]], # Expects coroutine returning job_id
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]] # Needed by async wrappers
):
    """コード管理関連のMCPツールを登録"""
    logger.info("Registering code management tools...")

    try:
        # Get necessary base directories using helpers
        detectors_dir = get_detectors_src_dir(config)
        improved_versions_dir = get_improved_versions_dir(config)
        # Define allowed directories for path validation
        allowed_base_dirs = [detectors_dir, improved_versions_dir]
        logger.info(f"Code tools allowed base directories: {allowed_base_dirs}")
    except (ConfigError, FileError) as e:
        logger.error(f"Failed to get necessary directories from config for code tools: {e}", exc_info=True)
        raise ConfigError(f"Code tools setup failed: {e}") from e # Stop registration if setup fails

    @mcp.tool("get_code")
    async def get_code_tool(
        detector_name: str,
        version: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """指定された検出器のコードを取得します (改善版が存在する場合はそれを優先)。"""
        job_id = await start_async_job_func(
            _run_get_code_async, # Call the async wrapper
            tool_name="get_code",
            session_id=session_id,
            # Pass necessary context to the async wrapper
            add_history_async_func=add_history_async_func,
            detectors_dir=detectors_dir,
            improved_versions_dir=improved_versions_dir,
            allowed_base_dirs=allowed_base_dirs,
            detector_name=detector_name,
            version=version
        )
        return {"status": "job_started", "job_id": job_id}

    @mcp.tool("save_code")
    async def save_code_tool(
        detector_name: str,
        code: str,
        version_tag: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """指定されたコードを検出器の改善版として保存します。バージョンタグがなければタイムスタンプを使用します。"""
        job_id = await start_async_job_func(
            _run_save_code_async, # Call the async wrapper
            tool_name="save_code",
            session_id=session_id,
            # Pass necessary context to the async wrapper
            add_history_async_func=add_history_async_func,
            improved_versions_dir=improved_versions_dir,
            allowed_base_dirs=allowed_base_dirs,
            detector_name=detector_name,
            code=code,
            version_tag=version_tag
        )
        return {"status": "job_started", "job_id": job_id}

    logger.info("Code management tools registered.")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 