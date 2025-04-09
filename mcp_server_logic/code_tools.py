import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine, List, Tuple
import time
import re # Added for regex matching
import json # Added for json.loads

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP
from src.utils.path_utils import get_detector_path as get_detector_path_core
from src.utils.path_utils import ensure_dir
from src.utils.exception_utils import FileError, ConfigError, MirexError

logger = logging.getLogger('mcp_server.code_tools')

# --- Helper Function --- #

def get_detector_path(
    detectors_dir: Path,
    improved_versions_dir: Path,
    detector_name: str,
    version: Optional[str] = None,
    session_id: Optional[str] = None
) -> Path:
    """
    検出器コードのパスを取得します。
    バージョン指定があれば improved_versions から探し、なければ最新版、それもなければ src から探します。
    指定バージョンが見つからない場合はエラー。最新版が見つからない場合はベースコードにフォールバック。
    """
    src_detectors_dir = detectors_dir
    improved_dir = improved_versions_dir

    base_file = src_detectors_dir / f"{detector_name}.py"

    if version:
        # Specific version requested
        versioned_file = improved_dir / f"{detector_name}_{version}.py"
        if versioned_file.is_file(): # Check if it's a file
            logger.debug(f"Found specified version: {versioned_file}")
            return versioned_file
        else:
            # If specific version is requested but not found, it's an error
            logger.error(f"Specified version '{version}' for detector '{detector_name}' not found in {improved_dir}")
            raise FileNotFoundError(f"Specified version '{version}' for detector '{detector_name}' not found.")
    else:
        # Latest version requested (version is None or empty)
        latest_version_file: Optional[Path] = None
        all_versions: List[Tuple[str, Path]] = [] # Store tuples (tag, path)

        if improved_dir.is_dir(): # Check if directory exists
            # Collect all improved versions for this detector
            for f in improved_dir.glob(f"{detector_name}_*.py"):
                 if f.is_file():
                     # Extract version tag (everything after the first underscore)
                     match = re.match(rf"{re.escape(detector_name)}_(.+)\.py", f.name)
                     if match:
                          tag = match.group(1)
                          all_versions.append((tag, f)) # Store (tag, path)

        if all_versions:
            # Sort versions to find the latest
            def sort_key(version_tuple: Tuple[str, Path]):
                tag = version_tuple[0]
                # Try YYYYMMDD_HHMMSS
                match_ts = re.match(r"(\d{8})_(\d{6})", tag)
                if match_ts:
                    try:
                        # Ensure valid date/time before creating timestamp
                        ts_val = time.mktime(time.strptime(f"{match_ts.group(1)}{match_ts.group(2)}", "%Y%m%d%H%M%S"))
                        return (2, ts_val, tag) # Prioritize timestamp format (type 2), use value, fallback to tag string
                    except ValueError:
                        pass # Invalid date/time format
                # Try v<unix_ts>
                match_vts = re.match(r"v(\d+)", tag)
                if match_vts:
                    try:
                        return (1, int(match_vts.group(1)), tag) # Prioritize v<timestamp> (type 1)
                    except ValueError:
                         pass # Invalid integer
                # Fallback: use the tag string itself for lexicographical sort
                return (0, 0, tag) # Lowest priority (type 0), use tag string

            all_versions.sort(key=sort_key, reverse=True) # Sort descending (latest first)
            latest_version_tag, latest_version_file = all_versions[0]
            logger.debug(f"Found latest improved version (tag: {latest_version_tag}): {latest_version_file}")
            return latest_version_file

        # No improved versions found, fall back to base code
        if base_file.is_file():
            logger.debug(f"No improved versions found for '{detector_name}'. Using base detector code: {base_file}")
            return base_file
        else:
            # Base code also doesn't exist
            logger.error(f"Detector code not found for '{detector_name}' (neither improved nor base version)")
            raise FileNotFoundError(f"Detector '{detector_name}' not found in {src_detectors_dir} or {improved_dir}")


# --- Synchronous Task Functions --- #

def _run_get_code(
    job_id: str,
    detectors_dir: Path,
    improved_versions_dir: Path,
    add_history_sync_func: Callable,
    detector_name: str,
    version: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """検出器コード取得ジョブ (同期)"""
    logger.info(f"[Job {job_id}] Getting code for '{detector_name}' (Version: {version or 'latest'})... Session: {session_id}")
    try:
        # Pass session_id if needed by get_detector_path in future hierarchical structure
        detector_path = get_detector_path(detectors_dir, improved_versions_dir, detector_name, version, session_id)
        with open(detector_path, "r", encoding="utf-8") as f:
            code_content = f.read()
        logger.info(f"[Job {job_id}] Successfully retrieved code from {detector_path}")
        # 履歴は不要かもしれない (読み取り操作なので)
        return code_content
    except FileNotFoundError as e:
        logger.error(f"[Job {job_id}] Code not found error for '{detector_name}' (Version: {version or 'latest'}): {e}")
        raise # エラーをそのまま Job Worker に伝える
    except ConfigError as e: # Catch ConfigError from get_detector_path (if it still raises it)
         logger.error(f"[Job {job_id}] Configuration error getting code for '{detector_name}': {e}")
         raise
    except Exception as e:
        logger.error(f"[Job {job_id}] Error getting code for '{detector_name}': {e}", exc_info=True)
        raise FileError(f"Failed to get code: {e}") from e

def _run_save_code(
    job_id: str,
    improved_versions_dir: Path,
    add_history_sync_func: Callable,
    detector_name: str,
    code: str,
    version_tag: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """コード保存ジョブ (同期)"""
    logger.info(f"[Job {job_id}] Saving code for '{detector_name}' (Version: {version_tag or 'auto'})... Session: {session_id}")

    final_version_tag = version_tag # Keep track of the final tag used

    # Generate version tag if not provided or empty
    if not final_version_tag:
        final_version_tag = time.strftime("%Y%m%d_%H%M%S") # Use YYYYMMDD_HHMMSS format
        logger.info(f"[Job {job_id}] No version tag provided, generated timestamp tag: {final_version_tag}")

    # Validate final version tag format (allow YYYYMMDD_HHMMSS, v<ts>, etc.)
    # Keep existing validation flexible, but ensure it's not empty or problematic
    if not final_version_tag or not re.match(r"^[a-zA-Z0-9_\\-\\.]+$", final_version_tag) or '/' in final_version_tag or '\\\\' in final_version_tag:
         err_msg = f"Invalid version tag generated or provided: '{final_version_tag}'"
         logger.error(f"[Job {job_id}] {err_msg}")
         raise ValueError(err_msg)

    improved_dir = improved_versions_dir

    try:
        ensure_dir(improved_dir) # path_utils.ensure_dir を使用
    except Exception as e:
         logger.error(f"[Job {job_id}] Failed to ensure directory {improved_dir}: {e}", exc_info=True)
         raise FileError(f"Failed to create or access directory: {improved_dir}") from e

    # Validate detector name (basic check)
    if not detector_name or not re.match(r"^[a-zA-Z0-9_\\-]+$", detector_name) or '/' in detector_name or '\\\\' in detector_name:
         err_msg = f"Invalid detector name: {detector_name}"
         logger.error(f"[Job {job_id}] {err_msg}")
         raise ValueError(err_msg)

    # Construct filename using the final version tag
    filename = f"{detector_name}_{final_version_tag}.py"
    file_path = improved_dir / filename

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"[Job {job_id}] Successfully saved code to {file_path} (Version: {final_version_tag})") # Log the final version used

        result = {
            "detector_name": detector_name,
            "version": final_version_tag, # Use the final version tag in the result
            "file_path": str(file_path),
            "timestamp": time.time(), # Keep original timestamp
            "status": "saved"
        }

        # Add history using the final version tag
        if session_id:
            try:
                add_history_sync_func(session_id, "code_saved", {"detector": detector_name, "version": final_version_tag, "path": str(file_path)})
            except Exception as hist_e:
                logger.warning(f"[Job {job_id}] Failed to add code saved history: {hist_e}")

        return result

    except Exception as e:
        logger.error(f"[Job {job_id}] Error saving code to '{file_path}': {e}", exc_info=True)
        # Add failure history using the intended version tag
        if session_id:
             try:
                 add_history_sync_func(session_id, "code_save_failed", {"detector": detector_name, "version": final_version_tag, "error": str(e)})
             except Exception as hist_e:
                 logger.warning(f"[Job {job_id}] Failed to add code save failed history: {hist_e}")
        raise FileError(f"Failed to save code: {e}") from e

# --- Tool Registration --- #

def register_code_tools(
    mcp: FastMCP,
    config: Dict[str, Any],
    start_async_job_func: Callable[..., Coroutine[Any, Any, str]],
    add_history_sync_func: Callable # Needed by task funcs
):
    """コード管理関連のMCPツールを登録"""
    logger.info("Registering code management tools...")

    # Extract necessary paths from config once
    paths_config = config.get('paths', {})
    detectors_dir = Path(paths_config.get('detectors', ''))
    improved_versions_dir = Path(paths_config.get('improved_versions', ''))

    # Check if paths are valid (optional, but good practice)
    if not detectors_dir.is_dir():
        logger.warning(f"Detector source directory not found or not a directory: {detectors_dir}")
        # Optionally raise ConfigError here if base detectors are mandatory
    if not improved_versions_dir:
         logger.warning(f"Improved versions directory path is empty in config.")
         # ensure_dir in _run_save_code will handle creation if possible

    # get_code は同期的にも実行可能かもしれないが、一貫性のため非同期ジョブとする
    @mcp.tool("get_code")
    async def get_code_tool(detector_name: str, version: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """指定された検出器のコードを取得します。バージョンが指定されない場合は最新版を取得します。"""
        job_id = await start_async_job_func( # Use await for async function call
            _run_get_code,
            tool_name="get_code",
            session_id_for_job=session_id, # Pass session_id for job tracking if needed
            # Task-specific arguments below
            detectors_dir=detectors_dir,
            improved_versions_dir=improved_versions_dir,
            add_history_sync_func=add_history_sync_func,
            detector_name=detector_name,
            version=version,
            session_id=session_id
        )
        return {"job_id": job_id, "status": "pending"}

    @mcp.tool("save_code")
    async def save_code_tool(detector_name: str, code: str, version: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]: # API uses 'version'
        """指定された検出器のコードを指定されたバージョン (または自動生成) で保存します。"""
        job_id = await start_async_job_func( # Use await for async function call
            _run_save_code,
            tool_name="save_code",
            session_id_for_job=session_id,
            # Task-specific arguments below
            improved_versions_dir=improved_versions_dir,
            add_history_sync_func=add_history_sync_func,
            detector_name=detector_name,
            code=code,
            version_tag=version, # Pass API 'version' as 'version_tag' to task func
            session_id=session_id
        )
        return {"job_id": job_id, "status": "pending"}

    logger.info("Code management tools registered.")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 