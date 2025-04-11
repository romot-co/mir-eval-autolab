import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine, List, Tuple, Awaitable
import time
import re # Added for regex matching
import json # Added for json.loads
import asyncio # Added for run_in_executor
from functools import partial
import subprocess # 追加: Gitコマンド実行用
import traceback # 追加: エラートレースバック取得用

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP
from src.utils.path_utils import (
    # get_detector_path as get_detector_path_core, # Replaced by local helper
    ensure_dir,
    is_safe_path_component,
    get_detectors_src_dir, # Import specific getters
    get_improved_versions_dir,
    validate_path_within_allowed_dirs,
    get_workspace_dir # get_workspace_dir をインポートに追加
)
from src.utils.exception_utils import FileError, ConfigError, MirexError
from .. import schemas # schemas モジュール自体をインポート

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
    # --- 追加: Git連携用情報 --- #
    session_id: Optional[str] = None,
    parent_version: Optional[str] = None,
    changes_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """コード保存ジョブ (同期実処理)。Git連携を含む。"""
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

    # --- Git Helper Function --- #
    def _run_git_command(cmd: List[str], cwd: Path, check: bool = True, ignore_errors: bool = False) -> Tuple[int, str, str]:
        """指定されたディレクトリでGitコマンドを実行する。"""
        try:
            logger.debug(f"Running git command: {' '.join(cmd)} in {cwd}")
            result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True, encoding='utf-8') # check=False initially
            if check and result.returncode != 0:
                # Raise CalledProcessError manually if check is True and failed
                raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
            
            stdout_clean = result.stdout.strip() if result.stdout else ""
            stderr_clean = result.stderr.strip() if result.stderr else ""
            logger.debug(f"Git command stdout:\n{stdout_clean}")
            if stderr_clean:
                # Log stderr as warning unless return code was non-zero
                log_level = logging.ERROR if result.returncode != 0 else logging.WARNING
                logger.log(log_level, f"Git command stderr:\n{stderr_clean}")
            return result.returncode, stdout_clean, stderr_clean
        except FileNotFoundError:
            logger.error("Git command not found. Please ensure Git is installed and in the system PATH.")
            if ignore_errors: return -1, "", "Git not found"
            raise
        except subprocess.CalledProcessError as e:
            # Logged already by the logic above if check=True
            # logger.error(f"Git command failed: {' '.join(cmd)}")
            # logger.error(f"Return code: {e.returncode}")
            # logger.error(f"Stderr: {e.stderr}")
            if ignore_errors: return e.returncode, e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else ""
            raise
        except Exception as e:
            logger.error(f"Unexpected error running git command: {e}", exc_info=True)
            if ignore_errors: return -1, "", f"Unexpected error: {e}"
            raise

    # --- Git Repository Initialization Check --- #
    is_git_repo = False
    if improved_dir.is_dir(): # Only check/init if the base dir exists
        git_dir = improved_dir / ".git"
        is_git_repo = git_dir.is_dir()
        if not is_git_repo:
            logger.info(f"Directory {improved_dir} is not a Git repository. Initializing...")
            try:
                # ensure_dir(improved_dir) # Ensure base directory exists - redundant if improved_dir.is_dir() is true
                _run_git_command(["git", "init"], cwd=improved_dir, check=True)
                is_git_repo = True
                logger.info(f"Successfully initialized Git repository in {improved_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize Git repository in {improved_dir}: {e}. Proceeding without Git.")
                is_git_repo = False # Proceed without Git if init fails
    else:
         logger.warning(f"Improved versions directory {improved_dir} does not exist. Cannot perform Git operations.")

    # --- File Path Validation and Saving --- #
    file_path: Optional[Path] = None
    try:
        # Ensure the improved_versions_dir itself exists (should be handled by getter, but double-check)
        ensure_dir(improved_dir, check_writable=True)

        # --- Final Path Validation --- #
        file_path = validate_path_within_allowed_dirs(
            file_path_unvalidated,
            allowed_base_dirs,
            check_existence=False, # File should not exist yet
            allow_absolute=True # Path is constructed as absolute
        )
        logger.debug(f"Validated save path: {file_path}")
        # --- Validation End --- #

        # Check if file already exists
        if file_path.exists():
            logger.warning(f"File {file_path} already exists. Overwriting.")

        # --- File Saving --- #
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"Successfully saved code to {file_path}")

    except Exception as e:
        logger.error(f"Error validating path or saving code to {file_path_unvalidated}: {e}", exc_info=True)
        # Re-raise the error to be caught by the async wrapper
        raise FileError(f"Failed to validate path or save code to {file_path_unvalidated}: {e}") from e

    # --- Git Commit --- #
    commit_hash: Optional[str] = None
    if is_git_repo and file_path:
        try:
            # Git Add
            relative_path_str = str(file_path.relative_to(improved_dir))
            _run_git_command(["git", "add", relative_path_str], cwd=improved_dir, check=True)

            # Git Commit
            commit_message = f"Save detector '{detector_name}' version '{final_version_tag}'"
            if session_id:
                commit_message += f"\n\nSession: {session_id}"
            if parent_version:
                commit_message += f"\nParent: {parent_version}"
            if changes_summary:
                commit_message += f"\n\nSummary:\n{changes_summary}"

            _run_git_command(["git", "commit", "-m", commit_message], cwd=improved_dir, check=True)

            # Get commit hash
            ret_code, hash_out, hash_err = _run_git_command(["git", "rev-parse", "HEAD"], cwd=improved_dir, check=True)
            commit_hash = hash_out
            logger.info(f"Successfully committed {relative_path_str} with hash: {commit_hash}")

        except Exception as e:
            # Log error but don't fail the whole operation, just skip Git info
            logger.error(f"Git operation failed after saving file {file_path}: {e}. Commit hash will be missing.", exc_info=True)
            # Fall through, commit_hash remains None

    # Return result including the version tag and commit hash
    result_data = {
        "file_path": str(file_path) if file_path else None, # file_path might be None if saving failed early
        "version_tag": final_version_tag,
        "commit_hash": commit_hash # Can be None if Git failed or skipped
    }
    return result_data

# --- Asynchronous Task Wrapper Functions --- #

async def _run_get_code_async(
    job_id: str,
    # add_history_async_func: Callable[..., Awaitable[None]], # ★ 削除
    detectors_dir: Path,
    improved_versions_dir: Path,
    allowed_base_dirs: List[Path],
    detector_name: str,
    version: Optional[str] = None,
    session_id: Optional[str] = None
) -> str: # ★ 戻り値を直接コード文字列に
    """コード取得ジョブ (非同期ラッパー)。履歴記録は削除。"""
    start_time = time.monotonic()
    logger.info(f"[Job {job_id}] Starting get_code for '{detector_name}' (Version: {version or 'latest'})... Session: {session_id}")
    error_msg: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # --- 履歴記録削除 ---
    # await add_history_async_func(
    #     session_id=session_id,
    #     event_type="code_load.started",
    #     event_data=schemas.CodeLoadStartedData(
    #         job_id=job_id,
    #         detector_name=detector_name,
    #         version=version
    #     ).model_dump()
    # )

    try:
        loop = asyncio.get_running_loop()
        # partial を使って同期関数に必要な引数を渡す
        sync_task = partial(
            _run_get_code,
            detectors_dir=detectors_dir,
            improved_versions_dir=improved_versions_dir,
            allowed_base_dirs=allowed_base_dirs,
            detector_name=detector_name,
            version=version
        )
        # 同期関数を実行
        code_content = await loop.run_in_executor(None, sync_task)

        logger.info(f"[Job {job_id}] Code get successful for '{detector_name}'. Duration: {time.monotonic() - start_time:.2f}s")

        # --- 履歴記録削除 ---
        # await add_history_async_func(
        #     session_id=session_id,
        #     event_type="code_load.completed",
        #     event_data=schemas.CodeLoadCompleteData(
        #         job_id=job_id,
        #         detector_name=detector_name,
        #         version=version, # TODO: get_detector_path が実際のバージョンを返すようにすべき
        #         duration=time.monotonic() - start_time,
        #         # file_path も get_detector_path から取得できると良い
        #     ).model_dump()
        # )
        return code_content # コード文字列を直接返す

    except (ValueError, FileNotFoundError, ConfigError, FileError) as e:
        error_msg = f"Validation or File Error: {str(e)}"
        error_details = {"error_type": type(e).__name__, "message": str(e)}
        logger.warning(f"[Job {job_id}] Code get failed for '{detector_name}': {error_msg}")
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        error_details = {"error_type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        logger.error(f"[Job {job_id}] Unexpected error getting code for '{detector_name}': {error_msg}", exc_info=True)

    # --- 履歴記録削除 ---
    # fail_data = schemas.CodeLoadFailedData(
    #     job_id=job_id,
    #     detector_name=detector_name,
    #     version=version,
    #     error_message=error_msg,
    #     error_details=error_details,
    #     duration=time.monotonic() - start_time
    # )
    # await add_history_async_func(
    #     session_id=session_id,
    #     event_type="code_load.failed",
    #     event_data=fail_data.model_dump()
    # )
    # エラー時は例外を再発生させる (job_worker で捕捉)
    raise FileError(error_msg or "Failed to get code")

async def _run_save_code_async(
    job_id: str,
    # add_history_async_func: Callable[..., Awaitable[None]], # ★ 削除
    improved_versions_dir: Path,
    allowed_base_dirs: List[Path],
    detector_name: str,
    code: str,
    session_id: Optional[str] = None,
    parent_version: Optional[str] = None,
    changes_summary: Optional[str] = None,
) -> schemas.CodeSaveResultData: # 戻り値はスキーマのまま維持
    """コード保存ジョブ (非同期ラッパー)。履歴記録は削除。"""
    start_time = time.monotonic()
    logger.info(f"[Job {job_id}] Starting save_code for '{detector_name}' (Session: {session_id})")
    file_path_str: Optional[str] = None
    version_tag_final: Optional[str] = None
    commit_hash_final: Optional[str] = None
    error_msg: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # --- 履歴記録削除 ---
    # await add_history_async_func(
    #     session_id=session_id,
    #     event_type="code_save.started",
    #     event_data=schemas.CodeSaveStartedData(
    #         job_id=job_id,
    #         detector_name=detector_name,
    #         session_id=session_id,
    #         parent_version=parent_version
    #     ).model_dump()
    # )

    try:
        loop = asyncio.get_running_loop()
        # partial を使って同期関数に必要な引数を渡す
        sync_task = partial(
            _run_save_code,
            improved_versions_dir=improved_versions_dir,
            allowed_base_dirs=allowed_base_dirs,
            detector_name=detector_name,
            code=code,
            version_tag=None,
            session_id=session_id,
            parent_version=parent_version,
            changes_summary=changes_summary
        )
        # 同期関数を実行
        save_result = await loop.run_in_executor(None, sync_task)

        file_path_str = save_result.get('file_path')
        version_tag_final = save_result.get('version_tag')
        commit_hash_final = save_result.get('commit_hash')

        logger.info(f"[Job {job_id}] Code saved successfully: {file_path_str} (Version: {version_tag_final}, Commit: {commit_hash_final or 'N/A'})")

        result_data = schemas.CodeSaveResultData(
            # message=f"Code for {detector_name} saved successfully.", # message は不要か
            # file_path=file_path_str,
            # version_tag=version_tag_final,
            version=version_tag_final or "unknown", # version スキーマフィールドに合わせる
            file_path=file_path_str or "unknown",
            commit_hash=commit_hash_final
        )

        # --- 履歴記録削除 ---
        # await add_history_async_func(
        #     session_id=session_id,
        #     event_type="code_save.completed",
        #     event_data=schemas.CodeSaveCompleteData(
        #         job_id=job_id,
        #         detector_name=detector_name,
        #         version_tag=version_tag_final,
        #         commit_hash=commit_hash_final,
        #         duration=time.monotonic() - start_time,
        #         file_path=file_path_str
        #     ).model_dump()
        # )
        return result_data # スキーマオブジェクトを返す

    except (ValueError, FileNotFoundError, ConfigError, FileError) as e:
        error_msg = f"Validation or File Error: {str(e)}"
        error_details = {"error_type": type(e).__name__, "message": str(e)}
        logger.warning(f"[Job {job_id}] Error saving code for '{detector_name}': {error_msg}")
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        error_details = {"error_type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        logger.error(f"[Job {job_id}] Unexpected error saving code for '{detector_name}': {error_msg}", exc_info=True)

    # --- 履歴記録削除 ---
    # fail_data = schemas.CodeSaveFailedData(
    #     job_id=job_id,
    #     detector_name=detector_name,
    #     error_message=error_msg,
    #     error_details=error_details,
    #     duration=time.monotonic() - start_time
    # )
    # await add_history_async_func(
    #     session_id=session_id,
    #     event_type="code_save.failed",
    #     event_data=fail_data.model_dump()
    # )
    # エラー時は例外を再発生させる (job_worker で捕捉)
    raise FileError(error_msg or "Failed to save code")

# --- Tool Registration --- #

def register_code_tools(
    mcp: FastMCP,
    config: Dict[str, Any],
    start_async_job_func: Callable[..., Coroutine[Any, Any, Dict[str, str]]], # ★ JobStartResponse形式を返す
    # add_history_async_func: Callable[..., Awaitable[None]] # ★ 削除
):
    """コード管理関連のMCPツールを登録します。"""
    logger.info("コード管理ツールを登録中...")

    # --- 設定からパスを取得 --- #
    try:
        detectors_dir = get_detectors_src_dir(config)
        improved_versions_dir = get_improved_versions_dir(config)
        workspace_dir = get_workspace_dir(config)
        allowed_base_dirs = [detectors_dir, improved_versions_dir, workspace_dir]
        logger.info(f"Detector base dir: {detectors_dir}")
        logger.info(f"Improved versions dir: {improved_versions_dir}")
    except (ConfigError, FileError) as e:
        logger.critical(f"コードツール登録失敗: 設定されたパスが無効です: {e}")
        raise RuntimeError(f"Failed to register code tools due to invalid paths: {e}") from e

    # --- Helper to start job --- #
    async def _start_code_job(
        task_coroutine_factory: Callable[..., Coroutine],
        tool_name: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]: # ★ JobStartResponse 形式
        job_start_info = await start_async_job_func(
            config=config,
            db_path=db_path,
            task_coroutine_func=task_coroutine_factory(
                # Pass necessary args to the factory
                job_id="placeholder", # Will be replaced by actual job_id
                # add_history_async_func=add_history_async_func, # 削除
                **kwargs # Pass other specific args
            ),
            tool_name=tool_name,
            session_id=session_id,
            # Pass original kwargs to be stored in DB/used by worker?
            **kwargs # Pass kwargs like detector_name, version etc.
        )
        # Ensure job_id is updated in the coroutine factory if needed? No, worker gets it.
        return job_start_info

    # --- MCP Tool Definitions --- #

    @mcp.tool("get_code", input_schema=schemas.GetCodeInput)
    async def get_code_tool(
        detector_name: str,
        version: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]: # ★ JobStartResponse 形式
        """指定された検出器のコードを取得するジョブを開始します。"""
        logger.info(f"MCP Tool: get_code called for '{detector_name}' (Version: {version or 'latest'})")
        task_coro_factory = partial(
            _run_get_code_async,
            # add_history_async_func=add_history_async_func, # 削除
            detectors_dir=detectors_dir,
            improved_versions_dir=improved_versions_dir,
            allowed_base_dirs=allowed_base_dirs,
            detector_name=detector_name,
            version=version,
            session_id=session_id
        )
        return await _start_code_job(task_coro_factory, "get_code", session_id, detector_name=detector_name, version=version)

    @mcp.tool("save_code", input_schema=schemas.SaveCodeInput)
    async def save_code_tool(
        detector_name: str,
        code: str,
        session_id: Optional[str] = None,
        parent_version: Optional[str] = None,
        changes_summary: Optional[str] = None,
        # ★ version_tag は入力スキーマから削除し、サーバー側で生成する方針
        # version_tag: Optional[str] = None,
    ) -> Dict[str, Any]: # ★ JobStartResponse 形式
        """提供されたコードを新しいバージョンとして保存するジョブを開始します。"""
        logger.info(f"MCP Tool: save_code called for '{detector_name}' (Session: {session_id})")
        # version_tag は _run_save_code_async / _run_save_code 内で生成されるため、ここでは None を渡す
        task_coro_factory = partial(
            _run_save_code_async,
            # add_history_async_func=add_history_async_func, # 削除
            improved_versions_dir=improved_versions_dir,
            allowed_base_dirs=allowed_base_dirs,
            detector_name=detector_name,
            code=code,
            session_id=session_id,
            parent_version=parent_version,
            changes_summary=changes_summary
        )
        return await _start_code_job(task_coro_factory, "save_code", session_id, detector_name=detector_name)

    logger.info("コード管理ツール登録完了。")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 