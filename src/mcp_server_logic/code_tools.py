import asyncio  # Added for run_in_executor
import json  # Added for json.loads
import logging
import os
import re  # Added for regex matching
import subprocess  # 追加: Gitコマンド実行用
import time
import traceback  # 追加: エラートレースバック取得用
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Tuple

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP

from src.utils.exception_utils import ConfigError, FileError, MirexError
from src.utils.path_utils import get_detectors_src_dir  # Import specific getters
from src.utils.path_utils import (  # get_workspace_dir をインポートに追加; get_detector_path as get_detector_path_core, # Replaced by local helper
    ensure_dir,
    get_improved_versions_dir,
    get_workspace_dir,
    is_safe_path_component,
    validate_path_within_allowed_dirs,
)

from . import schemas  # schemas モジュール自体をインポート
from .schemas import CodeResult  # CodeResultクラスを明示的にインポート

logger = logging.getLogger("mcp_server.code_tools")

# --- Helper Functions --- #


def get_code_version_info(code: str, file_path: Path) -> Dict[str, Any]:
    """コードからバージョン情報を抽出する

    Args:
        code: コード文字列
        file_path: コードファイルのパス

    Returns:
        バージョン情報を含む辞書
    """
    version_info = {
        "filename": file_path.name,
        "path": str(file_path),
        "size_bytes": len(code),
    }

    # ファイル名からバージョンを推測
    filename = file_path.stem
    version_match = re.search(r"_([v\d].+)$", filename)
    if version_match:
        version_info["version_from_filename"] = version_match.group(1)

    # 最終更新日時
    try:
        mtime = file_path.stat().st_mtime
        version_info["last_modified"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(mtime)
        )
    except Exception:
        pass

    # コード内のバージョン情報を検索
    version_pattern = re.compile(
        r'^\s*(VERSION|__version__)\s*=\s*[\'"]([^\'"]+)[\'"]', re.MULTILINE
    )
    version_match = version_pattern.search(code)
    if version_match:
        version_info["version_from_code"] = version_match.group(2)

    # ハッシュ生成
    version_info["version_hash"] = f"v{hash(code) % 100000:05d}"

    return version_info


def get_detector_path(
    detector_name: str,
    config: Dict[str, Any],
    version: Optional[str] = None,
    use_original: bool = False,
) -> Path:
    """検出器のパスを取得する

    Args:
        detector_name: 検出器名
        config: 設定オブジェクト
        version: バージョン情報（オプション）
        use_original: 元のバージョンを使用するかどうか

    Returns:
        検出器ファイルのパス
    """
    try:
        detectors_dir = get_detectors_src_dir(config)
        improved_versions_dir = get_improved_versions_dir(config)

        if not is_safe_path_component(detector_name):
            raise ValueError(f"検出器名が不正です: {detector_name}")

        # 検出器ファイル名を生成
        if detector_name.endswith(".py"):
            filename = detector_name
        else:
            # CamelCase を snake_case に変換
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", detector_name)
            filename = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower() + ".py"

        # バージョン指定がある場合
        if version and not use_original:
            if not is_safe_path_component(version):
                raise ValueError(f"バージョン指定が不正です: {version}")

            base_name = filename[:-3]  # .py を除去
            versioned_filename = f"{base_name}_{version}.py"
            return improved_versions_dir / versioned_filename

        # 通常のパス
        return detectors_dir / filename

    except Exception as e:
        logger.error(f"検出器パスの取得に失敗しました: {e}", exc_info=True)
        raise


# --- Synchronous Task Functions (run in executor) --- #


def _run_get_code(
    # Removed job_id, add_history_sync_func - managed by async wrapper
    detectors_dir: Path,
    improved_versions_dir: Path,
    allowed_base_dirs: List[Path],  # For validation
    detector_name: str,
    version: Optional[str] = None,
    # session_id: Optional[str] = None # Managed by wrapper
) -> str:
    """検出器コード取得ジョブ (同期実処理)"""
    # logger.info(f"[Job {job_id}] Getting code for '{detector_name}' (Version: {version or 'latest'})... Session: {session_id}") # Logged in wrapper

    # Input Validation happens in get_detector_path and wrapper

    try:
        # Use the helper function with passed paths
        detector_path_unvalidated = get_detector_path(
            detectors_dir, improved_versions_dir, detector_name, version
        )

        # Validate the found path is within allowed directories
        detector_path = validate_path_within_allowed_dirs(
            detector_path_unvalidated,
            allowed_base_dirs,
            check_existence=True,
            check_is_file=True,
            allow_absolute=True,  # Paths returned by get_detector_path are absolute
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
        raise FileError(
            f"Unexpected error getting code for '{detector_name}': {e}"
        ) from e


def _run_save_code(
    # Removed job_id, add_history_sync_func - managed by async wrapper
    improved_versions_dir: Path,
    allowed_base_dirs: List[Path],  # For validation
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

    final_version_tag = version_tag  # Keep track of the final tag used

    # Generate version tag if not provided or empty
    if not final_version_tag:
        final_version_tag = time.strftime("%Y%m%d_%H%M%S")  # Use YYYYMMDD_HHMMSS format
        logger.info(
            f"No version tag provided, generated timestamp tag: {final_version_tag}"
        )
        # Re-validate the generated tag
        if not is_safe_path_component(final_version_tag):
            err_msg = f"Generated version tag is invalid: {final_version_tag}"
            logger.critical(f"{err_msg}")  # Critical: should not happen
            raise ValueError(err_msg)

    # Infer base filename from detector name (consistent with get_detector_path)
    if detector_name.endswith(".py"):
        filename_base = detector_name[:-3]
    else:
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", detector_name)
        filename_base = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    filename = f"{filename_base}_{final_version_tag}.py"
    # Target directory is directly improved_versions_dir
    improved_dir = improved_versions_dir
    file_path_unvalidated = improved_dir / filename

    # --- Git Helper Function --- #
    def _run_git_command(
        cmd: List[str], cwd: Path, check: bool = True, ignore_errors: bool = False
    ) -> Tuple[int, str, str]:
        """指定されたディレクトリでGitコマンドを実行する。"""
        try:
            logger.debug(f"Running git command: {' '.join(cmd)} in {cwd}")
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )  # check=False initially
            if check and result.returncode != 0:
                # Raise CalledProcessError manually if check is True and failed
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, output=result.stdout, stderr=result.stderr
                )

            stdout_clean = result.stdout.strip() if result.stdout else ""
            stderr_clean = result.stderr.strip() if result.stderr else ""
            logger.debug(f"Git command stdout:\n{stdout_clean}")
            if stderr_clean:
                # Log stderr as warning unless return code was non-zero
                log_level = logging.ERROR if result.returncode != 0 else logging.WARNING
                logger.log(log_level, f"Git command stderr:\n{stderr_clean}")
            return result.returncode, stdout_clean, stderr_clean
        except FileNotFoundError:
            logger.error(
                "Git command not found. Please ensure Git is installed and in the system PATH."
            )
            if ignore_errors:
                return -1, "", "Git not found"
            raise
        except subprocess.CalledProcessError as e:
            # Logged already by the logic above if check=True
            # logger.error(f"Git command failed: {' '.join(cmd)}")
            # logger.error(f"Return code: {e.returncode}")
            # logger.error(f"Stderr: {e.stderr}")
            if ignore_errors:
                return (
                    e.returncode,
                    e.stdout.strip() if e.stdout else "",
                    e.stderr.strip() if e.stderr else "",
                )
            raise
        except Exception as e:
            logger.error(f"Unexpected error running git command: {e}", exc_info=True)
            if ignore_errors:
                return -1, "", f"Unexpected error: {e}"
            raise

    # --- Git Repository Initialization Check --- #
    is_git_repo = False
    if improved_dir.is_dir():  # Only check/init if the base dir exists
        git_dir = improved_dir / ".git"
        is_git_repo = git_dir.is_dir()
        if not is_git_repo:
            logger.info(
                f"Directory {improved_dir} is not a Git repository. Initializing..."
            )
            try:
                # ensure_dir(improved_dir) # Ensure base directory exists - redundant if improved_dir.is_dir() is true
                _run_git_command(["git", "init"], cwd=improved_dir, check=True)
                is_git_repo = True
                logger.info(
                    f"Successfully initialized Git repository in {improved_dir}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize Git repository in {improved_dir}: {e}. Proceeding without Git."
                )
                is_git_repo = False  # Proceed without Git if init fails
    else:
        logger.warning(
            f"Improved versions directory {improved_dir} does not exist. Cannot perform Git operations."
        )

    # --- File Path Validation and Saving --- #
    file_path: Optional[Path] = None
    try:
        # Ensure the improved_versions_dir itself exists (should be handled by getter, but double-check)
        ensure_dir(improved_dir, check_writable=True)

        # --- Final Path Validation --- #
        file_path = validate_path_within_allowed_dirs(
            file_path_unvalidated,
            allowed_base_dirs,
            check_existence=False,  # File should not exist yet
            allow_absolute=True,  # Path is constructed as absolute
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
        logger.error(
            f"Error validating path or saving code to {file_path_unvalidated}: {e}",
            exc_info=True,
        )
        # Re-raise the error to be caught by the async wrapper
        raise FileError(
            f"Failed to validate path or save code to {file_path_unvalidated}: {e}"
        ) from e

    # --- Git Commit --- #
    commit_hash: Optional[str] = None
    if is_git_repo and file_path:
        try:
            # Git Add
            relative_path_str = str(file_path.relative_to(improved_dir))
            _run_git_command(
                ["git", "add", relative_path_str], cwd=improved_dir, check=True
            )

            # Git Commit
            commit_message = (
                f"Save detector '{detector_name}' version '{final_version_tag}'"
            )
            if session_id:
                commit_message += f"\n\nSession: {session_id}"
            if parent_version:
                commit_message += f"\nParent: {parent_version}"
            if changes_summary:
                commit_message += f"\n\nSummary:\n{changes_summary}"

            _run_git_command(
                ["git", "commit", "-m", commit_message], cwd=improved_dir, check=True
            )

            # Get commit hash
            ret_code, hash_out, hash_err = _run_git_command(
                ["git", "rev-parse", "HEAD"], cwd=improved_dir, check=True
            )
            commit_hash = hash_out
            logger.info(
                f"Successfully committed {relative_path_str} with hash: {commit_hash}"
            )

        except Exception as e:
            # Log error but don't fail the whole operation, just skip Git info
            logger.error(
                f"Git operation failed after saving file {file_path}: {e}. Commit hash will be missing.",
                exc_info=True,
            )
            # Fall through, commit_hash remains None

    # Return result including the version tag and commit hash
    result_data = {
        "file_path": (
            str(file_path) if file_path else None
        ),  # file_path might be None if saving failed early
        "version_tag": final_version_tag,
        "commit_hash": commit_hash,  # Can be None if Git failed or skipped
    }
    return result_data


# --- Asynchronous Task Wrapper Functions --- #


async def _run_get_code_async(
    job_id: str,
    config: Dict[str, Any],
    detector_name: str,
    version: Optional[str] = None,
) -> CodeResult:
    """検出器のコードを取得する非同期タスク"""
    try:
        logger.info(
            f"[Job {job_id}] GET_CODE: Starting to fetch code for detector '{detector_name}', version={version}"
        )

        # 検証用: ジョブステータスを明示的にログ
        logger.info(f"[Job {job_id}] GET_CODE: Job status check - this job is running")

        # コード取得開始
        detector_path = get_detector_path(
            detector_name, config, version, use_original=True
        )

        logger.info(f"[Job {job_id}] GET_CODE: Found detector path: {detector_path}")

        if not detector_path.exists():
            error_msg = f"Detector file not found: {detector_path}"
            logger.error(f"[Job {job_id}] GET_CODE: {error_msg}")
            raise FileNotFoundError(error_msg)

        # ファイル読み込み開始
        logger.info(f"[Job {job_id}] GET_CODE: Reading detector file content")
        try:
            with open(detector_path, "r", encoding="utf-8") as f:
                code = f.read()
                logger.info(
                    f"[Job {job_id}] GET_CODE: Successfully read detector file. Content length: {len(code)} chars"
                )
        except Exception as read_err:
            logger.error(
                f"[Job {job_id}] GET_CODE: Error reading detector file: {read_err}",
                exc_info=True,
            )
            raise FileError(f"Failed to read detector file: {read_err}") from read_err

        # バージョン情報取得
        version_info = {}
        try:
            version_info = get_code_version_info(code, detector_path)
            logger.info(f"[Job {job_id}] GET_CODE: Got version info: {version_info}")
        except Exception as ver_err:
            logger.warning(
                f"[Job {job_id}] GET_CODE: Failed to get code version info: {ver_err}"
            )
            # バージョン情報取得失敗はエラーにせず継続

        # 結果の構築
        result = CodeResult(
            detector_name=detector_name,
            code=code,
            version=version_info.get("version_hash", version),
            file_path=str(detector_path),
            language="python",
            version_info=version_info,
        )

        logger.info(
            f"[Job {job_id}] GET_CODE: Successfully completed for '{detector_name}'"
        )
        return result

    except FileNotFoundError as e:
        logger.error(f"[Job {job_id}] GET_CODE: File not found: {e}", exc_info=True)
        raise  # 再送出
    except FileError as e:
        logger.error(f"[Job {job_id}] GET_CODE: File error: {e}", exc_info=True)
        raise  # 再送出
    except Exception as e:
        logger.error(f"[Job {job_id}] GET_CODE: Unexpected error: {e}", exc_info=True)
        raise FileError(f"Failed to retrieve code: {e}") from e


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
) -> schemas.CodeSaveResultData:  # 戻り値はスキーマのまま維持
    """コード保存ジョブ (非同期ラッパー)。履歴記録は削除。"""
    start_time = time.monotonic()
    logger.info(
        f"[Job {job_id}] Starting save_code for '{detector_name}' (Session: {session_id})"
    )
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
            changes_summary=changes_summary,
        )
        # 同期関数を実行
        save_result = await loop.run_in_executor(None, sync_task)

        file_path_str = save_result.get("file_path")
        version_tag_final = save_result.get("version_tag")
        commit_hash_final = save_result.get("commit_hash")

        logger.info(
            f"[Job {job_id}] Code saved successfully: {file_path_str} (Version: {version_tag_final}, Commit: {commit_hash_final or 'N/A'})"
        )

        result_data = schemas.CodeSaveResultData(
            # message=f"Code for {detector_name} saved successfully.", # message は不要か
            # file_path=file_path_str,
            # version_tag=version_tag_final,
            version=version_tag_final
            or "unknown",  # version スキーマフィールドに合わせる
            file_path=file_path_str or "unknown",
            commit_hash=commit_hash_final,
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
        return result_data  # スキーマオブジェクトを返す

    except (ValueError, FileNotFoundError, ConfigError, FileError) as e:
        error_msg = f"Validation or File Error: {str(e)}"
        error_details = {"error_type": type(e).__name__, "message": str(e)}
        logger.warning(
            f"[Job {job_id}] Error saving code for '{detector_name}': {error_msg}"
        )
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        error_details = {
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        logger.error(
            f"[Job {job_id}] Unexpected error saving code for '{detector_name}': {error_msg}",
            exc_info=True,
        )

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
    start_async_job_func: Callable[
        ..., Coroutine[Any, Any, Dict[str, str]]
    ],  # ★ JobStartResponse形式を返す
    # add_history_async_func: Callable[..., Awaitable[None]] # ★ 削除
    db_path: Path,  # DB_path引数を追加
):
    """コード管理関連のMCPツールを登録します。"""
    logger.info("コード管理ツールを登録中...")

    # --- 設定からパスを取得 --- #
    detectors_dir = None
    improved_versions_dir = None
    workspace_dir = None
    try:
        logger.debug("Attempting to get detectors source directory...")
        detectors_dir = get_detectors_src_dir(config)
        logger.info(f"Detector base dir obtained: {detectors_dir}")

        # ★ 修正点: improved_versions の設定を安全に取得
        logger.debug("Safely getting improved_versions directory path config...")
        paths_config = config.get("paths", {})
        improved_versions_path_config = paths_config.get("improved_versions")

        # もし設定値が絶対パスのように見える場合、または安全でない場合はデフォルト名を使う
        safe_subdir_name = "improved_versions"  # デフォルト
        if isinstance(improved_versions_path_config, str) and is_safe_path_component(
            improved_versions_path_config
        ):
            safe_subdir_name = improved_versions_path_config
        elif improved_versions_path_config is not None:
            logger.warning(
                f"Config value for paths.improved_versions ('{improved_versions_path_config}') seems unsafe or absolute. Using default name '{safe_subdir_name}'."
            )

        # get_improved_versions_dir に渡すための一時的な config を作成 (元の config は変更しない)
        temp_config_for_improved = config.copy()
        if "paths" not in temp_config_for_improved:
            temp_config_for_improved["paths"] = {}
        temp_config_for_improved["paths"][
            "improved_versions"
        ] = safe_subdir_name  # 安全な名前を設定

        logger.debug(
            f"Attempting to get improved versions directory using safe name '{safe_subdir_name}'..."
        )
        improved_versions_dir = get_improved_versions_dir(
            temp_config_for_improved
        )  # 修正したconfigで呼び出す
        logger.info(f"Improved versions dir obtained: {improved_versions_dir}")

        logger.debug("Attempting to get workspace directory...")
        workspace_dir = get_workspace_dir(config)  # これは元の config を使う
        logger.info(f"Workspace dir obtained: {workspace_dir}")

        allowed_base_dirs = [detectors_dir, improved_versions_dir, workspace_dir]
        logger.debug(f"Allowed base directories: {allowed_base_dirs}")

        # improved_versions ディレクトリの存在と書き込み権限をここで再確認
        logger.info(
            f"Explicitly ensuring improved_versions directory: {improved_versions_dir}"
        )
        ensure_dir(improved_versions_dir, check_writable=True)
        logger.info(
            f"Successfully ensured improved_versions directory: {improved_versions_dir}"
        )

    except (ConfigError, FileError) as e:
        logger.critical(f"コードツール登録失敗: 設定されたパスが無効です。Error: {e}")
        logger.critical(
            f"Paths at time of error: detectors_dir={detectors_dir}, improved_versions_dir={improved_versions_dir}, workspace_dir={workspace_dir}"
        )
        logger.critical(traceback.format_exc())  # スタックトレースをログに出力
        raise RuntimeError(
            f"Failed to register code tools due to invalid paths: {e}"
        ) from e
    except Exception as e:
        logger.critical(f"コードツール登録中に予期せぬエラーが発生しました: {e}")
        logger.critical(traceback.format_exc())
        raise RuntimeError(
            f"Unexpected error during code tool registration: {e}"
        ) from e

    # --- Helper to start job --- #
    async def _start_code_job(
        task_coroutine_factory: Callable[..., Coroutine],
        tool_name: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:  # ★ JobStartResponse 形式
        # ファクトリを実行して実際のコルーチンを取得
        actual_task_coroutine = task_coroutine_factory(
            job_id="placeholder",  # Will be replaced by actual job_id in job_manager
            # add_history_async_func=add_history_async_func, # 削除
            **kwargs,  # Pass other specific args like detector_name, version
        )

        # mcp_server.py の lambda に合わせて呼び出し
        # task_coroutine_func は最初の位置引数として渡す
        # config と db_path は lambda が内部でグローバル変数から取得するため不要
        job_start_info = await start_async_job_func(
            actual_task_coroutine,  # コルーチンを最初の位置引数として渡す
            tool_name,
            session_id,  # キーワード引数ではなく位置引数として渡す
            # **kwargs は lambda 経由で job_manager.start_async_job に渡される
            **kwargs,  # Pass kwargs like detector_name, version etc.
        )
        return job_start_info

    # --- MCP Tool Definitions --- #

    @mcp.tool("get_code")
    async def get_code_tool(
        detector_name: str,
        version: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:  # ★ JobStartResponse 形式
        """指定された検出器のコードを取得するジョブを開始します。"""
        logger.info(
            f"MCP Tool: get_code called for '{detector_name}' (Version: {version or 'latest'})"
        )
        task_coro_factory = partial(
            _run_get_code_async,
            # add_history_async_func=add_history_async_func, # 削除
            config=config,
            detector_name=detector_name,
            version=version,
            session_id=session_id,
        )
        return await _start_code_job(
            task_coro_factory,
            "get_code",
            session_id,
            detector_name=detector_name,
            version=version,
        )

    @mcp.tool("save_code")
    async def save_code_tool(
        detector_name: str,
        code: str,
        session_id: Optional[str] = None,
        parent_version: Optional[str] = None,
        changes_summary: Optional[str] = None,
        # ★ version_tag は入力スキーマから削除し、サーバー側で生成する方針
        # version_tag: Optional[str] = None,
    ) -> Dict[str, Any]:  # ★ JobStartResponse 形式
        """提供されたコードを新しいバージョンとして保存するジョブを開始します。"""
        logger.info(
            f"MCP Tool: save_code called for '{detector_name}' (Session: {session_id})"
        )
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
            changes_summary=changes_summary,
        )
        return await _start_code_job(
            task_coro_factory, "save_code", session_id, detector_name=detector_name
        )

    # --- 利用可能な検出器を取得するヘルパー関数 --- #
    def _get_available_detectors(
        search_dirs: List[Path],
    ) -> Dict[str, List[Dict[str, str]]]:
        """指定されたディレクトリから利用可能な検出器の一覧を取得します。"""
        base_detectors = []
        improved_detectors = []

        # 基本検出器ディレクトリを検索
        for dir_path in search_dirs:
            if not dir_path.exists() or not dir_path.is_dir():
                logger.warning(f"検出器ディレクトリが存在しません: {dir_path}")
                continue

            # Pythonファイルを探す（__init__.pyを除く）
            python_files = [f for f in dir_path.glob("*.py") if f.name != "__init__.py"]

            for file_path in python_files:
                # 基本ディレクトリの場合
                if dir_path == detectors_dir:
                    detector_name = file_path.stem  # 拡張子なしのファイル名
                    base_detectors.append(
                        {"name": detector_name, "file_path": str(file_path)}
                    )
                # 改良バージョンディレクトリの場合
                elif dir_path == improved_versions_dir:
                    # 名前とバージョンを分離（例: base_detector_v1.py -> base_detector, v1）
                    file_stem = file_path.stem
                    # バージョンがある場合（_区切り）
                    match = re.match(r"(.+)_(.+)$", file_stem)
                    if match:
                        detector_name = match.group(1)
                        version_tag = match.group(2)
                        improved_detectors.append(
                            {
                                "name": detector_name,
                                "version": version_tag,
                                "file_path": str(file_path),
                            }
                        )

        return {
            "base_detectors": base_detectors,
            "improved_detectors": improved_detectors,
        }

    # --- 検出器一覧を取得するツール --- #
    @mcp.tool("list_detectors")
    async def list_detectors_tool(include_improved: bool = True) -> Dict[str, Any]:
        """利用可能な検出器の一覧を取得します。

        Args:
            include_improved: 改良バージョンも含める場合はTrue

        Returns:
            利用可能な検出器のリスト
        """
        logger.info(
            f"MCP Tool: list_detectors called (include_improved={include_improved})"
        )

        search_dirs = [detectors_dir]
        if include_improved:
            search_dirs.append(improved_versions_dir)

        try:
            detectors_list = _get_available_detectors(search_dirs)
            return {"status": "success", "detectors": detectors_list}
        except Exception as e:
            logger.error(f"検出器一覧の取得に失敗しました: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"検出器一覧の取得に失敗しました: {e}",
            }

    logger.info("コード管理ツール登録完了。")


# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.")
