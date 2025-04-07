#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import uuid
import shutil
import subprocess
import tempfile
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
import logging
import requests
import time
import random
import threading
import sys
import asyncio
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Image, Context

# --- Helper Functions ---
def generate_id():
    return str(uuid.uuid4())

def get_timestamp():
    return time.time()

# ワークスペースディレクトリの存在確認・作成関数 (CONFIG定義前に移動)
def ensure_workspace_directories(config):
    """
    ワークスペースディレクトリが存在することを確認し、必要に応じて作成します。
    アクセス権の問題を処理し、成功または失敗を報告します。
    
    Args:
        config: 設定辞書
    
    Returns:
        bool: すべてのディレクトリが正常に作成または確認された場合はTrue
    """
    # ルートディレクトリを検証
    if "writable_paths" not in config or not isinstance(config["writable_paths"], dict):
        logger.error("設定にwritable_pathsエントリがないか、無効な形式です")
        # エラーを返さずに続行（CONFIGに別のパスが定義されている場合）
        # 設定内のすべてのディレクトリパスを確認
        workspace_dirs = [
            config.get('workspace_dir'),
            config.get('detectors_dir'),
            config.get('evaluation_results_dir'),
            config.get('grid_search_results'), # 追加
            config.get('algorithm_versions_dir'),
            config.get('audio_dir'),
            config.get('reference_dir'),
            config.get('visualizations_dir'), # 追加
            config.get('scientific_output_dir') # 追加
        ]
        workspace_dirs = [d for d in workspace_dirs if d]
    else:
        # ワークスペース関連ディレクトリ
        workspace_dirs = []
        
        # writable_pathsからディレクトリを収集
        for path_key, path_val in config["writable_paths"].items():
            if not path_val:
                logger.warning(f"不正なパス設定: {path_key}")
                continue
                
            try:
                path_str = str(path_val)
                workspace_dirs.append(path_str)
            except Exception as e:
                logger.error(f"パス変換エラー ({path_key}): {e}")
    
    # 作成ディレクトリとエラーを追跡
    created_dirs = []
    error_dirs = []
    
    # 必要なすべてのディレクトリを作成
    for dir_path in workspace_dirs:
        try:
            p = Path(dir_path) # Pathオブジェクトに変換
            if not p.exists():
                logger.info(f"ディレクトリを作成します: {p}")
                # 再帰的に作成（必要な親ディレクトリも一緒に）
                p.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(p))
            
            # 書き込みアクセス権をチェック
            if not os.access(str(p), os.W_OK):
                logger.error(f"ディレクトリへの書き込み権限がありません: {p}")
                error_dirs.append(str(p))
                
        except PermissionError:
            logger.error(f"権限エラー: ディレクトリ {dir_path} を作成できません")
            error_dirs.append(dir_path)
        except FileExistsError:
            # すでにファイルが存在する場合（ディレクトリでない）
            logger.error(f"パス {dir_path} はファイルとして存在し、ディレクトリとして作成できません")
            error_dirs.append(dir_path)
        except Exception as e:
            logger.error(f"ディレクトリ {dir_path} の作成中にエラーが発生しました: {e}")
            error_dirs.append(dir_path)
    
    # 結果をログに記録
    if created_dirs:
        logger.info(f"作成されたディレクトリ: {len(created_dirs)}")
    
    if error_dirs:
        logger.error(f"作成/確認できなかったディレクトリ: {len(error_dirs)}")
        return False
    
    logger.info("すべてのワークスペースディレクトリが確認/作成されました。")
    return True

# 拡張機能をインポート
try:
    import mcp_server_extensions
    has_extensions = True
    logger = logging.getLogger('mcp_server')
    logger.info("MCP拡張機能をインポートしました")
except ImportError:
    has_extensions = False
    logger = logging.getLogger('mcp_server')
    logger.warning("MCP拡張機能がインポートできませんでした")

# PILはオプショナル依存
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# プロジェクトルートの特定
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR  # このスクリプトがプロジェクトルートにある前提

# 環境変数から作業ディレクトリを取得、または一時ディレクトリを使用
workspace_env = os.environ.get("MIREX_WORKSPACE")
workspace_base = None
if workspace_env:
    try:
        p = Path(workspace_env)
        p.mkdir(exist_ok=True, parents=True)
        # 書き込みテスト
        test_file = p / f".write_test_{generate_id()}"
        test_file.touch()
        test_file.unlink()
        workspace_base = p
        logging.info(f"環境変数 MIREX_WORKSPACE から作業ディレクトリを使用: {workspace_base}")
    except (PermissionError, OSError) as e:
        logging.warning(f"環境変数 MIREX_WORKSPACE ({workspace_env}) への書き込み不可 ({e})。フォールバックします。")
        workspace_base = None # フォールバックを強制

if not workspace_base:
    # ホームディレクトリ配下の.mirex_workspaceを試みる
    home_workspace = Path.home() / ".mirex_workspace"
    try:
        home_workspace.mkdir(exist_ok=True, parents=True)
        # 書き込みテスト
        test_file = home_workspace / f".write_test_{generate_id()}"
        test_file.touch()
        test_file.unlink()
        workspace_base = home_workspace
        logging.info(f"ホームディレクトリの .mirex_workspace を使用: {workspace_base}")
    except (PermissionError, OSError) as e:
        logging.warning(f"ホームディレクトリ ({home_workspace}) への書き込み不可 ({e})。一時ディレクトリを使用します。")
        # 一時ディレクトリを使用
        workspace_base = Path(tempfile.gettempdir()) / f"mirex_workspace_{generate_id()[:8]}"
        try:
            workspace_base.mkdir(exist_ok=True, parents=True)
            # 書き込みテスト (一時ディレクトリでも念のため)
            test_file = workspace_base / f".write_test_{generate_id()}"
            test_file.touch()
            test_file.unlink()
            logging.info(f"一時作業ディレクトリを使用: {workspace_base}")
        except (PermissionError, OSError) as e_temp:
             logging.error(f"一時ディレクトリ ({workspace_base}) への書き込みも失敗しました: {e_temp}")
             print(f"エラー: 書き込み可能な作業ディレクトリを確保できませんでした。", file=sys.stderr)
             sys.exit(1)


# 作業ディレクトリ確定
workspace_dir = workspace_base
logging.info(f"最終的な作業ディレクトリ: {workspace_dir}")

# サブディレクトリの設定 (確定した workspace_dir を使用)
WORKSPACE_DIRS = {
    "detectors": workspace_dir / "src" / "detectors",
    "improved_versions": workspace_dir / "src" / "detectors" / "improved_versions",
    "evaluation_results": workspace_dir / "evaluation_results",
    "grid_search_results": workspace_dir / "grid_search_results",
    "data": workspace_dir / "data",
    "visualizations": workspace_dir / "visualizations", # 拡張機能用
    "scientific_output": workspace_dir / "scientific_output", # 拡張機能用
}

# 必要なディレクトリを確保
for dir_path in WORKSPACE_DIRS.values():
    try:
        dir_path.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        logging.error(f"サブディレクトリの作成に失敗しました: {dir_path} - {e}")
        # ここでエラーが発生する場合、上位の workspace_dir の問題の可能性が高い
        print(f"エラー: サブディレクトリ {dir_path} を作成できませんでした。権限を確認してください。", file=sys.stderr)
        sys.exit(1)

# PYTHONPATHにプロジェクトルートを追加（サブプロセス用）
PYTHON_PATH_ENV = {
    "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
}

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('mcp_server')

# MCPサーバー初期化
mcp = FastMCP("MIREX Algorithm Improver", dependencies=["pandas", "numpy", "requests", "pillow"])

# 設定 - 書き込み可能なディレクトリを使用
# workspace_dir は上で確定済み
CONFIG = {
    'workspace_dir': workspace_dir,
    'detectors_dir': WORKSPACE_DIRS['detectors'],
    'evaluation_results_dir': WORKSPACE_DIRS['evaluation_results'],
    'grid_search_results': WORKSPACE_DIRS['grid_search_results'], # grid_search 用ディレクトリ追加
    'algorithm_versions_dir': WORKSPACE_DIRS['improved_versions'],
    'audio_dir': WORKSPACE_DIRS['data'] / 'synthesized' / 'audio',
    'reference_dir': WORKSPACE_DIRS['data'] / 'synthesized' / 'labels',
    'visualizations_dir': WORKSPACE_DIRS['visualizations'], # 拡張機能用
    'scientific_output_dir': WORKSPACE_DIRS['scientific_output'], # 拡張機能用
    'writable_paths': WORKSPACE_DIRS, # writable_paths エントリを更新
}

# 初期化時にディレクトリを確認・作成
if not ensure_workspace_directories(CONFIG):
     logger.critical("必要なワークスペースディレクトリを確保できませんでした。終了します。")
     print("エラー: 必要なワークスペースディレクトリを確保できませんでした。ログを確認してください。", file=sys.stderr)
     sys.exit(1)

# セッション管理
sessions = {}
jobs = {}
# スレッドセーフ化のためのロック
sessions_lock = threading.RLock()
jobs_lock = threading.RLock()

# 最大セッション保持数
MAX_SESSIONS = 10
# 最大ジョブ保持数
MAX_JOBS = 50
# セッションタイムアウト（秒）
SESSION_TIMEOUT = 3600  # 1時間
# ジョブタイムアウト（秒）
JOB_TIMEOUT = 600  # 10分

def cleanup_old_sessions():
    """古いセッションをクリーンアップ"""
    current_time = time.time()
    sessions_to_remove = []
    
    with sessions_lock:
        for session_id, session in sessions.items():
            # タイムアウトまたはセッション数が上限を超えた場合
            if current_time - session.get("last_updated", 0) > SESSION_TIMEOUT:
                sessions_to_remove.append(session_id)
        
        # 古いものから削除
        for session_id in sessions_to_remove:
            logger.info(f"古いセッションを削除: {session_id}")
            del sessions[session_id]
        
        # セッション数が上限を超えた場合、最も古いものを削除
        if len(sessions) > MAX_SESSIONS:
            oldest_session_id = min(sessions.items(), key=lambda x: x[1].get("last_updated", 0))[0]
            logger.info(f"セッション数上限到達のため削除: {oldest_session_id}")
            del sessions[oldest_session_id]

def cleanup_old_jobs():
    """古いジョブをクリーンアップ"""
    current_time = time.time()
    jobs_to_remove = []
    
    with jobs_lock:
        for job_id, job in jobs.items():
            # タイムアウトまたはジョブ数が上限を超えた場合
            if current_time - job.get("created_at", 0) > JOB_TIMEOUT:
                jobs_to_remove.append(job_id)
        
        # 古いものから削除
        for job_id in jobs_to_remove:
            logger.info(f"古いジョブを削除: {job_id}")
            del jobs[job_id]
        
        # ジョブ数が上限を超えた場合、最も古いものを削除
        if len(jobs) > MAX_JOBS:
            oldest_job_id = min(jobs.items(), key=lambda x: x[1].get("created_at", 0))[0]
            logger.info(f"ジョブ数上限到達のため削除: {oldest_job_id}")
            del jobs[oldest_job_id]

# 定期的なクリーンアップを実行するスレッドの開始
def start_cleanup_thread():
    """定期的にクリーンアップを行うスレッドを開始"""
    def cleanup_thread():
        while True:
            try:
                cleanup_old_sessions()
                cleanup_old_jobs()
            except Exception as e:
                logger.error(f"クリーンアップ中にエラーが発生: {e}")
            
            # 10分ごとにクリーンアップ
            time.sleep(600)
    
    thread = threading.Thread(target=cleanup_thread, daemon=True)
    thread.start()
    logger.info("クリーンアップスレッドを開始しました")

# エラーハンドリング関数
def handle_api_error(func):
    """API関数のエラーを適切に処理するデコレータ"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"API呼び出し中にエラーが発生: {func.__name__}: {str(e)}", exc_info=True)
            # jsonify を削除し、通常の辞書を返す
            return {
                "status": "error",
                "error": str(e),
                "function": func.__name__
            }
    
    return wrapper

# --- LLM Configuration ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-3-opus-20240229"

# --- Asynchronous Task Execution ---
def _run_job_async(job_id: str, task_function, *args, **kwargs):
    """Runs a task in a separate thread and updates job status."""
    try:
        logger.info(f"Starting job {job_id} (Task: {task_function.__name__})")
        # Ensure job exists before updating status
        with jobs_lock:
            if job_id not in jobs:
                logger.error(f"Job {job_id} not found in registry before starting.")
                return # Or handle appropriately
            jobs[job_id]['status'] = 'running'
        
        result = task_function(*args, **kwargs)
        
        # Ensure job exists before updating result
        with jobs_lock:
            if job_id not in jobs:
                logger.error(f"Job {job_id} disappeared during execution.")
                return
            jobs[job_id]['result'] = result
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['completed_at'] = get_timestamp()
            
        logger.info(f"Job {job_id} completed successfully.")
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        # Ensure job exists before updating error status
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = str(e)
                jobs[job_id]['completed_at'] = get_timestamp()
            else:
                logger.error(f"Job {job_id} disappeared after failure.")


def start_async_job(task_function, *args, **kwargs) -> str:
    """Starts an asynchronous job and returns its ID."""
    job_id = generate_id()
    
    with jobs_lock:
        jobs[job_id] = {
            'job_id': job_id, # Add job_id to the job data itself
            'status': 'pending',
            'result': None,
            'error': None,
            'task_type': task_function.__name__,
            'created_at': get_timestamp(),
            'completed_at': None
        }
    
    thread = threading.Thread(target=_run_job_async, args=(job_id, task_function, *args), kwargs=kwargs)
    thread.daemon = True # Allow main thread to exit even if background threads are running
    thread.start()
    logger.info(f"Job {job_id} queued (Task: {task_function.__name__}).")
    return job_id

# --- LLM Call Logic ---
def call_llm_api(prompt: str, request_json: bool = False) -> Any:
    """Calls the LLM API (e.g., Anthropic Claude API)."""
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set. Using mock LLM response.")
        return mock_llm_response(prompt, request_json)

    try:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }

        # Add instruction for JSON output if requested
        system_prompt_json = "Please provide the output strictly in valid JSON format."
        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": 4096, # Increased max tokens
            "messages": messages,
        }
        # Add system prompt only if requesting JSON, as it might affect code generation otherwise
        if request_json:
            payload["system"] = system_prompt_json


        logger.debug(f"Sending prompt to LLM (request_json={request_json}): {prompt[:200]}...") # Log truncated prompt
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=300) # Add timeout

        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        llm_response_text = data["content"][0]["text"]
        logger.debug(f"Received LLM response: {llm_response_text[:200]}...") # Log truncated response

        if request_json:
            try:
                # Attempt to parse the response as JSON
                # Handle potential markdown code blocks around JSON
                if llm_response_text.strip().startswith("```json"):
                     llm_response_text = llm_response_text.strip()[7:-3].strip()
                elif llm_response_text.strip().startswith("```"):
                     llm_response_text = llm_response_text.strip()[3:-3].strip()

                json_response = json.loads(llm_response_text)
                # Basic validation: check if 'improved_code' exists if that's expected
                if "improved_code" not in json_response:
                     logger.warning("LLM JSON response lacks 'improved_code' field.")
                     # Return dict anyway, caller needs to handle missing keys
                return json_response
            except json.JSONDecodeError as json_err:
                logger.warning(f"LLM did not return valid JSON despite request. Error: {json_err}. Raw text: '{llm_response_text[:100]}...'")
                # Fallback: Try to extract code block from text if JSON parsing fails
                code = extract_code_from_text(llm_response_text)
                if code:
                    # Return a dictionary mimicking the expected structure
                    return {"improved_code": code, "summary": "Raw text response (JSON parse failed).", "changes": []}
                else:
                    # If no code found either, return an error structure
                    return {"error": "JSON parse failed and no code block found", "raw_response": llm_response_text}

        else: # Raw text response expected
            # Try to extract python code block if present
             code = extract_code_from_text(llm_response_text)
             return code if code else llm_response_text # Return extracted code or full text

    except requests.Timeout:
        logger.error("LLM API call timed out.")
        return {"error": "LLM API call timed out."} # Return error structure
    except requests.exceptions.RequestException as req_err:
         logger.error(f"LLM API request error: {req_err}", exc_info=True)
         return {"error": f"LLM API request error: {req_err}"} # Return error structure
    except Exception as e:
        logger.error(f"LLM API call error: {e}", exc_info=True)
        return {"error": f"Unexpected LLM API error: {e}"} # Return error structure

def extract_code_from_text(text: str) -> Optional[str]:
     """Extracts Python code block from text, handling potential markdown."""
     # Prioritize ```python blocks
     start_tag = "```python"
     end_tag = "```"
     start_index = text.find(start_tag)
     if start_index != -1:
          start_index += len(start_tag)
          end_index = text.find(end_tag, start_index)
          if end_index != -1:
               return text[start_index:end_index].strip()

     # Fallback: check for triple backticks without language specifier
     start_tag_plain = "```"
     start_index = text.find(start_tag_plain)
     if start_index != -1:
         # Ensure it's not the start of a json block
         if text[start_index:start_index+7].lower() == "```json":
              return None # Don't extract json as python code
         start_index += len(start_tag_plain)
         end_index = text.find(end_tag, start_index)
         if end_index != -1:
             # Basic check if it looks like python code (crude)
             code_candidate = text[start_index:end_index].strip()
             if "def " in code_candidate or "class " in code_candidate or "import " in code_candidate or "print(" in code_candidate:
                 return code_candidate

     # If no blocks found, assume the whole text might be code if it looks like it
     text_stripped = text.strip()
     if ("def " in text_stripped or "class " in text_stripped or "import " in text_stripped) and not text_stripped.startswith("{"):
         return text_stripped

     logger.debug("Could not extract Python code block from text.")
     return None # Return None if no code block found


def mock_llm_response(prompt: str, request_json: bool = False) -> Any:
    """Mock LLM response implementation."""
    logger.info("Generating mock LLM response.")
    time.sleep(random.uniform(0.5, 1.5)) # Simulate processing time
    code = extract_code_from_text(prompt) # Try to get original code from prompt
    if not code:
        code = "# Mock improvement: Original code not found in prompt\nprint('Hello from mock improved code!')"

    # Simple modification for mock
    issues_str = "General performance and specific edge cases" # Default mock issue
    if "リバーブ" in prompt: issues_str = "Reverb handling"
    elif "ノイズ" in prompt: issues_str = "Noise robustness"
    elif "和音" in prompt: issues_str = "Polyphony detection"

    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    modified_code = f"""# Mock LLM Improvement ({current_time})
# Target: {issues_str}
# - Adjusted sensitivity parameters slightly.
# - Added a placeholder for error handling.

{code}

# Mock modification: Adding a dummy function
def _mock_helper_function_{int(time.time())}():
    # Placeholder for future logic
    pass

# Mock modification: Basic error handling placeholder
try:
    # Original logic might be here
    pass
except Exception as e:
    print(f"Mock error handler caught: {{e}}")

"""
    # Modify a parameter slightly if possible
    threshold_line_found = False
    lines = modified_code.splitlines()
    for idx, line in enumerate(lines):
        if "DEFAULT_F0_SCORE_THRESHOLD" in line and "=" in line:
            try:
                parts = line.split('=')
                current_val_str = parts[1].split('#')[0].strip()
                current_val = float(current_val_str)
                new_val = max(0.05, current_val * 0.95) # Decrease by 5%
                lines[idx] = f"DEFAULT_F0_SCORE_THRESHOLD = {new_val:.3f} # Mock adjustment"
                threshold_line_found = True
                break # Modify only the first occurrence
            except ValueError:
                pass # Ignore if value is not float

    modified_code = "\n".join(lines)


    if request_json:
        return {
            "improved_code": modified_code,
            "summary": f"Mock improvement targeting {issues_str}.",
            "changes": ["Adjusted sensitivity parameters", "Added basic error handling placeholder", "Added dummy helper function."]
        }
    else:
        # For non-json requests, just return the code string
        return modified_code


# --- Evaluation and Grid Search Execution Logic ---
# Note: These functions now run synchronously within the async job thread
def _execute_evaluation(session_id: str, detector_names: List[str], audio_dir: str, ref_dir: str, ref_pattern: str) -> Dict[str, Any]:
    """Executes the evaluation script (synchronous part of the async job)."""
    # Ensure detector names are strings
    safe_detector_names = [str(d) for d in detector_names]
    # 出力ディレクトリを CONFIG から取得
    eval_results_base_dir = CONFIG['evaluation_results_dir']
    output_dir = eval_results_base_dir / f"mcp_session_{session_id}_{generate_id()[:8]}"
    
    try:
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"評価結果出力ディレクトリ: {output_dir}")
    except Exception as e:
        logger.error(f"評価結果出力ディレクトリの作成に失敗: {output_dir} - {e}")
        raise RuntimeError(f"評価出力ディレクトリ作成失敗: {e}") from e

    cmd = [
        sys.executable, # Use the same python interpreter that runs the server
        str(PROJECT_ROOT / "run_evaluate.py"), # スクリプトパスを明確化
        "--audio-dir", audio_dir,
        "--reference-dir", ref_dir,
        "--reference-pattern", ref_pattern,
        "--detectors", ",".join(safe_detector_names),
        "--output-dir", str(output_dir), # Pathオブジェクトを文字列に変換
        "--save-plots", "--save-results-json"
    ]
    logger.info(f"Executing evaluation: {' '.join(cmd)}")
    try:
        # Set PYTHONPATH to include the project root directory if necessary
        env = os.environ.copy()
        # Assuming the server runs from the project root (PROJECT_ROOT を使用)
        env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')

        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600, env=env) # Add timeout and env
        logger.info(f"Evaluation completed. Output (stdout):\n{result.stdout[:500]}...")
        if result.stderr:
             logger.warning(f"Evaluation completed with stderr output:\n{result.stderr[:500]}...")

        result_file = output_dir / "evaluation_results.json" # Path オブジェクトを使用
        if result_file.exists():
            with open(result_file, 'r') as f:
                eval_data = json.load(f)
            # Add output dir path to results for reference
            eval_data["output_directory"] = str(output_dir) # 文字列で保存
            return eval_data
        else:
            # Sometimes output might be empty if no files are processed or detector fails early
            logger.error(f"Evaluation result file not found: {result_file}. Check evaluation logs/stdout above.")
            # Return a minimal structure indicating failure/no results
            return {
                 "error": "Evaluation result file not found.",
                 "output_directory": output_dir,
                 "overall_metrics": {"note": {"f_measure": 0.0, "precision": 0.0, "recall": 0.0}}, # Default metrics
                 "per_file_metrics": {}
            }
    except subprocess.TimeoutExpired:
        logger.error("Evaluation script timed out.")
        raise TimeoutError("Evaluation timed out")
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation script failed with code {e.returncode}. Error:\n{e.stderr}")
        raise RuntimeError(f"Evaluation failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Error during evaluation execution: {e}", exc_info=True)
        raise

def _execute_grid_search(session_id: str, detector_name: str, audio_dir: str, ref_dir: str, base_config: str, grid_params: Dict[str, List[str]]) -> Dict[str, Any]:
    """Executes the grid search script (synchronous part of the async job)."""
    # 出力ディレクトリを CONFIG から取得
    grid_results_base_dir = CONFIG['grid_search_results']
    output_dir_base = grid_results_base_dir / f"mcp_session_{session_id}_{generate_id()[:8]}"
    output_dir_run = output_dir_base / detector_name
    grid_config_dir = output_dir_base # 設定ファイルも同じベースディレクトリに

    try:
        output_dir_run.mkdir(exist_ok=True, parents=True)
        grid_config_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"グリッドサーチ結果出力ディレクトリ: {output_dir_run}")
        logger.info(f"グリッドサーチ設定ファイルディレクトリ: {grid_config_dir}")
    except Exception as e:
        logger.error(f"グリッドサーチディレクトリの作成に失敗: {e}")
        raise RuntimeError(f"グリッドサーチディレクトリ作成失敗: {e}") from e

    grid_config_file = grid_config_dir / f"grid_config_{detector_name}.yaml"

    # Build create-config command
    create_cmd = [
        sys.executable, # Use the same python interpreter
        str(PROJECT_ROOT / "run_grid_search.py"), "create-config", # スクリプトパスを明確化
        "--detector", str(detector_name), # Ensure string
        "--audio-dir", audio_dir,
        "--reference-dir", ref_dir,
        "--output", str(grid_config_file) # Pathを文字列に変換
    ]
    # Ensure grid param values are strings for command line
    for param, values in grid_params.items():
        safe_values = [str(v) for v in values]
        create_cmd.extend(["--param", str(param)] + safe_values) # Ensure param name is string

    logger.info(f"Generating grid config: {' '.join(create_cmd)}")
    try:
        env = os.environ.copy()
        project_root = os.path.dirname(os.path.abspath(__file__))
        env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

        # Generate grid config file
        result_create = subprocess.run(create_cmd, capture_output=True, text=True, check=True, timeout=60, env=env)
        logger.info(f"Grid config generation output:\n{result_create.stdout[:500]}...")
        if result_create.stderr:
             logger.warning(f"Grid config generation stderr:\n{result_create.stderr[:500]}...")
        if not os.path.exists(grid_config_file):
            raise FileNotFoundError(f"Grid config file not generated: {grid_config_file}")

        # Build run command
        run_cmd = [
            sys.executable, # Use the same python interpreter
            str(PROJECT_ROOT / "run_grid_search.py"), "run", # スクリプトパスを明確化
            "--config", base_config, # base_config はファイルパスなのでそのまま
            "--grid-config", str(grid_config_file), # Pathを文字列に変換
            "--output-dir", str(output_dir_run), # Pathを文字列に変換
            "--best-metric", "note.f_measure" # Default metric
        ]
        logger.info(f"Executing grid search: {' '.join(run_cmd)}")
        # Run grid search (potentially long running)
        result_run = subprocess.run(run_cmd, capture_output=True, text=True, check=True, timeout=1800, env=env) # 30 min timeout
        logger.info(f"Grid search completed. Output:\n{result_run.stdout[:500]}...")
        if result_run.stderr:
             logger.warning(f"Grid search stderr:\n{result_run.stderr[:500]}...")

        # Read best parameters
        best_params_file = output_dir_run / "best_params.json" # Path オブジェクトを使用
        best_metrics_file = output_dir_run / "best_metrics.json" # Also get best metrics

        results = {"output_directory": str(output_dir_run)} # 文字列で保存
        if best_params_file.exists():
            try:
                with open(best_params_file, 'r') as f:
                    results["best_params"] = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from {best_params_file}")
                results["best_params"] = None
        else:
            logger.warning(f"Best params file not found: {best_params_file}")
            results["best_params"] = None

        if os.path.exists(best_metrics_file):
             try:
                 with open(best_metrics_file, 'r') as f:
                     results["best_metrics"] = json.load(f)
             except json.JSONDecodeError:
                 logger.error(f"Failed to decode JSON from {best_metrics_file}")
                 results["best_metrics"] = None

        else:
            logger.warning(f"Best metrics file not found: {best_metrics_file}")
            results["best_metrics"] = None

        # If no best params found, still return success but indicate outcome
        if results["best_params"] is None:
            results["message"] = "Grid search completed, but no optimal parameters found or file was invalid."

        return results

    except subprocess.TimeoutExpired:
        logger.error("Grid search script timed out.")
        raise TimeoutError("Grid search timed out")
    except subprocess.CalledProcessError as e:
        logger.error(f"Grid search script failed with code {e.returncode}. Error:\n{e.stderr}")
        raise RuntimeError(f"Grid search failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Error during grid search execution: {e}", exc_info=True)
        raise


# --- MCP Tools ---

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """サーバーの健全性をチェックします。"""
    return {"status": "healthy", "timestamp": get_timestamp(), "version": "1.0.0"}

@mcp.tool()
async def start_session(base_algorithm: str = "Unknown") -> Dict[str, Any]:
    """
    改善セッションを開始します。
    
    Args:
        base_algorithm: 改善対象のアルゴリズム名
        
    Returns:
        Dict[str, Any]: セッション情報
    """
    session_id = generate_id()
    with sessions_lock:
        sessions[session_id] = {
            "id": session_id,
            "created_at": get_timestamp(),
            "updated_at": get_timestamp(),
            "base_algorithm": base_algorithm,
            "status": "active",
            "history": []
        }
    return sessions[session_id]

@mcp.tool()
async def get_session_info(session_id: str) -> Dict[str, Any]:
    """
    セッション情報を取得します。
    
    Args:
        session_id: セッションID
        
    Returns:
        Dict[str, Any]: セッション情報
    """
    with sessions_lock:
        session = sessions.get(session_id)
        if not session:
            raise ValueError(f"セッション {session_id} が見つかりません")
        return session

@mcp.tool()
async def add_session_history(session_id: str, event_type: str, event_data: str) -> Dict[str, Any]:
    """
    セッション履歴にイベントを追加します。
    
    Args:
        session_id: セッションID
        event_type: イベントタイプ
        event_data: イベントデータ（JSON文字列）
        
    Returns:
        Dict[str, Any]: 更新されたセッション情報
    """
    with sessions_lock:
        session = sessions.get(session_id)
        if not session:
            raise ValueError(f"セッション {session_id} が見つかりません")
        
        # イベントデータをパース（JSONの場合）
        try:
            if isinstance(event_data, str):
                data = json.loads(event_data)
            else:
                data = event_data
        except json.JSONDecodeError:
            data = event_data  # JSONでない場合はそのまま使用
            
        # 履歴に追加
        event = {
            "id": generate_id(),
            "timestamp": get_timestamp(),
            "type": event_type,
            "data": data
        }
        session["history"].append(event)
        session["updated_at"] = get_timestamp()
        
        return session

@mcp.tool()
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    非同期ジョブのステータスを取得します。
    
    Args:
        job_id: ジョブID
        
    Returns:
        Dict[str, Any]: ジョブステータス情報
    """
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise ValueError(f"ジョブ {job_id} が見つかりません")
        
        status = job.get("status", "unknown")
        result = job.get("result")
        error = job.get("error")
        
        response = {
            "job_id": job_id,
            "status": status,
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at")
        }
        
        if status == "completed" and result is not None:
            response["result"] = result
            
        if status == "failed" and error is not None:
            response["error"] = error
            
        return response

@mcp.tool()
async def get_code(detector_name: str) -> str:
    """
    検出器のソースコードを取得します。
    
    Args:
        detector_name: 検出器名
        
    Returns:
        str: ソースコード
    """
    from src.utils.path_utils import get_detector_path
    
    # 検出器パスを取得
    detector_path = get_detector_path(detector_name)
    if not detector_path or not detector_path.exists():
        raise FileNotFoundError(f"検出器 {detector_name} が見つかりません")
    
    # コードを読み込み
    try:
        with open(detector_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        return source_code
    except Exception as e:
        logger.error(f"検出器コードの読み込みに失敗しました: {detector_name} - {e}")
        raise IOError(f"検出器コードの読み込みに失敗しました: {e}")

@mcp.tool()
async def save_code(detector_name: str, code: str, version: str = None) -> Dict[str, Any]:
    """
    改善されたコードを保存します。
    
    Args:
        detector_name: 検出器名
        code: 保存するソースコード
        version: バージョン番号（省略時は自動生成）
        
    Returns:
        Dict[str, Any]: 保存結果情報
    """
    if version is None:
        version = f"v{int(time.time())}"
    
    # 改善バージョンのディレクトリ
    improved_dir = WORKSPACE_DIRS["improved_versions"]
    improved_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存ファイル名の生成
    filename = f"{detector_name}_{version}.py"
    file_path = improved_dir / filename
    
    # コードを保存
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        return {
            "detector_name": detector_name,
            "version": version,
            "file_path": str(file_path),
            "timestamp": get_timestamp(),
            "status": "saved"
        }
    except Exception as e:
        logger.error(f"コードの保存に失敗しました: {filename} - {e}")
        raise IOError(f"コードの保存に失敗しました: {e}")

@mcp.tool()
async def improve_code(prompt: str, session_id: str = "") -> Dict[str, Any]:
    """
    LLMを使用してコードを改善します。
    
    Args:
        prompt: LLMに送信するプロンプト
        session_id: セッションID（省略可）
        
    Returns:
        Dict[str, Any]: ジョブ情報
    """
    # 非同期ジョブを開始
    job_id = start_async_job(call_llm_api, prompt, request_json=True)
    
    # セッション履歴に追加（セッションIDが指定されている場合）
    if session_id:
        try:
            with sessions_lock:
                if session_id in sessions:
                    # セッション履歴に追加
                    add_session_history(session_id, "improve_code_request", {"prompt": prompt, "job_id": job_id})
        except Exception as e:
            logger.warning(f"セッション履歴の更新に失敗しました: {e}")
    
    return {"job_id": job_id, "status": "pending"}

@mcp.tool()
async def run_evaluation(detector_name: str, audio_dir: str = None, reference_dir: str = None) -> Dict[str, Any]:
    """
    検出器の評価を実行します。
    
    Args:
        detector_name: 評価する検出器名
        audio_dir: 音声ファイルディレクトリ（省略時はデフォルト）
        reference_dir: 正解ラベルディレクトリ（省略時はデフォルト）
        
    Returns:
        Dict[str, Any]: ジョブ情報
    """
    # デフォルトディレクトリ
    if audio_dir is None:
        audio_dir = str(CONFIG["audio_dir"])
    if reference_dir is None:
        reference_dir = str(CONFIG["reference_dir"])
    
    # 非同期ジョブを開始
    job_id = start_async_job(_execute_evaluation, "", [detector_name], audio_dir, reference_dir, "*.csv")
    
    return {"job_id": job_id, "status": "pending", "detector_name": detector_name}

@mcp.tool()
async def run_grid_search(detector_name: str, grid_params_json: str, 
                         audio_dir: str = None, reference_dir: str = None) -> Dict[str, Any]:
    """
    パラメータグリッドサーチを実行します。
    
    Args:
        detector_name: 検出器名
        grid_params_json: グリッドサーチパラメータ（JSON文字列）
        audio_dir: 音声ファイルディレクトリ（省略時はデフォルト）
        reference_dir: 正解ラベルディレクトリ（省略時はデフォルト）
        
    Returns:
        Dict[str, Any]: ジョブ情報
    """
    # デフォルトディレクトリ
    if audio_dir is None:
        audio_dir = str(CONFIG["audio_dir"])
    if reference_dir is None:
        reference_dir = str(CONFIG["reference_dir"])
    
    try:
        # パラメータをJSONからパース
        grid_params = json.loads(grid_params_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"グリッドサーチパラメータの解析に失敗しました: {e}")
    
    # 非同期ジョブを開始
    job_id = start_async_job(_execute_grid_search, "", detector_name, audio_dir, reference_dir, "config.yaml", grid_params)
    
    return {"job_id": job_id, "status": "pending", "detector_name": detector_name}

@mcp.tool()
async def analyze_code_segment(code_segment: str, question: str = "") -> Dict[str, Any]:
    """
    コードセグメントを分析します。
    
    Args:
        code_segment: 分析するコード
        question: 分析に関する質問（省略可）
        
    Returns:
        Dict[str, Any]: ジョブ情報
    """
    # デフォルトの質問
    if not question:
        question = "このコードの問題点と改善案を提案してください。"
    
    # プロンプトの構築
    prompt = f"""
    以下のコードを分析してください。

    ```python
    {code_segment}
    ```

    質問: {question}
    """
    
    # 非同期ジョブを開始
    job_id = start_async_job(call_llm_api, prompt, request_json=False)
    
    return {"job_id": job_id, "status": "pending"}

@mcp.tool()
async def suggest_parameters(source_code: str, context: str = "") -> Dict[str, Any]:
    """
    アルゴリズムコードからパラメータ調整範囲を提案します。
    
    Args:
        source_code: 検出器のソースコード
        context: コンテキスト情報（省略可）
        
    Returns:
        Dict[str, Any]: ジョブ情報
    """
    # デフォルトのコンテキスト
    if not context:
        context = "このアルゴリズムのパラメータとその適切な調整範囲を提案してください。"
    
    # プロンプトの構築
    prompt = f"""
    以下の音楽情報検出アルゴリズムのコードからパラメータを特定し、チューニングに適した値の範囲を提案してください。

    ```python
    {source_code}
    ```

    コンテキスト: {context}
    
    以下の形式でJSON形式で回答してください:
    {{
        "params": [
            {{
                "name": "パラメータ名",
                "current_value": 現在の値,
                "suggested_range": [最小値, 最大値],
                "step": ステップ幅,
                "description": "パラメータの説明"
            }}
        ],
        "explanation": "提案の根拠や説明"
    }}
    """
    
    # 非同期ジョブを開始
    job_id = start_async_job(call_llm_api, prompt, request_json=True)
    
    return {"job_id": job_id, "status": "pending"}

@mcp.tool()
async def create_thumbnail(image_path: str, size: int = 100) -> Image:
    """画像からサムネイルを作成します。
    
    Args:
        image_path: 元画像のパス
        size: サムネイルのサイズ（ピクセル）
    
    Returns:
        サムネイル画像
    """
    if not HAS_PIL:
        raise ImportError("PILのインストールが必要です: pip install pillow")
    
    # パスが有効か確認
    if not os.path.exists(image_path):
        # ワークスペース内の相対パスを試す
        workspace_path = os.path.join(CONFIG['workspace_dir'], image_path)
        if os.path.exists(workspace_path):
            image_path = workspace_path
        else:
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
    
    # 画像を読み込み、サムネイルを作成
    img = PILImage.open(image_path)
    img.thumbnail((size, size))
    
    # バイト列に変換
    with tempfile.BytesIO() as output:
        img_format = img.format or 'PNG'
        img.save(output, format=img_format)
        image_data = output.getvalue()
    
    # MCPのImageオブジェクトを返す
    return Image(data=image_data, format=img_format.lower())

@mcp.tool()
async def visualize_spectrogram(audio_file: str, output_format: str = "png") -> Image:
    """音声ファイルからスペクトログラムを生成します。
    
    Args:
        audio_file: 音声ファイルのパス
        output_format: 出力画像のフォーマット（png/jpg）
    
    Returns:
        スペクトログラム画像
    """
    if not HAS_PIL:
        raise ImportError("PILのインストールが必要です: pip install pillow")
    
    try:
        import librosa
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("解析に必要なライブラリが不足しています: pip install librosa matplotlib")
    
    # パスが有効か確認
    if not os.path.exists(audio_file):
        # ワークスペース内の相対パスを試す
        workspace_path = os.path.join(CONFIG['workspace_dir'], audio_file)
        if os.path.exists(workspace_path):
            audio_file = workspace_path
        else:
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_file}")
    
    # 音声を読み込み
    y, sr = librosa.load(audio_file)
    
    # スペクトログラムを作成
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.amplitude_to_db(librosa.stft(y), ref=np.max),
        y_axis='log', x_axis='time'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'スペクトログラム: {os.path.basename(audio_file)}')
    
    # 画像をバイト列に変換
    with tempfile.BytesIO() as output:
        plt.savefig(output, format=output_format)
        plt.close()
        image_data = output.getvalue()
    
    # MCPのImageオブジェクトを返す
    return Image(data=image_data, format=output_format)

@mcp.tool()
async def process_audio_files(files: list[str], ctx: Context) -> dict:
    """複数の音声ファイルを処理し、処理の進捗を報告します。
    
    Args:
        files: 処理する音声ファイルのリスト
        ctx: MCP Context オブジェクト
    
    Returns:
        処理結果の概要
    """
    results = []
    
    # ファイルが存在するか確認
    valid_files = []
    for i, file in enumerate(files):
        if not os.path.exists(file):
            # ワークスペース内で探す
            workspace_path = os.path.join(CONFIG['workspace_dir'], file)
            if os.path.exists(workspace_path):
                valid_files.append(workspace_path)
            else:
                ctx.warning(f"ファイルが見つかりません: {file}")
        else:
            valid_files.append(file)
    
    if not valid_files:
        ctx.error("有効なファイルが見つかりませんでした")
        return {"error": "No valid files found", "processed": 0}
    
    # ファイル処理を開始
    for i, file in enumerate(valid_files):
        file_name = os.path.basename(file)
        ctx.info(f"処理中: {file_name} ({i+1}/{len(valid_files)})")
        
        # 進捗を報告
        await ctx.report_progress(i, len(valid_files))
        
        try:
            # ファイルの基本情報を収集
            file_size = os.path.getsize(file)
            file_stats = {
                "name": file_name,
                "size": file_size,
                "path": file,
                "status": "processed"
            }
            
            # リソースを読み込み（例としてファイルパスを使用）
            try:
                resource_path = f"file://{file}"
                data, mime_type = await ctx.read_resource(resource_path)
                file_stats["mime_type"] = mime_type
                file_stats["data_length"] = len(data) if data else 0
            except Exception as e:
                file_stats["resource_error"] = str(e)
            
            results.append(file_stats)
        except Exception as e:
            ctx.error(f"ファイル処理エラー: {file_name} - {str(e)}")
            results.append({
                "name": file_name,
                "path": file,
                "status": "error",
                "error": str(e)
            })
    
    # 最終進捗を報告
    await ctx.report_progress(len(valid_files), len(valid_files))
    ctx.info(f"処理完了: {len(results)} ファイル")
    
    return {
        "processed": len(results),
        "success": len([r for r in results if r.get("status") == "processed"]),
        "errors": len([r for r in results if r.get("status") == "error"]),
        "results": results
    }

def run_subprocess(cmd: List[str], cwd: Optional[Union[str, Path]] = None, 
                  timeout: int = 60, env_vars: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """
    サブプロセスを実行し、終了コード、標準出力、標準エラー出力を返します。
    
    Args:
        cmd: 実行するコマンドとその引数のリスト
        cwd: 作業ディレクトリ（指定しない場合はカレントディレクトリ）
        timeout: タイムアウト（秒）
        env_vars: 追加の環境変数
        
    Returns:
        (終了コード, 標準出力, 標準エラー出力)のタプル
    """
    # 入力検証: コマンドは文字列のリストでなければならない
    if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
        logger.error("コマンドは文字列のリストである必要があります")
        return -1, "", "無効なコマンド形式"
    
    # シェルインジェクションを防止するための検証
    if any(";" in arg or "|" in arg or "&" in arg or ">" in arg or "<" in arg for arg in cmd):
        logger.error("コマンドに潜在的に安全でない文字が含まれています")
        return -1, "", "潜在的に安全でないコマンドが拒否されました"
    
    # タイムアウト値の検証
    try:
        timeout = int(timeout)
        if timeout <= 0:
            timeout = 60  # 無効な場合はデフォルト
    except (ValueError, TypeError):
        timeout = 60  # 無効な場合はデフォルト
    
    logger.debug(f"コマンド実行: {' '.join(cmd)}")
    
    # 環境変数の設定と検証
    env = os.environ.copy()
    env.update(PYTHON_PATH_ENV)  # PYTHONPATHを常に設定
    
    if env_vars:
        safe_env_vars = {}
        for k, v in env_vars.items():
            if isinstance(k, str) and isinstance(v, str):
                safe_env_vars[k] = v
        env.update(safe_env_vars)
    
    # 作業ディレクトリの検証
    if cwd is not None:
        cwd_str = str(cwd)
        if not os.path.isdir(cwd_str):
            logger.warning(f"指定された作業ディレクトリが存在しません: {cwd_str}")
            cwd_str = str(PROJECT_ROOT)
    else:
        cwd_str = str(PROJECT_ROOT)
    
    # 安全にプロセスを実行
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd_str,
            env=env,
            shell=False  # セキュリティのためshell=Falseを明示
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    
    except subprocess.TimeoutExpired:
        process.kill()
        logger.error(f"コマンドが{timeout}秒後にタイムアウト: {' '.join(cmd)}")
        return -1, "", f"{timeout}秒後のタイムアウト"
    
    except Exception as e:
        logger.error(f"コマンド実行エラー: {e}")
        return -1, "", str(e)

def setup_signal_handlers():
    """
    シグナルハンドラを設定して、適切なシャットダウンを確保します。
    """
    import signal
    
    def signal_handler(sig, frame):
        """
        シグナルを受信したときの処理
        """
        logger.info(f"シグナルを受信しました: {sig}")
        # クリーンアップ処理を実行
        try:
            logger.info("クリーンアップを実行しています...")
            # セッションとジョブの情報を保存
            save_sessions_and_jobs()
            logger.info("セッションとジョブの状態が保存されました")
        except Exception as e:
            logger.error(f"クリーンアップ中にエラーが発生しました: {e}")
        finally:
            logger.info("MCPサーバーをシャットダウンします")
            # 0で終了（正常終了）
            sys.exit(0)
    
    # SIGINT（Ctrl+C）とSIGTERM（kill）のハンドラを登録
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("シグナルハンドラが設定されました")

def save_sessions_and_jobs():
    """
    セッションとジョブの情報を永続化します。
    """
    try:
        data_dir = WORKSPACE_DIRS['data']
        os.makedirs(data_dir, exist_ok=True)
        
        # セッション情報を保存
        with sessions_lock:
            with open(data_dir / "sessions.json", "w") as f:
                # 保存可能な形式に変換
                serializable_sessions = {}
                for session_id, session in sessions.items():
                    serializable_sessions[session_id] = {
                        "created_at": session.get("created_at", 0),
                        "last_accessed": session.get("last_accessed", 0)
                    }
                json.dump(serializable_sessions, f)
        
        # ジョブ情報を保存
        with jobs_lock:
            with open(data_dir / "jobs.json", "w") as f:
                # 保存可能な形式に変換
                serializable_jobs = {}
                for job_id, job in jobs.items():
                    serializable_jobs[job_id] = {
                        "status": job.get("status", "unknown"),
                        "created_at": job.get("created_at", 0),
                        "completed_at": job.get("completed_at", 0)
                    }
                json.dump(serializable_jobs, f)
        
        logger.info("セッションとジョブの状態が正常に保存されました")
        return True
    except Exception as e:
        logger.error(f"状態の保存中にエラーが発生しました: {e}")
        return False

def load_sessions_and_jobs():
    """
    保存されたセッションとジョブの情報を読み込みます。
    """
    try:
        data_dir = WORKSPACE_DIRS['data']
        
        # セッション情報を読み込み
        session_file = data_dir / "sessions.json"
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                loaded_sessions = json.load(f)
                with sessions_lock:
                    sessions.clear()
                    for session_id, session_data in loaded_sessions.items():
                        sessions[session_id] = session_data
            logger.info(f"{len(sessions)}個のセッションを読み込みました")
        
        # ジョブ情報を読み込み
        job_file = data_dir / "jobs.json"
        if os.path.exists(job_file):
            with open(job_file, "r") as f:
                loaded_jobs = json.load(f)
                with jobs_lock:
                    jobs.clear()
                    for job_id, job_data in loaded_jobs.items():
                        jobs[job_id] = job_data
            logger.info(f"{len(jobs)}個のジョブを読み込みました")
        
        return True
    except Exception as e:
        logger.error(f"状態の読み込み中にエラーが発生しました: {e}")
        return False

# アプリケーション初期化の削除
# フラスク初期化後のクリーンアップスレッド開始 (これは main ブロックに移動しても良いが、ここでもOK)
start_cleanup_thread()

# Flask連携用の関数呼び出しを削除

# 拡張機能を登録（存在する場合） - このブロックは保持する
if has_extensions:
    logger.info("MCP拡張機能を登録しています...")
    # register_tools に CONFIG を渡す
    extension_tools = mcp_server_extensions.register_tools(mcp, sessions, {"start_job": start_async_job}, CONFIG) 
    logger.info(f"MCP拡張機能を登録しました: {', '.join(extension_tools.keys())}")
else:
    logger.info("MCP拡張機能は使用できません。可視化と科学的発見の自動化機能は無効です。")

# リソース定義 - このブロックは保持する
@mcp.resource("config://app")
def get_config() -> str:
    """アプリケーション設定データを提供します。"""
    config_data = {
        "app_name": "MIREX Algorithm Improver",
        "version": "1.0.0",
        "supported_file_types": ["wav", "mp3", "flac"],
        "workspace": str(CONFIG['workspace_dir']),
        "data_directories": {
            "audio": str(CONFIG['audio_dir']),
            "reference": str(CONFIG['reference_dir']),
            "results": str(CONFIG['evaluation_results_dir'])
        }
    }
    return json.dumps(config_data, indent=2)

# リソース定義 - このブロックは保持する
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """ユーザープロファイル情報を取得します。
    
    Args:
        user_id: ユーザーID
    """
    # 実際には認証・認可を行い、データベースからユーザー情報を取得する
    # このサンプルではダミーデータを返す
    dummy_profiles = {
        "admin": {
            "id": "admin",
            "name": "Administrator",
            "permissions": ["read", "write", "execute", "admin"],
            "created_at": "2023-01-01T00:00:00Z"
        },
        "guest": {
            "id": "guest",
            "name": "Guest User",
            "permissions": ["read"],
            "created_at": "2023-01-01T00:00:00Z"
        }
    }
    
    profile = dummy_profiles.get(user_id, {
        "id": user_id,
        "name": f"User {user_id}",
        "permissions": ["read", "write"],
        "created_at": datetime.now().isoformat()
    })
    
    return json.dumps(profile, indent=2)

@mcp.resource("file://detectors/{detector_name}.py")
async def get_detector_file(detector_name: str) -> str:
    """
    検出器ソースコードファイルを取得します。
    
    Args:
        detector_name: 検出器名
        
    Returns:
        str: ソースコード
    """
    from src.utils.path_utils import get_detector_path
    
    detector_path = get_detector_path(detector_name)
    if not detector_path or not detector_path.exists():
        raise FileNotFoundError(f"検出器ファイル {detector_name}.py が見つかりません")
    
    try:
        with open(detector_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"検出器ファイルの読み込みに失敗しました: {detector_name} - {e}")
        raise IOError(f"検出器ファイルの読み込みに失敗しました: {e}")

@mcp.resource("file://improved_versions/{detector_name}_{version}.py")
async def get_improved_detector_file(detector_name: str, version: str) -> str:
    """
    改善された検出器ソースコードファイルを取得します。
    
    Args:
        detector_name: 検出器名
        version: バージョン
        
    Returns:
        str: ソースコード
    """
    file_path = WORKSPACE_DIRS["improved_versions"] / f"{detector_name}_{version}.py"
    if not file_path.exists():
        raise FileNotFoundError(f"改善バージョン {detector_name}_{version}.py が見つかりません")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"改善バージョンファイルの読み込みに失敗しました: {detector_name}_{version}.py - {e}")
        raise IOError(f"改善バージョンファイルの読み込みに失敗しました: {e}")

@mcp.resource("file://evaluation_results/{session_id}/{run_id}/evaluation_result.json")
async def get_evaluation_result_file(session_id: str, run_id: str) -> str:
    """
    評価結果ファイルを取得します。
    
    Args:
        session_id: セッションID
        run_id: 実行ID
        
    Returns:
        str: 評価結果のJSON文字列
    """
    file_path = CONFIG["evaluation_results_dir"] / session_id / run_id / "evaluation_result.json"
    if not file_path.exists():
        raise FileNotFoundError(f"評価結果ファイルが見つかりません: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"評価結果ファイルの読み込みに失敗しました: {file_path} - {e}")
        raise IOError(f"評価結果ファイルの読み込みに失敗しました: {e}")

@mcp.resource("file://grid_search_results/{session_id}/{run_id}/grid_search_result.json")
async def get_grid_search_result_file(session_id: str, run_id: str) -> str:
    """
    グリッドサーチ結果ファイルを取得します。
    
    Args:
        session_id: セッションID
        run_id: 実行ID
        
    Returns:
        str: グリッドサーチ結果のJSON文字列
    """
    file_path = CONFIG["grid_search_results"] / session_id / run_id / "grid_search_result.json"
    if not file_path.exists():
        raise FileNotFoundError(f"グリッドサーチ結果ファイルが見つかりません: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"グリッドサーチ結果ファイルの読み込みに失敗しました: {file_path} - {e}")
        raise IOError(f"グリッドサーチ結果ファイルの読み込みに失敗しました: {e}")

# --- MCP プロンプトリソース ---
@mcp.prompt("code-improvement")
def get_code_improvement_prompt() -> str:
    """コード改善用プロンプトテンプレート"""
    return """
    以下の音楽情報検出アルゴリズムを改善してください。

    ## 現在のコード
    ```python
    {current_code}
    ```

    ## 評価結果
    {evaluation_result}

    ## 改善目標
    {improvement_goal}

    ## 具体的な指示
    - コードの構造と処理フローを保ちつつ、アルゴリズムの精度を改善してください
    - 特にノイズやリバーブなどの環境での耐性を強化してください
    - コード内のコメントは保持または改善してください
    - 機能が明確に理解できるよう、適切な変数名とコメントを使用してください
    - 完全なPythonスクリプトとして動作する最終的なコードを提供してください

    ## 回答形式
    ```python
    # ここに改善されたコード全体を記載
    ```
    """

@mcp.prompt("evaluation-analysis")
def get_evaluation_analysis_prompt() -> str:
    """評価結果分析用プロンプトテンプレート"""
    return """
    以下の音楽情報検出アルゴリズムの評価結果を分析し、改善点を指摘してください。

    ## アルゴリズム名
    {detector_name}

    ## 評価結果
    {evaluation_result}

    ## 分析観点
    - F値（F-measure）、適合率（Precision）、再現率（Recall）の値と傾向
    - 検出が難しいケースの傾向
    - アルゴリズム改善のための主要なポイント
    - 最も優先すべき改善項目

    ## 回答形式
    分析結果をJSONで提供してください：
    {
        "overall_performance": "全体的な性能評価",
        "strengths": ["強み1", "強み2", ...],
        "weaknesses": ["弱み1", "弱み2", ...],
        "improvement_suggestions": [
            {
                "priority": "高/中/低",
                "area": "改善領域（例: ノイズ耐性、オンセット検出など）",
                "description": "詳細な説明と改善アプローチ"
            },
            ...
        ]
    }
    """

@mcp.prompt("parameter-tuning")
def get_parameter_tuning_prompt() -> str:
    """パラメータチューニング用プロンプトテンプレート"""
    return """
    以下の音楽情報検出アルゴリズムのパラメータ探索結果を分析し、最適なパラメータ設定を提案してください。

    ## アルゴリズム名
    {detector_name}

    ## グリッドサーチ結果
    {grid_search_result}

    ## 分析観点
    - 最高のF値を達成したパラメータ組み合わせ
    - パラメータ値と性能の相関関係
    - 安定性と過学習のバランス
    - 推奨するパラメータ設定とその理由

    ## 回答形式
    分析結果をJSONで提供してください：
    {
        "best_parameters": {
            "param1": 値1,
            "param2": 値2,
            ...
        },
        "performance": {
            "f_measure": F値,
            "precision": 適合率,
            "recall": 再現率
        },
        "analysis": "パラメータと性能の関係に関する詳細な分析",
        "recommendation": "最終的な推奨設定とその理由"
    }
    """

@mcp.prompt("workflow-guide")
def get_workflow_guide_prompt() -> str:
    """ワークフローガイド用プロンプトテンプレート"""
    return """
    # MIRアルゴリズム改善ワークフロー

    音楽情報検索（MIR）アルゴリズムを改善するための標準的なワークフローです。
    以下のステップに従って進めてください。

    ## ステップ1: セッション開始
    ```python
    session = await start_session(base_algorithm="検出器名")
    session_id = session["id"]
    ```

    ## ステップ2: 現在のコードを取得
    ```python
    current_code = await get_code(detector_name="検出器名")
    ```

    ## ステップ3: 初期評価を実行
    ```python
    eval_job = await run_evaluation(detector_name="検出器名")
    eval_result = await get_job_status(job_id=eval_job["job_id"])
    # 完了するまで定期的にステータスを確認
    ```

    ## ステップ4: コードを改善
    ```python
    improvement_prompt = "改善目標を含むプロンプト"
    improve_job = await improve_code(prompt=improvement_prompt, session_id=session_id)
    improved_code_result = await get_job_status(job_id=improve_job["job_id"])
    # 完了するまで定期的にステータスを確認
    ```

    ## ステップ5: 改善コードを保存
    ```python
    save_result = await save_code(detector_name="検出器名", code=improved_code, version="v1")
    ```

    ## ステップ6: 改善コードを評価
    ```python
    new_eval_job = await run_evaluation(detector_name=f"検出器名_{version}")
    new_eval_result = await get_job_status(job_id=new_eval_job["job_id"])
    # 完了するまで定期的にステータスを確認
    ```

    ## ステップ7: 評価結果を比較
    • F値、適合率、再現率を比較
    • 改善があるかを判断
    • 必要に応じてステップ4-7を繰り返す

    ## ステップ8: パラメータ最適化（オプション）
    ```python
    grid_params = {"param1": [val1, val2], "param2": [val1, val2]}
    grid_job = await run_grid_search(detector_name="検出器名_最終版", grid_params_json=json.dumps(grid_params))
    grid_result = await get_job_status(job_id=grid_job["job_id"])
    # 完了するまで定期的にステータスを確認
    ```

    ## ステップ9: 最終レポート作成
    改善プロセス全体の概要と成果をまとめます。
    """

# サーバー起動はそのまま
if __name__ == "__main__":
    # コマンドライン引数の解析
    import argparse
    parser = argparse.ArgumentParser(description="MCP対応AIアルゴリズム改善サーバー")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", 5002)),
                        help="サーバーのポート (デフォルト: 環境変数MCP_PORTまたは5002)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default=os.environ.get("MCP_LOG_LEVEL", "INFO"),
                        help="ログレベル (デフォルト: 環境変数MCP_LOG_LEVELまたはINFO)")
    parser.add_argument("--workspace", type=str, default=os.environ.get("MIREX_WORKSPACE"),
                        help="ワークスペースディレクトリ (デフォルト: 環境変数MIREX_WORKSPACE)")
    args = parser.parse_args()
    
    # ログレベル設定
    logging.getLogger('mcp_server').setLevel(getattr(logging, args.log_level))
    
    # ワークスペースディレクトリが指定された場合は環境変数を設定
    if args.workspace:
        os.environ["MIREX_WORKSPACE"] = args.workspace
        logger.info(f"ワークスペースディレクトリを設定: {args.workspace}")
    
    # クリーンアップスレッドの開始
    start_cleanup_thread()
    
    # セッションと非同期ジョブの読み込み
    try:
        load_sessions_and_jobs()
        logger.info("セッションと非同期ジョブを読み込みました")
    except Exception as e:
        logger.warning(f"セッションと非同期ジョブの読み込みに失敗しました: {e}")
    
    # シグナルハンドラの設定
    def signal_handler(sig, frame):
        """
        シグナル受信時の処理
        - セッションと非同期ジョブを保存
        - ログメッセージを出力
        - プログラム終了
        """
        logger.info(f"シグナル {sig} を受信しました。終了します...")
        try:
            save_sessions_and_jobs()
            logger.info("セッションと非同期ジョブを保存しました")
        except Exception as e:
            logger.error(f"セッションと非同期ジョブの保存に失敗しました: {e}")
        sys.exit(0)
    
    # シグナルハンドラの登録
    import signal
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill
    
    # サーバー起動メッセージ
    logger.info(f"MCPサーバーを起動しています（ポート: {args.port}）")
    logger.info(f"ワークスペースディレクトリ: {workspace_dir}")
    
    # FastMCPサーバーを起動（既存のmcpオブジェクトを使用）
    try:
        # ポート番号を設定（既存のmcpオブジェクトを使用）
        mcp.port = args.port
        
        # サーバーを起動
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("キーボード割り込みを受信しました。終了します...")
        save_sessions_and_jobs()
    except Exception as e:
        logger.critical(f"サーバー起動中にエラーが発生しました: {e}")
        save_sessions_and_jobs()
        sys.exit(1)