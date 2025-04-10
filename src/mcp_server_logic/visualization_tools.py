import logging
import os
import tempfile
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine, List, Awaitable
import asyncio

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP, Image
from . import db_utils
from src.utils.path_utils import (
    ensure_dir,
    get_db_dir,
    get_output_base_dir,
    get_workspace_dir, # Keep for allowed_base_dirs
    validate_path_within_allowed_dirs
)
from src.utils.exception_utils import FileError, ConfigError, MirexError, VisualizationError
# オプション: Pillow, librosa, matplotlib
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
try:
    import librosa
    import matplotlib.pyplot as plt
    HAS_AUDIO_VIZ = True
except ImportError:
    HAS_AUDIO_VIZ = False

# visualization と science_automation モジュール
# (mcp_server_extensions.py から移動)
try:
    from src.visualization.code_impact_graph import CodeChangeAnalyzer
    HAS_CODE_IMPACT = True
except ImportError:
    HAS_CODE_IMPACT = False
try:
    from src.visualization.heatmap_generator import HeatmapGenerator
    HAS_HEATMAP = True
except ImportError:
    HAS_HEATMAP = False
try:
    from src.science_automation.output_generator import OutputGenerator
    HAS_SCIENCE_OUTPUT = True
except ImportError:
    HAS_SCIENCE_OUTPUT = False

logger = logging.getLogger('mcp_server.visualization_tools')

# --- 修正: 非同期 Task Functions --- #

# --- 修正: _run_create_thumbnail を非同期化 --- #
async def _run_create_thumbnail(
    job_id: str,
    allowed_base_dirs: List[Path],
    # --- 修正: 引数名と型ヒント変更 --- #
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    image_path: str,
    size: int = 100,
    session_id: Optional[str] = None
) -> Image:
    """サムネイル作成ジョブ (非同期)"""
    logger.info(f"[Job {job_id}] Creating thumbnail for '{image_path}'... Session: {session_id}")
    # --- 修正: await を使用 --- #
    await add_history_async_func(session_id, "job_started", {"job_id": job_id, "tool_name": "create_thumbnail", "image_path": image_path, "size": size})

    if not HAS_PIL:
        error_msg = "Pillow is not installed. Cannot create thumbnail."
        logger.error(f"[Job {job_id}] {error_msg}")
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": error_msg})
        raise ImportError(error_msg)

    try:
        # Validate path (同期)
        validated_image_path = validate_path_within_allowed_dirs(image_path, allowed_base_dirs, check_existence=True)
        logger.debug(f"[Job {job_id}] Validated image path: {validated_image_path}")

        # --- 修正: Pillow処理を run_in_executor で実行 --- #
        loop = asyncio.get_running_loop()
        def sync_thumbnail_creation():
            with PILImage.open(validated_image_path) as img:
                img.thumbnail((size, size))
                # メモリ上で処理
                with tempfile.SpooledTemporaryFile() as output:
                    img_format = img.format or 'PNG' # 元のフォーマットを保持、なければPNG
                    img.save(output, format=img_format)
                    output.seek(0)
                    image_data = output.read()
                    return image_data, img_format

        image_data, img_format = await loop.run_in_executor(None, sync_thumbnail_creation)

        result = Image(data=image_data, format=img_format.lower())
        logger.info(f"[Job {job_id}] Thumbnail created successfully.")
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_completed", {"job_id": job_id, "result_summary": f"Thumbnail {size}x{size} created"})
        # MCPのImage型で返す
        return result
    except (ImportError, FileNotFoundError, ValueError) as e:
        error_msg = f"Thumbnail creation failed (validation/setup/file issue): {e}"
        logger.error(f"[Job {job_id}] {error_msg}", exc_info=True)
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": str(e)})
        raise VisualizationError(error_msg) from e
    except Exception as e:
        error_msg = f"Error creating thumbnail for '{image_path}': {e}"
        logger.error(f"[Job {job_id}] {error_msg}", exc_info=True)
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": str(e)})
        raise VisualizationError(error_msg) from e

# --- 修正: _run_visualize_spectrogram を非同期化 --- #
async def _run_visualize_spectrogram(
    job_id: str,
    allowed_base_dirs: List[Path],
    # --- 修正: 引数名と型ヒント変更 --- #
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    audio_file: str,
    output_format: str = "png",
    session_id: Optional[str] = None
) -> Image:
    """スペクトログラム生成ジョブ (非同期)"""
    logger.info(f"[Job {job_id}] Generating spectrogram for '{audio_file}'... Session: {session_id}")
    # --- 修正: await を使用 --- #
    await add_history_async_func(session_id, "job_started", {"job_id": job_id, "tool_name": "visualize_spectrogram", "audio_file": audio_file, "output_format": output_format})

    if not HAS_AUDIO_VIZ:
        error_msg = "librosa or matplotlib not installed. Cannot generate spectrogram."
        logger.error(f"[Job {job_id}] {error_msg}")
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": error_msg})
        raise ImportError(error_msg)

    try:
        # Validate path (同期)
        validated_audio_path = validate_path_within_allowed_dirs(audio_file, allowed_base_dirs, check_existence=True)
        logger.debug(f"[Job {job_id}] Validated audio path: {validated_audio_path}")

        # --- 修正: librosa/matplotlib 処理を run_in_executor で実行 --- #
        loop = asyncio.get_running_loop()
        def sync_spectrogram_generation():
            y, sr = librosa.load(str(validated_audio_path))
            fig, ax = plt.subplots(figsize=(10, 4))
            image_data = None
            try:
                D = librosa.stft(y)
                S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
                img_plot = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax)
                fig.colorbar(img_plot, ax=ax, format="%+2.0f dB")
                ax.set_title(f'Spectrogram: {validated_audio_path.name}')

                with tempfile.SpooledTemporaryFile() as output:
                    plt.savefig(output, format=output_format, bbox_inches='tight')
                    output.seek(0)
                    image_data = output.read()
            finally:
                plt.close(fig) # 図を閉じる
            return image_data

        image_data = await loop.run_in_executor(None, sync_spectrogram_generation)
        if image_data is None:
            raise VisualizationError("Spectrogram generation failed (image data is None)")

        result = Image(data=image_data, format=output_format)
        logger.info(f"[Job {job_id}] Spectrogram generated successfully.")
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_completed", {"job_id": job_id, "result_summary": f"Spectrogram ({output_format}) generated"})
        return result
    except (ImportError, FileNotFoundError, ValueError) as e:
        error_msg = f"Spectrogram generation failed (validation/setup/file issue): {e}"
        logger.error(f"[Job {job_id}] {error_msg}", exc_info=True)
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": str(e)})
        raise VisualizationError(error_msg) from e
    except Exception as e:
        error_msg = f"Error generating spectrogram for '{audio_file}': {e}"
        logger.error(f"[Job {job_id}] {error_msg}", exc_info=True)
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": str(e)})
        raise VisualizationError(error_msg) from e

# --- Tasks from mcp_server_extensions.py --- #

# --- 修正: _run_visualize_code_impact を非同期化 --- #
async def _run_visualize_code_impact(
    job_id: str,
    db_path: Path,
    output_base_dir: Path, # Use base dir
    # --- 修正: 引数名と型ヒント変更 --- #
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    session_id: str,
    iteration: int
) -> Dict[str, Any]:
    """コード影響可視化の実行タスク (非同期)"""
    logger.info(f"[Job {job_id}] Visualizing code impact for iteration {iteration}... Session: {session_id}")
    # --- 修正: await を使用 --- #
    await add_history_async_func(session_id, "job_started", {"job_id": job_id, "tool_name": "visualize_code_impact", "iteration": iteration})

    if not HAS_CODE_IMPACT:
        error_msg = "CodeChangeAnalyzer not available (src.visualization.code_impact_graph). Cannot visualize code impact."
        logger.error(f"[Job {job_id}] {error_msg}")
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": error_msg})
        raise ImportError(error_msg)

    try:
        # --- 修正: DBアクセスを非同期化 --- #
        row = await db_utils.db_fetch_one_async(db_path, "SELECT history FROM sessions WHERE session_id = ?", (session_id,))
        if not row:
            raise ValueError(f"Session {session_id} not found.")
        # JSONデコードは同期的
        history = json.loads(row["history"] or '[]')

        original_code = None
        improved_code = None
        # イテレーションに対応するコードを探すロジック (改善の余地あり)
        iter_found = False
        improve_request_event = None
        improve_result_event = None

        for event in history:
            event_data = event.get('data', {})
            # Check for iteration start marker if available
            if event.get('type') == 'iteration_started' and event_data.get('iteration') == iteration:
                 iter_found = True
                 # Reset search within iteration boundary if needed
                 improve_request_event = None
                 improve_result_event = None

            # If within the target iteration (or if no iteration marker found yet, search globally)
            # Prioritize events that explicitly mention the iteration
            event_iter = event_data.get('iteration')
            if iter_found or event_iter == iteration:
                 if event.get('type') == 'improve_code_request':
                     # Use the latest request within the iteration
                      if event_data.get('original_code'):
                           improve_request_event = event
                 elif event.get('type') == 'improve_code_result':
                      if event_data.get('improved_code'):
                           # Use the first result after the request within the iteration
                           improve_result_event = event
                           if iter_found or event_iter == iteration: # Found result for specific iter
                               break # Stop search if we are sure it's for the correct iteration

        # Fallback: if no iteration marker or specific match, use the latest globally found pair
        if not improve_result_event:
            for event in reversed(history): # Search backwards for latest pair globally
                 event_data = event.get('data', {})
                 if not improve_result_event and event.get('type') == 'improve_code_result' and event_data.get('improved_code'):
                     improve_result_event = event
                 elif not improve_request_event and event.get('type') == 'improve_code_request' and event_data.get('original_code'):
                     improve_request_event = event
                 if improve_request_event and improve_result_event:
                     break # Found the latest pair

        if improve_request_event:
            original_code = improve_request_event.get('data', {}).get('original_code')
        if improve_result_event:
             improved_code = improve_result_event.get('data', {}).get('improved_code')

        if not original_code or not improved_code:
             error_msg = f"Original or improved code for iteration {iteration} not found in session history for session {session_id}."
             logger.error(f"[Job {job_id}] {error_msg}")
             # --- 修正: await を使用 --- #
             await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": error_msg})
             raise ValueError(error_msg)

        # 出力ディレクトリ設定 (引数を使用)
        vis_dir = output_base_dir / "visualizations" / session_id
        ensure_dir(vis_dir)
        output_path = vis_dir / f"code_impact_iter_{iteration}.png"
        output_path_str = str(output_path.resolve()) # Use resolved absolute path

        # --- 修正: 可視化実行を run_in_executor で実行 --- #
        loop = asyncio.get_running_loop()
        def sync_visualization():
            analyzer = CodeChangeAnalyzer()
            analyzer.analyze_code_change(original_code, improved_code)
            analyzer.visualize_impact(output_path_str, f"Code Change Impact - Iteration {iteration}")

        await loop.run_in_executor(None, sync_visualization)

        result = {"status": "success", "output_path": output_path_str}
        logger.info(f"[Job {job_id}] Code impact visualization saved to {output_path_str}")
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_completed", {"job_id": job_id, "result": result})
        # Also add specific viz history for easier finding
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "visualization_created", {"type": "code_impact", "iteration": iteration, "path": output_path_str})

        return result
    except (ImportError, FileNotFoundError, ValueError, ConfigError) as e:
        error_msg = f"Code impact visualization failed (setup/data/validation issue): {e}"
        logger.error(f"[Job {job_id}] {error_msg}", exc_info=True)
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": str(e)})
        raise VisualizationError(error_msg) from e
    except Exception as e:
        error_msg = f"Error visualizing code impact: {e}"
        logger.error(f"[Job {job_id}] {error_msg}", exc_info=True)
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": str(e)})
        raise VisualizationError(error_msg) from e

# --- 修正: _run_generate_heatmap を非同期化 --- #
async def _run_generate_heatmap(
    job_id: str,
    db_path: Path,
    output_base_dir: Path, # Use base dir
    # --- 修正: 引数名と型ヒント変更 --- #
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    session_id: str,
    param_x: str,
    param_y: str,
    metric: str
) -> Dict[str, Any]:
    """性能ヒートマップ生成タスク (非同期)"""
    logger.info(f"[Job {job_id}] Generating performance heatmap ({param_x} vs {param_y} for {metric})... Session: {session_id}")
    # --- 修正: await を使用 --- #
    await add_history_async_func(session_id, "job_started", {"job_id": job_id, "tool_name": "generate_heatmap", "param_x": param_x, "param_y": param_y, "metric": metric})

    if not HAS_HEATMAP:
         error_msg = "HeatmapGenerator not available (src.visualization.heatmap_generator). Cannot generate heatmap."
         logger.error(f"[Job {job_id}] {error_msg}")
         # --- 修正: await を使用 --- #
         await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": error_msg})
         raise ImportError(error_msg)

    try:
        # --- 修正: DBアクセスを非同期化 --- #
        row = await db_utils.db_fetch_one_async(db_path, "SELECT history FROM sessions WHERE session_id = ?", (session_id,))
        if not row: raise ValueError(f"Session {session_id} not found.")
        # JSONデコードは同期的
        history = json.loads(row["history"] or '[]')

        grid_search_results_list = None
        # Find the latest grid search completion event (同期)
        for event in reversed(history):
            if event.get('type') == 'grid_search_complete' or event.get('type') == 'parameter_optimization_complete':
                 grid_data = event.get('data', {})
                 # Handle different potential structures for results
                 results_candidate = grid_data.get('grid_search_results', grid_data.get('results', grid_data))
                 if isinstance(results_candidate, list) and results_candidate:
                      # Check if items look like results (have 'params' and 'metrics')
                      first_item = results_candidate[0]
                      if isinstance(first_item, dict) and 'params' in first_item and 'metrics' in first_item:
                           grid_search_results_list = results_candidate
                           break
                 elif isinstance(results_candidate, dict) and isinstance(results_candidate.get('results'), list):
                      # Handle nested 'results' key
                      nested_results = results_candidate['results']
                      if nested_results and isinstance(nested_results[0], dict) and 'params' in nested_results[0]:
                           grid_search_results_list = nested_results
                           break

        if not grid_search_results_list:
             error_msg = f"No suitable grid search results found in session history for session {session_id}."
             logger.error(f"[Job {job_id}] {error_msg}")
             # --- 修正: await を使用 --- #
             await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": error_msg})
             raise ValueError(error_msg)

        # 出力ディレクトリ設定 (同期)
        vis_dir = output_base_dir / "visualizations" / session_id
        ensure_dir(vis_dir)
        heatmap_filename = f"heatmap_{param_x}_vs_{param_y}_for_{metric}.png"
        output_path = vis_dir / heatmap_filename
        output_path_str = str(output_path.resolve())

        # --- 修正: ヒートマップ生成実行を run_in_executor で実行 --- #
        loop = asyncio.get_running_loop()
        def sync_heatmap_generation():
            generator = HeatmapGenerator(grid_search_results_list)
            generator.generate_heatmap(
                param_x=param_x,
                param_y=param_y,
                metric=metric,
                output_path=output_path_str,
                title=f"Performance Heatmap: {metric} ({param_x} vs {param_y})"
            )

        await loop.run_in_executor(None, sync_heatmap_generation)

        result = {"status": "success", "output_path": output_path_str}
        logger.info(f"[Job {job_id}] Heatmap visualization saved to {output_path_str}")
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_completed", {"job_id": job_id, "result": result})
        # Also add specific viz history
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "visualization_created", {"type": "heatmap", "params": [param_x, param_y], "metric": metric, "path": output_path_str})

        return result
    except (ImportError, FileNotFoundError, ValueError, KeyError, ConfigError) as e:
        error_msg = f"Heatmap generation failed (setup/data/validation issue): {e}"
        logger.error(f"[Job {job_id}] {error_msg}", exc_info=True)
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": str(e)})
        raise VisualizationError(error_msg) from e
    except Exception as e:
        error_msg = f"Error generating heatmap: {e}"
        logger.error(f"[Job {job_id}] {error_msg}", exc_info=True)
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": str(e)})
        raise VisualizationError(error_msg) from e

# --- 修正: _run_generate_scientific_outputs を非同期化 --- #
async def _run_generate_scientific_outputs(
    job_id: str,
    db_path: Path,
    output_base_dir: Path, # Use base dir
    # --- 修正: 引数名と型ヒント変更 --- #
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]],
    session_id: str
    # config: Dict[str, Any], # Removed, pass specific paths
    ) -> Dict[str, Any]:
    """科学的成果物生成タスク (非同期)"""
    logger.info(f"[Job {job_id}] Generating scientific outputs for session {session_id}...")
    # --- 修正: await を使用 --- #
    await add_history_async_func(session_id, "job_started", {"job_id": job_id, "tool_name": "generate_scientific_outputs"})

    if not HAS_SCIENCE_OUTPUT:
        error_msg = "OutputGenerator not available (src.science_automation.output_generator). Cannot generate outputs."
        logger.error(f"[Job {job_id}] {error_msg}")
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": error_msg})
        raise ImportError(error_msg)

    try:
        # --- 修正: DBアクセスを非同期化 --- #
        row = await db_utils.db_fetch_one_async(db_path, "SELECT history FROM sessions WHERE session_id = ?", (session_id,))
        if not row:
            raise ValueError(f"Session {session_id} not found.")
        # JSONデコードは同期的
        history = json.loads(row["history"] or '[]')

        # Define output directory based on output_base_dir (同期)
        output_dir = output_base_dir / "scientific_outputs" / session_id
        ensure_dir(output_dir)
        output_dir_str = str(output_dir.resolve())

        # --- 修正: Generate outputs を run_in_executor で実行 --- #
        loop = asyncio.get_running_loop()
        def sync_output_generation():
            generator = OutputGenerator(history, output_dir_str)
            # Example: Generate a summary report (add more specific methods as needed)
            report_path = generator.generate_summary_report() # Assuming this method exists and returns path
            # --- 複数のファイルが生成される可能性を考慮 --- #
            # generated_files = generator.generate_all() # 例
            return [report_path] # 仮にリストで返す

        generated_files = await loop.run_in_executor(None, sync_output_generation)

        result = {
            "status": "success",
            "output_directory": output_dir_str,
            "generated_files": generated_files # Store the list of paths
        }
        logger.info(f"[Job {job_id}] Scientific outputs generated in {output_dir_str}")
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_completed", {"job_id": job_id, "result": result})
        # Also add specific history
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "scientific_output_generated", {"path": output_dir_str, "files": result["generated_files"]})

        return result
    except (ImportError, ValueError, ConfigError, FileError) as e:
        error_msg = f"Scientific output generation failed (setup/data/validation issue): {e}"
        logger.error(f"[Job {job_id}] {error_msg}", exc_info=True)
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": str(e)})
        raise MirexError(error_msg) from e # Or a more specific error type
    except Exception as e:
        error_msg = f"Error generating scientific outputs: {e}"
        logger.error(f"[Job {job_id}] {error_msg}", exc_info=True)
        # --- 修正: await を使用 --- #
        await add_history_async_func(session_id, "job_failed", {"job_id": job_id, "error": str(e)})
        raise MirexError(error_msg) from e

def register_visualization_tools(
    mcp: FastMCP,
    config: Dict[str, Any],
    start_async_job_func: Callable[..., Coroutine[Any, Any, str]], # Should return job_id coroutine
    # --- 修正: 引数名と型ヒント変更 --- #
    add_history_async_func: Callable[..., Awaitable[Dict[str, Any]]]
):
    """可視化関連ツールをMCPに登録"""
    logger.info("Registering visualization tools...")

    try:
        # db_path は 非同期関数内で取得する
        output_base_dir = get_output_base_dir(config)
        workspace_dir = get_workspace_dir(config) # Needed for allowed dirs
        allowed_base_dirs = [workspace_dir, output_base_dir]
        logger.info(f"Visualization tools allowed base directories: {allowed_base_dirs}")
    except ConfigError as e:
        logger.error(f"Failed to get necessary directories from config: {e}", exc_info=True)
        # Optionally raise or prevent registration
        return

    @mcp.tool("create_thumbnail")
    async def create_thumbnail_tool(image_path: str, size: int = 100, session_id: Optional[str] = None) -> Dict[str, Any]:
        """指定された画像ファイルのサムネイルを生成します。"""
        job_id = await start_async_job_func(
            # --- 修正: 非同期タスク関数を渡す --- #
            _run_create_thumbnail,
            tool_name="create_thumbnail",
            session_id=session_id,
            # Pass required args to the task function
            allowed_base_dirs=allowed_base_dirs,
            # --- 修正: 引数名変更 --- #
            add_history_async_func=add_history_async_func,
            image_path=image_path,
            size=size
        )
        return {"status": "job_started", "job_id": job_id}

    @mcp.tool("visualize_spectrogram")
    async def visualize_spectrogram_tool(audio_file: str, output_format: str = "png", session_id: Optional[str] = None) -> Dict[str, Any]:
        """指定された音声ファイルのスペクトログラム画像を生成します。"""
        job_id = await start_async_job_func(
            # --- 修正: 非同期タスク関数を渡す --- #
            _run_visualize_spectrogram,
            tool_name="visualize_spectrogram",
            session_id=session_id,
            # Pass required args
            allowed_base_dirs=allowed_base_dirs,
            # --- 修正: 引数名変更 --- #
            add_history_async_func=add_history_async_func,
            audio_file=audio_file,
            output_format=output_format
        )
        return {"status": "job_started", "job_id": job_id}

    @mcp.tool("visualize_code_impact")
    async def visualize_code_impact_tool(session_id: str, iteration: int) -> Dict[str, Any]:
        """指定されたセッションとイテレーションのコード変更の影響を可視化します。"""
        db_path = Path(config['paths']['db_path']) # DBパスをここで取得
        job_id = await start_async_job_func(
            # --- 修正: 非同期タスク関数を渡す --- #
            _run_visualize_code_impact,
            tool_name="visualize_code_impact",
            session_id=session_id,
            # Pass required args
            db_path=db_path,
            output_base_dir=output_base_dir,
            # --- 修正: 引数名変更 --- #
            add_history_async_func=add_history_async_func,
            iteration=iteration
        )
        return {"status": "job_started", "job_id": job_id}

    @mcp.tool("generate_heatmap")
    async def generate_heatmap_tool(session_id: str, param_x: str, param_y: str, metric: str) -> Dict[str, Any]:
        """指定されたセッションのグリッドサーチ結果から性能ヒートマップを生成します。"""
        db_path = Path(config['paths']['db_path']) # DBパスをここで取得
        job_id = await start_async_job_func(
            # --- 修正: 非同期タスク関数を渡す --- #
            _run_generate_heatmap,
            tool_name="generate_heatmap",
            session_id=session_id,
            # Pass required args
            db_path=db_path,
            output_base_dir=output_base_dir,
            # --- 修正: 引数名変更 --- #
            add_history_async_func=add_history_async_func,
            param_x=param_x,
            param_y=param_y,
            metric=metric
        )
        return {"status": "job_started", "job_id": job_id}

    # Example registration for the scientific output tool (if intended)
    @mcp.tool("generate_scientific_outputs")
    async def generate_scientific_outputs_tool(session_id: str) -> Dict[str, Any]:
        """指定されたセッションの履歴から科学的な成果物 (レポート、図など) を生成します。"""
        db_path = Path(config['paths']['db_path']) # DBパスをここで取得
        job_id = await start_async_job_func(
            # --- 修正: 非同期タスク関数を渡す --- #
            _run_generate_scientific_outputs,
            tool_name="generate_scientific_outputs",
            session_id=session_id,
            # Pass required args
            db_path=db_path,
            output_base_dir=output_base_dir,
            # --- 修正: 引数名変更 --- #
            add_history_async_func=add_history_async_func
        )
        return {"status": "job_started", "job_id": job_id}

    logger.info("Visualization tools registered.")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 