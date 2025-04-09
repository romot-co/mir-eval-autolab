import logging
import os
import tempfile
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine

# 関連モジュールインポート
from mcp.server.fastmcp import FastMCP, Image
from . import db_utils
from src.utils.path_utils import ensure_dir
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

# --- Synchronous Task Functions --- #

def _run_create_thumbnail(
    job_id: str,
    workspace_dir: Path,
    add_history_sync_func: Callable,
    image_path: str,
    size: int = 100,
    session_id: Optional[str] = None
) -> Image:
    """サムネイル作成ジョブ (同期)"""
    logger.info(f"[Job {job_id}] Creating thumbnail for '{image_path}'... Session: {session_id}")
    if not HAS_PIL:
        raise ImportError("Pillow is not installed. Cannot create thumbnail.")

    full_image_path = Path(image_path)
    if not full_image_path.is_absolute():
         # If relative, assume it's relative to workspace
         full_image_path = workspace_dir / image_path

    if not full_image_path.exists():
         # Try original path again (maybe it was absolute but outside workspace)
         original_path_attempt = Path(image_path)
         if original_path_attempt.exists():
              full_image_path = original_path_attempt
         else:
              raise FileNotFoundError(f"Image file not found at '{image_path}' or '{full_image_path}'")

    try:
        with PILImage.open(full_image_path) as img:
            img.thumbnail((size, size))
            # メモリ上で処理
            with tempfile.SpooledTemporaryFile() as output:
                img_format = img.format or 'PNG' # 元のフォーマットを保持、なければPNG
                img.save(output, format=img_format)
                output.seek(0)
                image_data = output.read()
        logger.info(f"[Job {job_id}] Thumbnail created successfully.")
        # MCPのImage型で返す
        return Image(data=image_data, format=img_format.lower())
    except (ImportError, FileNotFoundError) as ie:
        logger.error(f"[Job {job_id}] Error creating thumbnail (setup or file issue): {ie}", exc_info=True)
        raise VisualizationError(f"Thumbnail creation failed: {ie}") from ie
    except Exception as e:
        logger.error(f"[Job {job_id}] Error creating thumbnail for '{full_image_path}': {e}", exc_info=True)
        raise VisualizationError(f"Thumbnail creation failed: {e}") from e

def _run_visualize_spectrogram(
    job_id: str,
    workspace_dir: Path,
    add_history_sync_func: Callable,
    audio_file: str,
    output_format: str = "png",
    session_id: Optional[str] = None
) -> Image:
    """スペクトログラム生成ジョブ (同期)"""
    logger.info(f"[Job {job_id}] Generating spectrogram for '{audio_file}'... Session: {session_id}")
    if not HAS_AUDIO_VIZ:
        raise ImportError("librosa or matplotlib not installed. Cannot generate spectrogram.")

    full_audio_path = Path(audio_file)
    if not full_audio_path.is_absolute():
        # If relative, assume it's relative to workspace
        full_audio_path = workspace_dir / audio_file

    if not full_audio_path.exists():
         # Try original path again
         original_path_attempt = Path(audio_file)
         if original_path_attempt.exists():
              full_audio_path = original_path_attempt
         else:
              raise FileNotFoundError(f"Audio file not found at '{audio_file}' or '{full_audio_path}'")

    try:
        y, sr = librosa.load(str(full_audio_path))
        fig, ax = plt.subplots(figsize=(10, 4))
        # Use amplitude_to_db and specshow
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(f'Spectrogram: {full_audio_path.name}')

        # メモリ上で画像を保存
        with tempfile.SpooledTemporaryFile() as output:
            plt.savefig(output, format=output_format, bbox_inches='tight')
            plt.close(fig) # 図を閉じる
            output.seek(0)
            image_data = output.read()

        logger.info(f"[Job {job_id}] Spectrogram generated successfully.")
        return Image(data=image_data, format=output_format)
    except (ImportError, FileNotFoundError) as ie:
        logger.error(f"[Job {job_id}] Error generating spectrogram (setup or file issue): {ie}", exc_info=True)
        if 'plt' in locals(): plt.close('all')
        raise VisualizationError(f"Spectrogram generation failed: {ie}") from ie
    except Exception as e:
        logger.error(f"[Job {job_id}] Error generating spectrogram for '{full_audio_path}': {e}", exc_info=True)
        if 'plt' in locals(): plt.close('all') # エラー時に図が残らないように
        raise VisualizationError(f"Spectrogram generation failed: {e}") from e

# --- Tasks from mcp_server_extensions.py --- #

def _run_visualize_code_impact(
    job_id: str,
    db_path: Path,
    visualizations_dir: Path,
    add_history_sync_func: Callable,
    session_id: str,
    iteration: int
) -> Dict[str, Any]:
    """コード影響可視化の実行タスク (同期)"""
    logger.info(f"[Job {job_id}] Visualizing code impact for iteration {iteration}... Session: {session_id}")
    if not HAS_CODE_IMPACT:
        raise ImportError("CodeChangeAnalyzer not available.")

    try:
        # DBから履歴を取得して original_code と improved_code を見つける
        row = db_utils._db_fetch_one(db_path, "SELECT history FROM sessions WHERE session_id = ?", (session_id,))
        if not row:
            raise ValueError(f"Session {session_id} not found.")
        history = json.loads(row["history"] or '[]')

        original_code = None
        improved_code = None
        # イテレーションに対応するコードを探すロジック (より堅牢に)
        iter_found = False
        for event in history:
            event_data = event.get('data', {})
            if event.get('type') == 'iteration_started' and event_data.get('iteration') == iteration:
                 iter_found = True
            if iter_found:
                 # ここでは単純化: improve_code_request と improve_code_result を探す
                 # プロンプトや履歴の構造に依存するため、要調整
                 if event.get('type') == 'improve_code_request':
                      original_code = event_data.get('original_code') # 仮定
                 elif event.get('type') == 'improve_code_result':
                      improved_code = event_data.get('improved_code') # 仮定
                      break # 最初に見つかった改善後コードを使用

        if not original_code or not improved_code:
             logger.error(f"[Job {job_id}] Original or improved code for iteration {iteration} not found in session history.")
             raise ValueError(f"Could not find original and improved code for iteration {iteration} in session history.")

        # 出力ディレクトリ設定 (引数を使用)
        vis_dir = visualizations_dir / session_id
        ensure_dir(vis_dir)
        output_path = vis_dir / f"code_impact_iter_{iteration}.png"

        # 可視化実行
        analyzer = CodeChangeAnalyzer()
        analyzer.analyze_code_change(original_code, improved_code)
        analyzer.visualize_impact(str(output_path), f"Code Change Impact - Iteration {iteration}")

        result = {"status": "success", "output_path": str(output_path)}
        try:
            add_history_sync_func(session_id, "visualization_created", {"type": "code_impact", "iteration": iteration, "path": str(output_path)})
        except Exception as hist_e:
             logger.warning(f"[Job {job_id}] Failed to add code impact viz history: {hist_e}")

        logger.info(f"[Job {job_id}] Code impact visualization saved to {output_path}")
        return result
    except (ImportError, FileNotFoundError, ValueError, ConfigError) as ie:
        logger.error(f"[Job {job_id}] Error visualizing code impact (setup or data issue): {ie}", exc_info=True)
        raise VisualizationError(f"Code impact visualization failed: {ie}") from ie
    except Exception as e:
        logger.error(f"[Job {job_id}] Error visualizing code impact: {e}", exc_info=True)
        raise VisualizationError(f"Code impact visualization failed: {e}") from e

def _run_generate_heatmap(
    job_id: str,
    db_path: Path,
    visualizations_dir: Path,
    add_history_sync_func: Callable,
    session_id: str,
    param_x: str,
    param_y: str,
    metric: str
) -> Dict[str, Any]:
    """性能ヒートマップ生成タスク (同期)"""
    logger.info(f"[Job {job_id}] Generating performance heatmap ({param_x} vs {param_y} for {metric})... Session: {session_id}")
    if not HAS_HEATMAP:
         raise ImportError("HeatmapGenerator not available.")

    try:
        # DBからグリッドサーチ結果を含む履歴を取得 (引数を使用)
        row = db_utils._db_fetch_one(db_path, "SELECT history FROM sessions WHERE session_id = ?", (session_id,))
        if not row: raise ValueError(f"Session {session_id} not found.")
        history = json.loads(row["history"] or '[]')

        grid_search_results = None
        for event in reversed(history):
            if event.get('type') == 'grid_search_complete' or event.get('type') == 'parameter_optimization_complete':
                 grid_data = event.get('data', {}).get('grid_search_results', event.get('data'))
                 if grid_data and isinstance(grid_data.get('results'), list):
                      grid_search_results = grid_data['results']
                      break
                 elif isinstance(event.get('data'), list):
                      if event['data'] and isinstance(event['data'][0], dict) and 'params' in event['data'][0]:
                           grid_search_results = event['data']
                           break

        if not grid_search_results:
             raise ValueError("No suitable grid search results found in session history.")

        # 出力ディレクトリ設定 (引数を使用)
        vis_dir = visualizations_dir / session_id
        ensure_dir(vis_dir)
        output_path = vis_dir / f"heatmap_{param_x}_{param_y}_{metric.replace('.', '_')}.png"

        # ヒートマップ生成実行
        generator = HeatmapGenerator(grid_search_results)
        generator.generate_heatmap(param_x, param_y, metric, str(output_path),
                                   title=f"Performance Heatmap ({metric}) - Session {session_id}")

        result = {"status": "success", "output_path": str(output_path)}
        try:
            add_history_sync_func(session_id, "visualization_created", {"type": "heatmap", "params": [param_x, param_y], "metric": metric, "path": str(output_path)})
        except Exception as hist_e:
             logger.warning(f"[Job {job_id}] Failed to add heatmap viz history: {hist_e}")

        logger.info(f"[Job {job_id}] Performance heatmap saved to {output_path}")
        return result
    except (ImportError, FileNotFoundError, ValueError, ConfigError) as ie:
        logger.error(f"[Job {job_id}] Error generating heatmap (setup or data issue): {ie}", exc_info=True)
        raise VisualizationError(f"Heatmap generation failed: {ie}") from ie
    except Exception as e:
        logger.error(f"[Job {job_id}] Error generating heatmap: {e}", exc_info=True)
        raise VisualizationError(f"Heatmap generation failed: {e}") from e

def _run_generate_scientific_outputs(job_id: str, config: Dict[str, Any], add_history_sync_func: Callable, session_id: str) -> Dict[str, Any]:
    """科学的成果物生成タスク (同期)"""
    logger.info(f"[Job {job_id}] Generating scientific outputs... Session: {session_id}")
    if not HAS_SCIENCE_OUTPUT:
         raise ImportError("OutputGenerator not available.")

    try:
        # DBから履歴を取得
        db_path = Path(config['paths']['db']) / db_utils.DB_FILENAME
        row = db_utils._db_fetch_one(db_path, "SELECT history, base_algorithm FROM sessions WHERE session_id = ?", (session_id,))
        if not row: raise ValueError(f"Session {session_id} not found.")
        history = json.loads(row["history"] or '[]')
        base_algorithm = row["base_algorithm"]

        if not history:
             raise ValueError("Session history is empty, cannot generate outputs.")

        # 出力ディレクトリ設定
        output_base_dir_str = config.get('paths', {}).get('scientific_output')
        if not output_base_dir_str: raise ConfigError("'scientific_output' path not found.")
        output_base_dir = Path(output_base_dir_str) / session_id
        ensure_dir(output_base_dir)

        # 成果物生成実行
        generator = OutputGenerator(history, base_algorithm, output_base_dir)
        generated_files = generator.generate_all()

        result = {"status": "success", "output_directory": str(output_base_dir), "generated_files": generated_files}
        try:
            add_history_sync_func(session_id, "scientific_outputs_generated", result)
        except Exception as hist_e:
             logger.warning(f"[Job {job_id}] Failed to add scientific output history: {hist_e}")

        logger.info(f"[Job {job_id}] Scientific outputs generated in {output_base_dir}")
        return result
    except (ImportError, FileNotFoundError, ValueError, ConfigError) as ie:
        logger.error(f"[Job {job_id}] Error generating scientific outputs (setup or data issue): {ie}", exc_info=True)
        raise VisualizationError(f"Scientific output generation failed: {ie}") from ie
    except Exception as e:
        logger.error(f"[Job {job_id}] Error generating scientific outputs: {e}", exc_info=True)
        raise VisualizationError(f"Scientific output generation failed: {e}") from e

# --- Tool Registration --- #

def register_visualization_tools(
    mcp: FastMCP,
    config: Dict[str, Any],
    start_async_job_func: Callable[..., str],
    add_history_sync_func: Callable
):
    """可視化関連のMCPツールを登録"""
    logger.info("Registering visualization tools...")

    # Extract necessary paths from config once
    paths_config = config.get('paths', {})
    workspace_dir = Path(paths_config.get('workspace', ''))
    db_path = Path(paths_config.get('db', '')) / db_utils.DB_FILENAME # Assuming db_path in config is the directory
    visualizations_dir = Path(paths_config.get('visualizations', ''))

    # Check paths (optional)
    if not workspace_dir.is_dir(): logger.warning(f"Workspace directory not found: {workspace_dir}")
    if not db_path.parent.is_dir(): logger.warning(f"DB directory not found: {db_path.parent}") # Check parent dir
    if not visualizations_dir: logger.warning("Visualizations directory path is empty in config.")

    # Helper to start async job, passing necessary context
    def _start_viz_job(task_func: Callable, tool_name: str, session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Pass config and sync history adder to the task function
        kwargs['session_id'] = session_id # Ensure session_id is in kwargs for the task

        # Selectively pass config values based on the task
        task_specific_config_args = {}
        if task_func in [_run_create_thumbnail, _run_visualize_spectrogram]:
            task_specific_config_args['workspace_dir'] = workspace_dir
        elif task_func in [_run_visualize_code_impact, _run_generate_heatmap]:
            task_specific_config_args['db_path'] = db_path
            task_specific_config_args['visualizations_dir'] = visualizations_dir
        # Add other tasks here if needed

        job_id = start_async_job_func(
            task_func,
            tool_name,
            session_id, # Pass session_id for job tracking
            # config, # Removed
            add_history_sync_func, # Pass history adder
            **task_specific_config_args, # Pass specific config paths
            **kwargs # Pass original tool arguments
        )
        # Add request history immediately (or consider doing it in the task start)
        if session_id:
            try:
                add_history_sync_func(session_id, f"{tool_name}_request", {"params": kwargs, "job_id": job_id})
            except Exception as e:
                logger.warning(f"Failed to add request history for {tool_name} (Session: {session_id}): {e}")
        return {"job_id": job_id, "status": "pending"}

    # Register tools using the helper
    @mcp.tool("create_thumbnail")
    async def create_thumbnail_tool(image_path: str, size: int = 100, session_id: Optional[str] = None) -> Dict[str, Any]:
        """指定された画像のサムネイルを生成します。"""
        task_kwargs = locals()
        task_kwargs.pop('session_id')
        task_kwargs.pop('self', None)
        return _start_viz_job(_run_create_thumbnail, "create_thumbnail", session_id, **task_kwargs)

    @mcp.tool("visualize_spectrogram")
    async def visualize_spectrogram_tool(audio_file: str, output_format: str = "png", session_id: Optional[str] = None) -> Dict[str, Any]:
        """指定された音声ファイルのスペクトログラム画像を生成します。"""
        task_kwargs = locals()
        task_kwargs.pop('session_id')
        task_kwargs.pop('self', None)
        return _start_viz_job(_run_visualize_spectrogram, "visualize_spectrogram", session_id, **task_kwargs)

    @mcp.tool("visualize_code_impact")
    async def visualize_code_impact_tool(session_id: str, iteration: int) -> Dict[str, Any]:
        """指定されたセッションとイテレーションのコード変更の影響を可視化します。"""
        task_kwargs = locals()
        task_kwargs.pop('session_id')
        task_kwargs.pop('self', None)
        return _start_viz_job(_run_visualize_code_impact, "visualize_code_impact", session_id, **task_kwargs)

    @mcp.tool("generate_heatmap")
    async def generate_heatmap_tool(session_id: str, param_x: str, param_y: str, metric: str) -> Dict[str, Any]:
        """指定されたセッションのグリッドサーチ結果から性能ヒートマップを生成します。"""
        task_kwargs = locals()
        task_kwargs.pop('session_id')
        task_kwargs.pop('self', None)
        return _start_viz_job(_run_generate_heatmap, "generate_heatmap", session_id, **task_kwargs)

    logger.info("Visualization tools registered.")

# --- Import Guard --- #
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 