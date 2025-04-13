from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Body
from pathlib import Path
from src.mcp_server_logic.job_manager import JobManager
from src.utils.logger import logger

app = APIRouter()
job_manager = JobManager()

@app.post("/api/jobs/list")
async def list_jobs_endpoint(
    request_data: Dict[str, Any] = Body(...),
    db_config: Dict[str, Any] = Depends(get_db_config)
) -> Dict[str, Any]:
    """ジョブの一覧を取得するエンドポイント
    
    Args:
        request_data: リクエストデータ（session_id, status, limitを含む辞書）
        db_config: データベース設定（依存性注入）
        
    Returns:
        ジョブ情報のリストを含む辞書
    """
    try:
        session_id = request_data.get("session_id")
        status = request_data.get("status")
        limit = request_data.get("limit", 50)
        
        db_path = Path(db_config["paths"]["db"])
        jobs = await job_manager.list_jobs(db_path, session_id=session_id, status=status, limit=limit)
        
        # JobInfoオブジェクトを辞書に変換
        job_dicts = [job.model_dump() for job in jobs]
        
        return {
            "status": "success",
            "jobs": job_dicts,
            "count": len(job_dicts)
        }
    except Exception as e:
        logger.error(f"Error listing jobs: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to list jobs: {str(e)}"
        }

@app.get("/api/jobs/running")
async def get_running_jobs_endpoint(
    session_id: Optional[str] = None,
    limit: int = 20,
    db_config: Dict[str, Any] = Depends(get_db_config)
) -> Dict[str, Any]:
    """実行中のジョブの一覧を取得するエンドポイント
    
    Args:
        session_id: フィルタリングするセッションID（オプション）
        limit: 最大取得件数（デフォルト: 20）
        db_config: データベース設定（依存性注入）
        
    Returns:
        実行中のジョブ情報のリストを含む辞書
    """
    try:
        db_path = Path(db_config["paths"]["db"])
        jobs = await job_manager.list_jobs(db_path, session_id=session_id, status="RUNNING", limit=limit)
        
        # JobInfoオブジェクトを辞書に変換
        job_dicts = [job.model_dump() for job in jobs]
        
        return {
            "status": "success",
            "running_jobs": job_dicts,
            "count": len(job_dicts)
        }
    except Exception as e:
        logger.error(f"Error getting running jobs: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to get running jobs: {str(e)}"
        }

@app.get("/api/system/worker-status")
async def get_worker_status_endpoint(
    stalled_threshold_seconds: int = 3600
) -> Dict[str, Any]:
    """ジョブワーカーの状態情報を取得するエンドポイント
    
    Args:
        stalled_threshold_seconds: この秒数以上実行されているジョブを停滞中と見なす（デフォルト: 3600秒）
        
    Returns:
        ワーカーの状態情報を含む辞書
    """
    try:
        status = await job_manager.get_worker_status(stalled_threshold_seconds)
        return {
            "status": "success",
            "worker_status": status
        }
    except Exception as e:
        logger.error(f"Error getting worker status: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to get worker status: {str(e)}"
        }

async def startup_event():
    """サーバー開始時のイベント"""
    logger.info("Starting up MCP server")
    
    try:
        # ジョブクリーンアップタスク
        cleanup_jobs_func = job_manager.cleanup_old_jobs
        await start_cleanup_task(config, cleanup_jobs_func)
        
        # 古いセッションのクリーンアップ
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True) 