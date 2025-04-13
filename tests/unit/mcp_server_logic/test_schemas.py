# tests/unit/mcp_server_logic/test_schemas.py
import pytest
from pydantic import ValidationError, BaseModel
from typing import List, Dict, Optional, Any, Union
from enum import Enum

# Assume schemas are defined in src.mcp_server_logic.schemas
# Provide dummies if the file or specific schemas don't exist yet
try:
    from src.mcp_server_logic.schemas import (
        JobStatus,
        JobInfo,
        SessionStatus,
        SessionHistoryEvent,
        EvaluationMetrics,  # Assuming this structure exists for best_metrics
        SessionInfoResponse,
        StartSessionInput,  # Example tool input
        GetJobStatusInput,  # Example tool input
        EvaluateDetectorInput,  # Example tool input
        EvaluateDetectorOutput,  # Example tool output
        # Add other key schemas mentioned or implied:
        GetCodeInput,
        GetCodeOutput,
        SaveCodeInput,
        SaveCodeOutput,
        ProposeStrategyInput,
        ProposeStrategyOutput,
        RunGridSearchInput,
        RunGridSearchOutput,
        # Base models if they exist
        BaseToolInput,
        BaseToolOutput,
        ErrorDetail,
    )
# Broad exception for initial setup, refine if possible
except ImportError:
    print("Warning: Using dummy implementations for mcp_server_logic.schemas.")

    class JobStatus(str, Enum):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"

    class SessionStatus(str, Enum):
        RUNNING = "running"
        STOPPED = "stopped"
        STALLED = "stalled"

    class BaseModelAlias(BaseModel):  # Basic Pydantic functionality
        class Config:
            allow_population_by_field_name = True
            orm_mode = True  # If used with ORM

    # --- Dummy Schemas ---
    class ErrorDetail(BaseModelAlias):
        type: str
        message: str
        details: Optional[str] = None

    class BaseToolInput(BaseModelAlias):
        pass

    class BaseToolOutput(BaseModelAlias):
        job_id: Optional[str] = None
        error: Optional[ErrorDetail] = None

    class JobInfo(BaseModelAlias):
        job_id: str
        tool_name: str
        status: JobStatus
        created_at: str  # Assuming string timestamp for simplicity
        updated_at: str
        result: Optional[Dict[str, Any]] = None
        error: Optional[ErrorDetail] = None
        input_params: Optional[Dict[str, Any]] = None

    class SessionHistoryEvent(BaseModelAlias):
        event_id: str
        session_id: str
        timestamp: str
        event_type: str  # e.g., 'session_start', 'tool_run', 'evaluation_complete'
        details: Dict[str, Any]

    # Dummy for complex nested structure
    class EvaluationMetrics(BaseModelAlias):
        overall_accuracy: Optional[float] = None
        f_measure: Optional[float] = None

    class SessionInfoResponse(BaseModelAlias):
        session_id: str
        status: SessionStatus
        created_at: str
        last_update: str
        best_metrics: Optional[EvaluationMetrics] = None
        history: List[SessionHistoryEvent] = []
        stagnation_count: int = 0

    class StartSessionInput(BaseToolInput):
        initial_prompt: Optional[str] = None

    class GetJobStatusInput(BaseToolInput):
        job_id: str

    class EvaluateDetectorInput(BaseToolInput):
        detector_name: str
        dataset_name: str
        parameters: Optional[Dict[str, Any]] = None

    class EvaluateDetectorOutput(BaseToolOutput):
        metrics: Optional[Dict[str, Any]] = None
        plot_path: Optional[str] = None

    class GetCodeInput(BaseToolInput):
        detector_name: str
        version: Optional[str] = "latest"  # Example default

    class GetCodeOutput(BaseToolOutput):
        code: Optional[str] = None
        version: Optional[str] = None

    class SaveCodeInput(BaseToolInput):
        detector_name: str
        code: str
        commit_message: Optional[str] = None

    class SaveCodeOutput(BaseToolOutput):
        version: Optional[str] = None
        message: Optional[str] = None

    class ProposeStrategyInput(BaseToolInput):
        current_metrics: EvaluationMetrics  # Use the dummy
        session_history: List[SessionHistoryEvent]  # Use the dummy

    class ProposeStrategyOutput(BaseToolOutput):
        proposed_strategy: Optional[str] = None
        reasoning: Optional[str] = None

    class RunGridSearchInput(BaseToolInput):
        detector_name: str
        dataset_name: str
        param_grid: Dict[str, List[Any]]

    class RunGridSearchOutput(BaseToolOutput):
        best_params: Optional[Dict[str, Any]] = None
        best_score: Optional[float] = None
        results_path: Optional[str] = None


# --- Test Fixtures ---
@pytest.fixture
def sample_job_info_data():
    return {
        "job_id": "job_123",
        "tool_name": "evaluate_detector",
        "status": JobStatus.COMPLETED,
        "created_at": "2023-10-27T10:00:00Z",
        "updated_at": "2023-10-27T10:05:00Z",
        "result": {"accuracy": 0.95},
        "input_params": {"detector_name": "TestDetector"},
    }


@pytest.fixture
def sample_session_info_data():
    history_event = {
        "event_id": "evt_abc",
        "session_id": "sess_xyz",
        "timestamp": "2023-10-27T09:00:00Z",
        "event_type": "session_start",
        "details": {"user": "test"},
    }
    best_metrics_data = {"f_measure": 0.88}
    return {
        "session_id": "sess_xyz",
        "status": SessionStatus.RUNNING,
        "created_at": "2023-10-27T09:00:00Z",
        "last_update": "2023-10-27T09:05:00Z",
        "best_metrics": best_metrics_data,
        "history": [history_event],
        "stagnation_count": 1,
    }


# --- Basic Validation Tests ---


def test_job_info_valid(sample_job_info_data):
    """Test successful JobInfo instantiation."""
    job_info = JobInfo(**sample_job_info_data)
    assert job_info.job_id == sample_job_info_data["job_id"]
    assert job_info.status == JobStatus.COMPLETED
    assert job_info.result == {"accuracy": 0.95}
    assert job_info.error is None


def test_job_info_missing_required():
    """Test JobInfo validation error on missing required fields."""
    with pytest.raises(ValidationError):
        JobInfo(
            job_id="job_123", tool_name="test"
        )  # Missing status, created_at, updated_at


def test_job_info_invalid_status():
    """Test JobInfo validation error on invalid status enum."""
    invalid_data = {
        "job_id": "job_123",
        "tool_name": "test",
        "status": "finished",  # Invalid status
        "created_at": "ts",
        "updated_at": "ts",
    }
    with pytest.raises(ValidationError):
        JobInfo(**invalid_data)


def test_job_info_optional_fields():
    """Test JobInfo with optional fields being None."""
    minimal_data = {
        "job_id": "job_min",
        "tool_name": "minimal",
        "status": JobStatus.PENDING,
        "created_at": "ts1",
        "updated_at": "ts2",
    }
    job_info = JobInfo(**minimal_data)
    assert job_info.result is None
    assert job_info.error is None
    assert job_info.input_params is None


def test_session_info_response_valid(sample_session_info_data):
    """Test successful SessionInfoResponse instantiation."""
    session_info = SessionInfoResponse(**sample_session_info_data)
    assert session_info.session_id == sample_session_info_data["session_id"]
    assert session_info.status == SessionStatus.RUNNING
    assert len(session_info.history) == 1
    assert isinstance(session_info.history[0], SessionHistoryEvent)
    # Check nested model instantiation
    assert isinstance(session_info.best_metrics, EvaluationMetrics)
    assert session_info.best_metrics.f_measure == 0.88
    assert session_info.stagnation_count == 1


def test_session_info_response_missing_required():
    """Test SessionInfoResponse validation error on missing fields."""
    with pytest.raises(ValidationError):
        # Missing status, created_at, last_update etc.
        SessionInfoResponse(session_id="sess_abc")


def test_session_info_response_invalid_status(sample_session_info_data):
    """Test SessionInfoResponse validation error on invalid status."""
    invalid_data = sample_session_info_data.copy()
    invalid_data["status"] = "active"  # Invalid status
    with pytest.raises(ValidationError):
        SessionInfoResponse(**invalid_data)


def test_session_info_response_empty_history_and_metrics():
    """Test SessionInfoResponse with empty history and no best metrics."""
    minimal_data = {
        "session_id": "sess_min",
        "status": SessionStatus.STOPPED,
        "created_at": "ts1",
        "last_update": "ts2",
        # best_metrics is Optional, defaults to None
        # history defaults to []
        # stagnation_count defaults to 0
    }
    session_info = SessionInfoResponse(**minimal_data)
    assert session_info.best_metrics is None
    assert session_info.history == []
    assert session_info.stagnation_count == 0


# --- Tool Specific Schema Tests (Examples) ---


def test_evaluate_detector_input_valid():
    """Test valid EvaluateDetectorInput."""
    data = {
        "detector_name": "MyDet",
        "dataset_name": "MyData",
        "parameters": {"sens": 0.5},
    }
    inp = EvaluateDetectorInput(**data)
    assert inp.detector_name == "MyDet"
    assert inp.parameters == {"sens": 0.5}


def test_evaluate_detector_input_missing_required():
    """Test EvaluateDetectorInput missing required fields."""
    with pytest.raises(ValidationError):
        EvaluateDetectorInput(detector_name="MyDet")  # Missing dataset_name


def test_get_code_output_valid():
    """Test valid GetCodeOutput."""
    data = {"code": "def detect(): pass", "version": "v1.0.1", "job_id": "job_get"}
    out = GetCodeOutput(**data)
    assert out.code == "def detect(): pass"
    assert out.version == "v1.0.1"
    assert out.job_id == "job_get"
    assert out.error is None


def test_get_code_output_error():
    """Test GetCodeOutput representing an error."""
    error_detail = {"type": "FileNotFound", "message": "Detector code not found"}
    data = {"error": error_detail, "job_id": "job_get_err"}
    out = GetCodeOutput(**data)
    assert out.code is None
    assert out.version is None
    assert isinstance(out.error, ErrorDetail)
    assert out.error.type == "FileNotFound"


def test_run_grid_search_input_valid():
    """Test valid RunGridSearchInput."""
    data = {
        "detector_name": "GSDet",
        "dataset_name": "GSData",
        "param_grid": {"threshold": [0.3, 0.5, 0.7], "window": [1024]},
    }
    inp = RunGridSearchInput(**data)
    assert inp.detector_name == "GSDet"
    assert inp.param_grid["threshold"] == [0.3, 0.5, 0.7]


def test_run_grid_search_input_invalid_grid():
    """Test RunGridSearchInput with invalid param_grid type."""
    with pytest.raises(ValidationError):
        RunGridSearchInput(
            detector_name="GSDet", dataset_name="GSData", param_grid="invalid"
        )


# Add more tests for other key schemas (SaveCodeInput/Output, ProposeStrategyInput/Output etc.)
# following the same pattern: test valid cases, missing required fields, invalid types.
