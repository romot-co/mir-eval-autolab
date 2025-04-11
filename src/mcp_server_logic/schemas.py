from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from enum import Enum

# --- LLM Tool Specific Schemas ---
# --- Output Schemas (for LLM response validation) ---
# Define these before they are referenced by ResultData schemas

class ParameterInfo(BaseModel):
    name: str = Field(..., description="Name of the parameter")
    type: str = Field(..., description="Data type (e.g., 'float', 'int', 'str', 'bool')")
    current_value: Optional[Any] = Field(None, description="Current value if known")
    suggested_range: Optional[List[Union[float, int]]] = Field(None, description="Suggested range [min, max] for numerical parameters")
    suggested_values: Optional[List[Any]] = Field(None, description="Suggested discrete values for categorical or boolean parameters")
    rationale: Optional[str] = Field(None, description="Reasoning behind the suggestion")

    @validator('suggested_range', always=True)
    def check_range_length(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError('suggested_range must have exactly two elements [min, max]')
        return v

class ParameterSuggestion(BaseModel):
    parameters: List[ParameterInfo] = Field(..., description="List of suggested parameter adjustments")
    overall_rationale: Optional[str] = Field(None, description="Overall reasoning for the parameter suggestions")

class EvaluationFinding(BaseModel):
    metric: str = Field(..., description="The metric being discussed (e.g., 'note.f_measure', 'onset_error_mean')")
    observation: str = Field(..., description="Observation about the metric's value or trend")
    interpretation: Optional[str] = Field(None, description="Interpretation of the observation (e.g., potential cause)")
    strength_or_weakness: str = Field(..., description="Whether this is considered a strength or weakness ('strength', 'weakness', 'neutral')")

class EvaluationAnalysis(BaseModel):
    overall_summary: str = Field(..., description="A brief summary of the overall performance")
    strengths: List[str] = Field(default_factory=list, description="List of identified strengths")
    weaknesses: List[str] = Field(default_factory=list, description="List of identified weaknesses")
    detailed_findings: List[EvaluationFinding] = Field(default_factory=list, description="Detailed observations about specific metrics")
    potential_next_steps: List[str] = Field(default_factory=list, description="Suggested next steps for improvement based on the analysis")

# --- Input Schemas (for tool input validation) ---
# (Keep Input schemas here or move them too if needed, but they likely don't cause this specific NameError)
class ImproveCodeInputSchema(BaseModel):
    session_id: str
    detector_code: str
    evaluation_results: Optional[Dict[str, Any]] = None
    user_goal: Optional[str] = None
    hypotheses: Optional[List[str]] = None
    previous_feedback: Optional[str] = None

class SuggestParametersInputSchema(BaseModel):
    session_id: str
    detector_code: str
    evaluation_results: Optional[Dict[str, Any]] = None
    user_goal: Optional[str] = None

class AnalyzeEvaluationInputSchema(BaseModel):
    session_id: str
    evaluation_results: Dict[str, Any]
    detector_code: Optional[str] = None
    user_goal: Optional[str] = None

# Add input schemas for other LLM tools as needed
# (e.g., GenerateHypothesesInputSchema, SuggestExplorationStrategyInputSchema)

# --- Job Result Specific Schemas ---
# These need to be defined before JobInfo which uses them in Union

class EvaluationResultData(BaseModel):
    """Result schema for a completed 'run_evaluation' job."""
    summary: Optional[Dict[str, Any]] = Field(None, description="Summarized evaluation results")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Detailed metrics dictionary (e.g., from mir_eval)")
    output_dir: Optional[str] = Field(None, description="Path to the directory where results were saved on the server")
    results_json_path: Optional[str] = Field(None, description="Path to the saved JSON results file on the server")
    plot_path: Optional[str] = Field(None, description="Path to the saved plot file on the server")
    code_version: Optional[str] = None
    parameters_used: Optional[Dict[str, Any]] = None

class GridSearchResultData(BaseModel):
    """Result schema for a completed 'run_grid_search' job."""
    summary: Optional[str] = Field(None, description="Summary of the grid search (e.g., best score and params)")
    best_params: Optional[Dict[str, Any]] = Field(None, description="Best parameter set found")
    best_score: Optional[float] = Field(None, description="Score achieved with the best parameters")
    best_params_path: Optional[str] = Field(None, description="Path to the file containing the best parameters on the server")
    results_csv_path: Optional[str] = Field(None, description="Path to the CSV file containing detailed grid search results on the server")
    output_dir: Optional[str] = Field(None, description="Path to the directory where grid search results were saved on the server")
    code_version: Optional[str] = None

class CodeSaveResultData(BaseModel):
    """Result schema for a completed 'save_code' operation."""
    version: str = Field(..., description="The generated version tag/hash for the saved code")
    file_path: str = Field(..., description="The path where the code was saved on the server")
    message: str = "Code saved successfully."

class ImproveCodeResultData(BaseModel):
    """Result schema for a completed 'improve_code' job."""
    code: str = Field(..., description="The improved code content")
    summary: Optional[str] = Field(None, description="Summary of the changes made by the LLM")
    version: Optional[str] = Field(None, description="Version identifier for the new code (e.g., hash or session-based tag)")
    file_path: Optional[str] = Field(None, description="Path where the improved code was saved (if saved automatically)")

class AnalyzeEvaluationResultData(BaseModel):
    """Result schema for a completed 'analyze_evaluation' job."""
    analysis: EvaluationAnalysis # Now defined above

class SuggestParametersResultData(BaseModel):
    """Result schema for a completed 'suggest_parameters' job."""
    suggestion: ParameterSuggestion # Now defined above

class GenerateHypothesesResultData(BaseModel):
    """Result schema for a completed 'generate_hypotheses' job."""
    hypotheses: List[str]

class SuggestStrategyResultData(BaseModel):
    """Result schema for a completed 'suggest_exploration_strategy' job."""
    action: str
    params: Dict[str, Any]
    explanation: Optional[str] = None

class GetCodeResultData(BaseModel):
     """Result schema for a completed 'get_code' job."""
     code: str
     version: Optional[str] = None
     detector_name: Optional[str] = None
     file_path: Optional[str] = None

# --- Base Schemas (JobInfo now comes after its result types and their dependencies) ---

class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"

class ErrorDetails(BaseModel):
    message: str = Field(..., description="エラーメッセージ")
    type: str = Field(default="UnknownError", description="エラーの型名 (例: ValueError, LLMError)")
    traceback: Optional[str] = Field(None, description="エラー発生時のスタックトレース (存在する場合)")

class JobInfo(BaseModel):
    job_id: str
    session_id: Optional[str] = None
    tool_name: Optional[str] = None
    status: JobStatus
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Union[
        EvaluationResultData,
        GridSearchResultData,
        CodeSaveResultData,
        ImproveCodeResultData,
        AnalyzeEvaluationResultData,
        SuggestParametersResultData,
        GenerateHypothesesResultData,
        SuggestStrategyResultData,
        GetCodeResultData,
        Dict[str, Any] # Fallback for unknown/simple results
    ]] = Field(None, description="Job result (decoded JSON), specific schema depends on tool_name")
    error_details: Optional[ErrorDetails] = Field(None, description="Job failure details")
    task_args: Optional[str] = Field(None, description="Job arguments (JSON string)")
    worker_id: Optional[str] = None
    created_at: Optional[float] = None

class JobStartResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    message: Optional[str] = "Job successfully queued."

class SessionInfoResponse(BaseModel):
    session_id: str
    base_algorithm: Optional[str] = None
    dataset_name: Optional[str] = None
    improvement_goal: Optional[str] = None
    current_cycle: int
    status: str
    best_code_version_id: Optional[str] = None
    best_metrics: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str
    updated_at: str
    potential_next_steps: List[str] = Field(default_factory=list, description="Suggested next steps for improvement based on the analysis")
    last_error: Optional[ErrorDetails] = Field(None, description="The error details from the last failed job associated with this session, if any.")

class HypothesisList(BaseModel):
    hypotheses: List[str] = Field(..., description="List of generated improvement hypotheses")

class ExplorationStrategySuggestion(BaseModel):
    action: str = Field(..., description="Suggested next action (e.g., 'improve_code', 'run_grid_search')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the suggested action")
    explanation: Optional[str] = Field(None, description="Reasoning behind the suggestion")

# --- Job Result Schemas ---

class CodeContent(BaseModel):
    """Output schema for get_code tool."""
    code: str
    version: Optional[str] = None
    detector_name: Optional[str] = None
    file_path: Optional[str] = None

class SuggestionResult(BaseModel):
    """Generic output schema for suggestion tools."""
    strategy: Dict[str, Any]
    status: str = "success"
    error: Optional[str] = None

# --- Session History Event Data Schemas ---
# Schemas for the 'data' field in session history events

class HistoryEventBaseData(BaseModel):
    """Base model for common fields in history event data."""
    # ここに共通フィールドを追加できます (例: timestamp)
    pass # NameError解消のためにクラスを定義

class CycleCompletedData(HistoryEventBaseData):
    """Data for 'cycle_completed' event."""
    cycle_number: int
    strategy_used: Dict[str, Any]
    action_taken: Optional[str] = None
    evaluation_job_id: Optional[str] = None
    evaluation_summary: Optional[str] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None
    code_version_evaluated: Optional[str] = None

class CycleFailedData(HistoryEventBaseData):
    """Data for 'cycle_failed' event."""
    cycle_number: int
    strategy_used: Optional[Dict[str, Any]] = None
    action_taken: Optional[str] = None
    evaluation_job_id: Optional[str] = None
    evaluation_error: Optional[Union[str, Dict]] = None
    code_version_evaluated: Optional[str] = None

class StoppedDueToStagnationData(BaseModel):
    """Data for 'stopped_due_to_stagnation' event."""
    stagnation_count: int
    max_stagnation: int

class StoppedByStrategyData(BaseModel):
    """Data for 'stopped_by_strategy' event."""
    strategy_details: Optional[Dict[str, Any]] = None

class CycleErrorData(BaseModel):
    """Data for 'cycle_error' event."""
    cycle_number: Optional[int] = None
    error: str
    traceback: Optional[str] = None

class AnalysisFailedData(HistoryEventBaseData):
    """Data for 'analysis_failed' event."""
    error: Optional[Union[str, Dict]] = None

class SuggestionFailedData(BaseModel):
    """Data for 'suggestion_failed' event."""
    error: Optional[Union[str, Dict]] = None

class CodeImprovementFailedData(HistoryEventBaseData):
    """Data for 'code_improvement_failed' event."""
    error: Optional[Union[str, Dict]] = None

class CodeSaveFailedData(BaseModel):
    """Data for 'code_save_failed' event."""
    error: str = Field(..., description="エラーメッセージ")
    error_type: Optional[str] = Field(None, description="エラーの型名")
    detector_name: Optional[str] = None
    file_path_attempted: Optional[str] = None

# Event Data Union (for potential future use, might be complex)
# HistoryEventData = Union[
#     EvaluationCompleteData, ParameterOptimizationStartedData, ..., CycleCompletedData, CycleFailedData, ...
# ]

# --- History Event Schemas (Continued) ---

class EvaluationFailedData(HistoryEventBaseData):
    """Data for 'evaluation_failed' event."""
    error: str = Field(..., description="エラーメッセージ")
    error_type: Optional[str] = Field(None, description="エラーの型名")
    traceback: Optional[str] = Field(None, description="スタックトレース")
    detector_name: Optional[str] = None # 失敗した検出器名
    dataset_name: Optional[str] = None # 使用したデータセット名
    parameters_used: Optional[Dict[str, Any]] = None # 使用したパラメータ

class GridSearchFailedData(HistoryEventBaseData):
    """Data for 'grid_search_failed' event."""
    error: str = Field(..., description="エラーメッセージ")
    error_type: Optional[str] = Field(None, description="エラーの型名")
    traceback: Optional[str] = Field(None, description="スタックトレース")
    detector_name: Optional[str] = None # 対象の検出器名
    config_used: Optional[Dict[str, Any]] = None # 使用したグリッド設定

class CodeLoadFailedData(HistoryEventBaseData):
    """Data for 'code_load_failed' event."""
    error: str = Field(..., description="エラーメッセージ")
    error_type: Optional[str] = Field(None, description="エラーの型名")
    detector_name: Optional[str] = None
    version: Optional[str] = None

class SessionErrorData(HistoryEventBaseData):
    """Data for 'session_error' event."""
    error: str = Field(..., description="セッションレベルのエラーメッセージ")
    error_type: Optional[str] = Field(None, description="エラーの型名")
    triggering_event: Optional[str] = Field(None, description="エラーを引き起こした可能性のあるイベントタイプ")

class AnalyzeEvaluationFailedData(HistoryEventBaseData):
    """Data for 'analyze_evaluation_failed' event."""
    error: str = Field(..., description="エラーメッセージ")
    error_type: Optional[str] = Field(None, description="エラーの型名")
    traceback: Optional[str] = Field(None, description="スタックトレース")
    prompt_used: Optional[str] = None
    llm_response_raw: Optional[Any] = None

class SessionStartedData(HistoryEventBaseData):
    """Data for 'session_started' event."""
    base_algorithm: str
    initial_config: Dict[str, Any]

class SessionResumedData(HistoryEventBaseData):
    """Data for 'session_resumed' event."""
    pass # No extra data needed?

class SessionPausedData(HistoryEventBaseData):
    """Data for 'session_paused' event."""
    reason: Optional[str] = None

class SessionStoppedData(HistoryEventBaseData):
    """Data for 'session_stopped' event."""
    reason: str

class SessionCompletedData(HistoryEventBaseData):
    """Data for 'session_completed' event."""
    final_best_metrics: Optional[Dict[str, Any]] = None
    final_best_code_version: Optional[str] = None

class SessionTimeoutData(HistoryEventBaseData):
    """Data for 'session_timeout' event."""
    timeout_seconds: int

class CycleStartedData(HistoryEventBaseData):
    """Data for 'cycle_started' event."""
    cycle_number: int
    strategy_action: Optional[str] = None
    hypothesis: Optional[str] = None

class StoppedByStrategyData(BaseModel):
    """Data for 'stopped_by_strategy' event."""
    strategy_details: Optional[Dict[str, Any]] = None

# ... (Other *FailedData schemas) ...

class CodeSaveFailedData(BaseModel):
    """Data for 'code_save_failed' event."""
    error: str = Field(..., description="エラーメッセージ")
    error_type: Optional[str] = Field(None, description="エラーの型名")
    detector_name: Optional[str] = None
    file_path_attempted: Optional[str] = None

# --- Tool Input Schemas (Used with @mcp.tool(input_schema=...)) ---

class GetCodeInput(BaseModel):
    detector_name: str
    version: Optional[str] = None
    session_id: Optional[str] = None

class SaveCodeInput(BaseModel):
    detector_name: str
    code: str
    session_id: Optional[str] = None
    parent_version: Optional[str] = None
    changes_summary: Optional[str] = None

class RunEvaluationInput(BaseModel):
    detector_name: str
    dataset_name: Optional[str] = None
    code_version: Optional[str] = None
    detector_params: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    save_plots: bool = Field(default=False, description="Whether to save plots on the server")
    save_results_json: bool = Field(default=True, description="Whether to save results JSON on the server")

class RunGridSearchInput(BaseModel):
    grid_config: Dict[str, Any] = Field(..., description="Grid search configuration dictionary")
    skip_existing: bool = Field(default=False, description="Whether to skip existing results on the server")
    best_metric: Optional[str] = Field(default="note.f_measure", description="Metric to optimize")
    code_version: Optional[str] = None
    session_id: Optional[str] = None

class AnalyzeEvaluationInput(BaseModel):
    session_id: str
    evaluation_results: Dict[str, Any]

class SuggestExplorationStrategyInput(BaseModel):
    session_id: str = Field(..., description="戦略を提案するセッションID")

class ImproveCodeInput(BaseModel):
    session_id: str
    code: str
    suggestion: str

class GenerateHypothesesInput(BaseModel):
    session_id: str
    num_hypotheses: int = Field(default=3, gt=0)
    analysis_results: Optional[Dict[str, Any]] = None
    current_metrics: Optional[Dict[str, Any]] = None

class VisualizeSessionInput(BaseModel):
    session_id: str
    output_dir: Optional[str] = None

# --- Session Management Tools --- #
class StartSessionInput(BaseModel):
    base_algorithm: str = Field(default="Unknown", description="ベースとなるアルゴリズム名")

class GetSessionInfoInput(BaseModel):
    session_id: str = Field(..., description="情報を取得するセッションID")

class AddSessionHistoryInput(BaseModel):
    """Input schema for the add_session_history tool."""
    session_id: str = Field(..., description="履歴を追加するセッションID")
    event_type: str = Field(..., description="履歴イベントのタイプ")
    data: Dict[str, Any] = Field(..., description="イベント固有のデータ")
    cycle_state_update: Optional[Dict[str, Any]] = Field(None, description="Cycle state を更新するフィールドの辞書")

# --- Job Management Tools --- #
class GetJobStatusInput(BaseModel):
    job_id: str = Field(..., description="ステータスを取得するジョブID")
# (Add ListJobsInput etc. if needed)

# --- Code Tools --- #
class GetCodeInput(BaseModel):
    detector_name: str

# --- Improvement Loop Tools --- #
class OptimizeParametersInput(BaseModel):
    detector_name: str = Field(..., description="最適化対象の検出器名")
    code_version: Optional[str] = Field(None, description="最適化対象のコードバージョン (指定なければ最新)")
    audio_dir: Optional[str] = Field(None, description="評価に使用する音声ファイルディレクトリ (データセット指定のため)")
    reference_dir: Optional[str] = Field(None, description="評価に使用する参照ラベルディレクトリ (データセット指定のため)")
    session_id: Optional[str] = Field(None, description="関連する改善セッションID")