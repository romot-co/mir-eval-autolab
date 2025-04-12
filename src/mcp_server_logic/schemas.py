from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from src.utils.misc_utils import generate_id

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

    @field_validator('suggested_range', mode="after")
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

class PromptData(BaseModel):
    """プロンプト生成タスクの結果データ"""
    prompt: str = Field(..., description="生成されたプロンプト文字列")

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
        PromptData,
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

class ExplorationStrategyAction(str, Enum):
    """可能な改善戦略アクション"""
    RUN_INITIAL_EVALUATION = "run_initial_evaluation"
    ANALYZE_EVALUATION = "analyze_evaluation"
    GENERATE_HYPOTHESES = "generate_hypotheses"
    IMPROVE_CODE = "improve_code"
    OPTIMIZE_PARAMETERS = "optimize_parameters"
    RUN_EVALUATION = "run_evaluation"
    ASSESS_IMPROVEMENT = "assess_improvement"
    STOP = "stop"
    UNKNOWN = "unknown" # フォールバック/エラー用

class ExplorationStrategySuggestion(BaseModel):
    """LLMによって提案された次の探索戦略"""
    action: ExplorationStrategyAction = Field(..., description="提案された次のアクション")
    reasoning: str = Field(..., description="このアクションが提案された理由の説明")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="提案されたアクションに必要なパラメータ (例: improve_code の suggestion)")
    confidence: Optional[float] = Field(None, description="提案の信頼度スコア (0.0 から 1.0)", ge=0.0, le=1.0)

    @field_validator('action', mode='before')
    def validate_action_enum(cls, value):
        if isinstance(value, str):
            try:
                return ExplorationStrategyAction(value)
            except ValueError:
                try:
                    # 大文字小文字を無視してマッチングを試みる
                    for enum_value in ExplorationStrategyAction:
                        if enum_value.value.lower() == value.lower():
                            return enum_value
                except:
                    pass
                return ExplorationStrategyAction.UNKNOWN
        return value

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

class HypothesisGeneratedData(BaseModel):
    """生成された単一の仮説"""
    id: str = Field(default_factory=lambda: f"hyp_{generate_id()}", description="仮説の一意なID") # utils.generate_id が必要
    description: str = Field(..., description="仮説の内容")
    priority: Optional[float] = Field(None, description="仮説の優先度 (例: 0.0-1.0)", ge=0.0, le=1.0)
    status: str = Field(default="pending", description="仮説の状態 (pending, testing, verified, rejected)")

class HypothesesGenerationStartedData(HistoryEventBaseData):
    job_id: str
    num_hypotheses_requested: int

class HypothesesGenerationCompleteData(HistoryEventBaseData):
    job_id: str
    hypotheses: List[HypothesisGeneratedData]
    prompt_used: Optional[str] = None
    llm_response_raw: Optional[Any] = None

class HypothesesGenerationFailedData(HistoryEventBaseData):
    job_id: str
    error: str
    error_type: str
    prompt_used: Optional[str] = None
    llm_response_raw: Optional[Any] = None

class AssessImprovementInput(BaseModel):
    """'assess_improvement' ツールの入力スキーマ"""
    session_id: str
    original_code: str
    improved_code: str
    evaluation_results_before: Dict[str, Any]
    evaluation_results_after: Dict[str, Any]
    hypothesis_tested: Optional[str] = None
    user_goal: Optional[str] = None
    previous_feedback: Optional[str] = None

class SessionCycleState(BaseModel):
    cycle_count: int = 0
    stagnation_count: int = 0
    last_action: Optional[str] = None
    last_code_version: Optional[str] = None
    last_evaluation: Optional[Dict[str, Any]] = None # 最新の評価結果metrics
    last_analysis: Optional[Dict[str, Any]] = None   # 最新の分析結果 (EvaluationAnalysis.model_dump())
    last_hypotheses: Optional[List[Dict[str, Any]]] = None # 最新の仮説リスト (List[HypothesisGeneratedData.model_dump()])
    last_assessment: Optional[Dict[str, Any]] = None # 最新の改善評価結果 (AssessmentResult.model_dump())
    last_optimized_params: Optional[Dict[str, Any]] = None # 最新の最適化パラメータ
    needs_evaluation: bool = False # コード/パラメータ変更後に評価が必要か
    needs_assessment: bool = False # 評価後に改善評価が必要か

class GetImproveCodePromptInput(BaseModel):
    """コード改善プロンプト生成ツールの入力スキーマ"""
    session_id: str = Field(..., description="プロンプトを生成するセッションID")
    code: str = Field(..., description="改善対象のコード")
    suggestion: str = Field(..., description="改善の提案内容")
    original_code_version_hash: Optional[str] = Field(None, description="元のコードバージョンハッシュ")

class GetSuggestParametersPromptInput(BaseModel):
    """パラメータ提案プロンプト生成ツールの入力スキーマ"""
    session_id: str = Field(..., description="プロンプトを生成するセッションID")
    detector_code: str = Field(..., description="検出器のコード")
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="分析結果")
    current_metrics: Optional[Dict[str, Any]] = Field(None, description="現在のメトリクス")
    cycle_state: Optional[Dict[str, Any]] = Field(None, description="サイクル状態")

class GetAnalyzeEvaluationPromptInput(BaseModel):
    """評価分析プロンプト生成ツールの入力スキーマ"""
    session_id: str = Field(..., description="プロンプトを生成するセッションID")
    evaluation_results: Dict[str, Any] = Field(..., description="評価結果データ")

class GetSuggestExplorationStrategyPromptInput(BaseModel):
    """探索戦略提案プロンプト生成ツールの入力スキーマ"""
    session_id: str = Field(..., description="プロンプトを生成するセッションID")

class GetGenerateHypothesesPromptInput(BaseModel):
    """仮説生成プロンプト生成ツールの入力スキーマ"""
    session_id: str = Field(..., description="プロンプトを生成するセッションID")
    num_hypotheses: int = Field(default=3, gt=0, description="生成する仮説の数")
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="分析結果")
    current_metrics: Optional[Dict[str, Any]] = Field(None, description="現在のメトリクス")

class GetAssessImprovementPromptInput(BaseModel):
    """改善評価プロンプト生成ツールの入力スキーマ"""
    session_id: str = Field(..., description="プロンプトを生成するセッションID")
    original_detector_code: str = Field(..., description="元の検出器コード")
    improved_detector_code: str = Field(..., description="改善された検出器コード")
    evaluation_results_before: Dict[str, Any] = Field(..., description="改善前の評価結果")
    evaluation_results_after: Dict[str, Any] = Field(..., description="改善後の評価結果")
    hypothesis_tested: Optional[str] = Field(None, description="テストされた仮説")
    user_goal: Optional[str] = Field(None, description="ユーザーの目標")
    previous_feedback: Optional[str] = Field(None, description="以前のフィードバック")

# --- プロンプト生成関連の履歴イベントデータスキーマ ---

# --- コード改善プロンプト生成履歴イベント ---
class ImproveCodePromptGenerationStartedData(HistoryEventBaseData):
    """コード改善プロンプト生成開始イベントデータ"""
    job_id: str
    original_code_version_hash: Optional[str] = None
    improvement_suggestion: Optional[str] = None

class ImproveCodePromptGenerationCompleteData(HistoryEventBaseData):
    """コード改善プロンプト生成完了イベントデータ"""
    job_id: str
    prompt: str

class ImproveCodePromptGenerationFailedData(HistoryEventBaseData):
    """コード改善プロンプト生成失敗イベントデータ"""
    job_id: str
    error: str
    error_type: str
    context_used: Dict[str, Any]

# --- パラメータ提案プロンプト生成履歴イベント ---
class ParameterSuggestionPromptGenerationStartedData(HistoryEventBaseData):
    """パラメータ提案プロンプト生成開始イベントデータ"""
    job_id: str

class ParameterSuggestionPromptGenerationCompleteData(HistoryEventBaseData):
    """パラメータ提案プロンプト生成完了イベントデータ"""
    job_id: str
    prompt: str

class ParameterSuggestionPromptGenerationFailedData(HistoryEventBaseData):
    """パラメータ提案プロンプト生成失敗イベントデータ"""
    job_id: str
    error: str
    error_type: str
    context_used: Dict[str, Any]

# --- 評価分析プロンプト生成履歴イベント ---
class AnalyzeEvaluationPromptGenerationStartedData(HistoryEventBaseData):
    """評価分析プロンプト生成開始イベントデータ"""
    job_id: str
    evaluation_results_summary: Optional[str] = None

class AnalyzeEvaluationPromptGenerationCompleteData(HistoryEventBaseData):
    """評価分析プロンプト生成完了イベントデータ"""
    job_id: str
    prompt: str

class AnalyzeEvaluationPromptGenerationFailedData(HistoryEventBaseData):
    """評価分析プロンプト生成失敗イベントデータ"""
    job_id: str
    error: str
    error_type: str
    context_used: Dict[str, Any]

# --- 探索戦略提案プロンプト生成履歴イベント ---
class StrategySuggestionPromptGenerationStartedData(HistoryEventBaseData):
    """探索戦略提案プロンプト生成開始イベントデータ"""
    job_id: str

class StrategySuggestionPromptGenerationCompleteData(HistoryEventBaseData):
    """探索戦略提案プロンプト生成完了イベントデータ"""
    job_id: str
    prompt: str

class StrategySuggestionPromptGenerationFailedData(HistoryEventBaseData):
    """探索戦略提案プロンプト生成失敗イベントデータ"""
    job_id: str
    error: str
    error_type: str
    context_used: Dict[str, Any]

# --- 仮説生成プロンプト生成履歴イベント ---
class HypothesesGenerationPromptGenerationStartedData(HistoryEventBaseData):
    """仮説生成プロンプト生成開始イベントデータ"""
    job_id: str
    num_hypotheses_requested: int

class HypothesesGenerationPromptGenerationCompleteData(HistoryEventBaseData):
    """仮説生成プロンプト生成完了イベントデータ"""
    job_id: str
    prompt: str

class HypothesesGenerationPromptGenerationFailedData(HistoryEventBaseData):
    """仮説生成プロンプト生成失敗イベントデータ"""
    job_id: str
    error: str
    error_type: str
    context_used: Dict[str, Any]

# --- 改善評価プロンプト生成履歴イベント ---
class AssessImprovementPromptGenerationStartedData(HistoryEventBaseData):
    """改善評価プロンプト生成開始イベントデータ"""
    job_id: str
    hypothesis_tested: Optional[str] = None

class AssessImprovementPromptGenerationCompleteData(HistoryEventBaseData):
    """改善評価プロンプト生成完了イベントデータ"""
    job_id: str
    prompt: str

class AssessImprovementPromptGenerationFailedData(HistoryEventBaseData):
    """改善評価プロンプト生成失敗イベントデータ"""
    job_id: str
    error: str
    error_type: str
    context_used: Dict[str, Any]
