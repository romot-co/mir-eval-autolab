import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine, List, Awaitable, Union, Type
import time
import asyncio
import sqlite3
import aiosqlite # aiosqlite をインポート

# 関連モジュールインポート (型ヒント用)
from mcp.server.fastmcp import FastMCP
from . import db_utils
from src.utils.misc_utils import generate_id, get_timestamp, format_timestamp # format_timestampを追加
from src.utils.exception_utils import StateManagementError, log_exception, FileError, ConfigError
from src.utils.json_utils import NumpyEncoder # NumpyEncoder をインポート
from src.utils.path_utils import validate_path_within_allowed_dirs, get_workspace_dir # パス検証用に追加
from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError
from . import schemas # スキーマをインポート

logger = logging.getLogger('mcp_server.session_manager')

# --- 定数定義 --- #
DEFAULT_IMPROVEMENT_THRESHOLD = 0.05 # 改善とみなす閾値 (F値など)

# --- MCP Tools --- #

async def start_session(config: Dict[str, Any], db_path: Path, base_algorithm: str = "Unknown") -> schemas.SessionInfoResponse:
    """改善セッションを開始します。"""
    # DBパスの検証
    try:
        # ワークスペースディレクトリを取得
        workspace_dir = get_workspace_dir(config)
        # DBパスが許可されたディレクトリ内にあることを検証
        validated_db_path = validate_path_within_allowed_dirs(
            db_path,
            allowed_base_dirs=[workspace_dir],
            check_existence=False,  # DBファイルが存在しない場合もある（初回実行時など）
            check_is_file=None,     # 存在チェックを行わないため不要
            allow_absolute=True     # 絶対パスを許可
        )
        # 検証済みパスを使用
        db_path = validated_db_path
    except (ConfigError, FileError, ValueError) as e:
        logger.error(f"Failed to validate DB path for starting session: {e}")
        raise StateManagementError(f"Failed to validate DB path: {e}") from e
    
    session_id = generate_id()
    current_time = get_timestamp()
    initial_cycle_state_model = schemas.SessionCycleState()
    initial_cycle_state_json = initial_cycle_state_model.model_dump_json()
    initial_config_json = json.dumps(config, cls=NumpyEncoder)

    try:
        await db_utils.db_execute_commit_async(
            db_path,
            "INSERT INTO sessions (session_id, base_algorithm, start_time, last_update, status, history, config, cycle_state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, base_algorithm, current_time, current_time, "active", "[]", initial_config_json, initial_cycle_state_json)
        )
        logger.info(f"新しいセッションが作成されました: {session_id}")
        return schemas.SessionInfoResponse(
            session_id=session_id,
            base_algorithm=base_algorithm,
            created_at=format_timestamp(current_time),
            updated_at=format_timestamp(current_time),
            status="active",
            history=[],
            config=config,
            cycle_state=initial_cycle_state_model.model_dump(),
            dataset_name=None,
            improvement_goal=None,
            current_cycle=initial_cycle_state_model.cycle_count,
            best_code_version_id=None,
            best_metrics=None,
            potential_next_steps=[]
        )
    except StateManagementError as sme:
        logger.error(f"セッション開始 DBエラー: {sme}", exc_info=False)
        raise sme
    except Exception as e:
        logger.error(f"セッション開始中に予期せぬエラー: {e}", exc_info=True)
        raise StateManagementError(f"Unexpected error starting session: {e}") from e

async def get_session_info(config: Dict[str, Any], db_path: Path, session_id: str) -> schemas.SessionInfoResponse:
    """セッション情報を取得し、SessionInfoResponse スキーマで返します。"""
    # DBパスの検証
    try:
        # ワークスペースディレクトリを取得
        workspace_dir = get_workspace_dir(config)
        # DBパスが許可されたディレクトリ内にあることを検証
        validated_db_path = validate_path_within_allowed_dirs(
            db_path,
            allowed_base_dirs=[workspace_dir],
            check_existence=True,  # DBファイルが存在することを確認
            check_is_file=True,    # ファイルであることを確認
            allow_absolute=True    # 絶対パスを許可
        )
        # 検証済みパスを使用
        db_path = validated_db_path
    except (ConfigError, FileError, ValueError) as e:
        logger.error(f"Failed to validate DB path for session info retrieval: {e}")
        raise StateManagementError(f"Failed to validate DB path: {e}") from e
    
    row = None
    try:
        sql = """SELECT session_id, base_algorithm, start_time, last_update, status,
                      history, config, cycle_state, current_metrics, best_metrics,
                      best_code_version, baseline_metrics
               FROM sessions WHERE session_id = ?"""
        row = await db_utils.db_fetch_one_async(db_path, sql, (session_id,))
    except StateManagementError as sme:
         logger.error(f"セッション情報取得 DBエラー ({session_id}): {sme}", exc_info=False)
         raise sme
    except Exception as e:
         logger.error(f"セッション情報取得中に予期せぬエラー ({session_id}): {e}", exc_info=True)
         raise StateManagementError(f"Unexpected error fetching session info: {e}") from e

    if row:
        try:
            session_data = dict(row)
            parsed_data = {}
            parse_errors = []

            parsed_data['session_id'] = session_data.get('session_id')
            parsed_data['base_algorithm'] = session_data.get('base_algorithm')
            parsed_data['status'] = session_data.get('status', 'unknown')
            parsed_data['best_code_version_id'] = session_data.get('best_code_version')

            start_time = session_data.get('start_time')
            last_update = session_data.get('last_update')
            parsed_data['created_at'] = format_timestamp(start_time) if start_time else "N/A"
            parsed_data['updated_at'] = format_timestamp(last_update) if last_update else "N/A"

            for key in ['history', 'config', 'cycle_state', 'current_metrics', 'best_metrics', 'baseline_metrics']:
                json_str = session_data.get(key)
                default_val = [] if key == 'history' else {}
                if json_str:
                    try:
                        parsed_data[key] = json.loads(json_str)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"セッション {session_id} の {key} JSON解析失敗: {e}。デフォルト値を使用します。")
                        parse_errors.append(f"Failed to parse {key}")
                        parsed_data[key] = default_val
                else:
                    parsed_data[key] = default_val

            parsed_data['current_cycle'] = parsed_data.get('cycle_state', {}).get('cycle_count', 0)
            parsed_data['potential_next_steps'] = []
            parsed_data['dataset_name'] = None
            parsed_data['improvement_goal'] = None

            # --- 追加: 最新の失敗ジョブのエラー情報を取得 --- #
            last_error_details: Optional[schemas.ErrorDetails] = None
            try:
                error_sql = "SELECT error_details FROM jobs WHERE session_id = ? AND status = ? ORDER BY end_time DESC LIMIT 1"
                error_row = await db_utils.db_fetch_one_async(db_path, error_sql, (session_id, schemas.JobStatus.FAILED.value))
                if error_row and error_row['error_details']:
                    try:
                        error_dict = json.loads(error_row['error_details'])
                        last_error_details = schemas.ErrorDetails(**error_dict)
                    except (json.JSONDecodeError, TypeError, ValidationError) as err_parse:
                        logger.warning(f"Failed to parse last error details for session {session_id}: {err_parse}")
            except Exception as db_err:
                 logger.error(f"Failed to fetch last error for session {session_id}: {db_err}", exc_info=True)
            parsed_data['last_error'] = last_error_details
            # --- 追加ここまで --- #

            if parse_errors:
                 logger.warning(f"セッション {session_id} のデータパース中にエラーが発生しました: {parse_errors}")

            session_info_model = schemas.SessionInfoResponse.model_validate(parsed_data)
            return session_info_model

        except Exception as e:
            logger.error(f"セッション情報処理/パース中に予期せぬエラー ({session_id}): {e}", exc_info=True)
            raise StateManagementError(f"Unexpected error processing/parsing session info: {e}") from e
    else:
        raise ValueError(f"セッション {session_id} が見つかりません")

# _add_session_history_sync 内のエラーハンドリングは db_utils.py で StateManagementError が
# raise されるようになったため、ここで再度ラップする必要性は低いが、
# ValueError (Session not found) は区別して処理する。

# --- 修正: add_session_history を非同期化し、スキーマ検証とシリアライズを追加 --- #
async def add_session_history(
    config: Dict[str, Any],
    db_path: Path,
    session_id: str,
    event_type: str,
    event_data: Dict[str, Any],
    cycle_state_update: Optional[Dict[str, Any]] = None
) -> None:
    """セッション履歴にイベントを追加し、cycle_state をアトミックに更新します (非同期、スキーマ検証あり)

    Args:
        config: Server configuration.
        db_path: Path to the SQLite database.
        session_id: The ID of the session to update.
        event_type: The type of the event (e.g., 'evaluation_complete', 'cycle_failed').
        event_data: A dictionary containing data specific to the event_type.
                    This data MUST conform to the corresponding Pydantic schema in schemas.py.
        cycle_state_update: An optional dictionary with fields to update in the session's cycle_state.
                            Keys should be valid fields of schemas.SessionCycleState.

    Raises:
        ValueError: If the session is not found or if event_data/cycle_state_update validation fails.
        StateManagementError: If a database error occurs.
    """
    # DBパスの検証
    try:
        # ワークスペースディレクトリを取得
        workspace_dir = get_workspace_dir(config)
        # DBパスが許可されたディレクトリ内にあることを検証
        validated_db_path = validate_path_within_allowed_dirs(
            db_path,
            allowed_base_dirs=[workspace_dir],
            check_existence=True,  # DBファイルが存在することを確認
            check_is_file=True,    # ファイルであることを確認
            allow_absolute=True    # 絶対パスを許可
        )
        # 検証済みパスを使用
        db_path = validated_db_path
    except (ConfigError, FileError, ValueError) as e:
        logger.error(f"Failed to validate DB path for session history: {e}")
        raise StateManagementError(f"Failed to validate DB path: {e}") from e
    
    current_timestamp = get_timestamp()
    validated_event_data_dict: Dict[str, Any]
    validated_update_dict: Optional[Dict[str, Any]] = None

    # --- 1. スキーマ検証とデータの準備 (ロック外) --- #
    try:
        # event_type -> schema class のマッピング (★ これを動的に取得する仕組みが必要)
        # 例: EVENT_SCHEMAS = {'evaluation_complete': schemas.EvaluationCompleteData, ...}
        # ここでは手動マッピングで実装するが、本来はより汎用的な方法が良い
        schema_class = _get_schema_for_event_type(event_type)

        if schema_class:
            # event_data をスキーマで検証・パース
            # Pydantic v2 では .model_validate() を使用
            validated_event_data_model = schema_class.model_validate(event_data)
            validated_event_data_dict = validated_event_data_model.model_dump(mode='json') # DB保存用にJSON互換形式でダンプ
            logger.debug(f"Validated event data for '{event_type}' using {schema_class.__name__}")
        else:
            # スキーマが見つからない場合、警告ログを出力し、辞書としてそのまま使用 (検証なし)
            logger.warning(f"No specific schema found for event_type '{event_type}'. Using raw data dict.")
            validated_event_data_dict = event_data # 検証なし

        # cycle_state_update を検証 (キーが SessionCycleState のフィールドかチェック)
        if cycle_state_update is not None:
            validated_update_dict = {}
            # SessionCycleState のフィールド定義を取得 (Pydantic v2)
            cycle_state_fields = schemas.SessionCycleState.model_fields.keys()
            for key, value in cycle_state_update.items():
                if key in cycle_state_fields:
                    # TODO: 値の型チェックも追加するのが望ましい
                    validated_update_dict[key] = value
                else:
                    logger.warning(f"Ignoring invalid key '{key}' in cycle_state_update for session {session_id}")
            if not validated_update_dict:
                 validated_update_dict = None # 有効なキーがなければ None に戻す
            else:
                 logger.debug(f"Validated cycle state update keys: {list(validated_update_dict.keys())}")

    except ValidationError as ve:
        logger.error(f"History event data validation failed for event '{event_type}' (Session: {session_id}): {ve}", exc_info=True)
        # エラー詳細を含めて ValueError を送出
        raise ValueError(f"Invalid data for history event '{event_type}': {ve}") from ve
    except Exception as e_val:
        logger.error(f"Unexpected error during history data validation (Session: {session_id}): {e_val}", exc_info=True)
        raise StateManagementError(f"Unexpected validation error: {e_val}") from e_val

    # new_event 辞書の data を検証済み辞書で更新
    new_event = {"id": generate_id(), "timestamp": current_timestamp, "type": event_type, "data": validated_event_data_dict}

    # --- 2. DB 操作 (ロックとトランザクション) ---
    updated_best_metrics_json: Optional[str] = None # 更新用の変数を準備
    updated_best_code_version: Optional[str] = None
    updated_best_code_path: Optional[str] = None
    log_msg_suffix = f"履歴追加 (Event: {event_type})" # ログメッセージの初期化

    logger.debug(f"[add_history] Attempting DB lock for {session_id} (Event: {event_type})...")
    async with db_utils.db_lock: # ★ DB操作全体をロックで囲む
        logger.debug(f"[add_history] Acquired DB lock for {session_id}")
        conn = None # finally節のために宣言
        try:
            conn = await db_utils.get_db_connection(db_path) # 非同期接続取得
            async with conn.execute("BEGIN"): # 手動トランザクション開始
                # 2a. 現在の状態を取得 (ロック内で実行)
                cursor = await conn.execute(
                    # config カラムも取得して improvement_threshold, max_history_events を読み取る
                    "SELECT history, cycle_state, best_metrics, config, best_code_version, best_code_path FROM sessions WHERE session_id = ?",
                    (session_id,)
                )
                row = await cursor.fetchone()
                if not row:
                    logger.warning(f"[add_history] Session not found inside lock: {session_id}")
                    raise ValueError(f"セッション {session_id} が見つかりません")

                # 2b. 履歴と状態の準備 (ロード)
                try:
                    # ★ history はリストなので json.loads のまま
                    current_history: List[Dict] = json.loads(row["history"] or '[]')
                    if not isinstance(current_history, list): current_history = []

                    # ★ cycle_state は model_validate_json を使用
                    current_cycle_state = schemas.SessionCycleState.model_validate_json(row["cycle_state"] or '{}')

                    # ★ best_metrics も model_validate_json を使用 (適切なスキーマが必要。ここではDictのまま扱う)
                    #    より厳密には best_metrics のスキーマ (例: EvaluationResultData['metrics']) を定義すべき
                    current_best_metrics_dict: Optional[Dict] = json.loads(row["best_metrics"] or 'null')
                    if not isinstance(current_best_metrics_dict, dict) and current_best_metrics_dict is not None:
                         current_best_metrics_dict = {}

                    # ★ config も model_validate_json (スキーマがあれば)
                    #    現状は辞書のまま扱う
                    session_config_dict: Dict = json.loads(row["config"] or '{}')
                    if not isinstance(session_config_dict, dict): session_config_dict = {}
                    improvement_threshold = session_config_dict.get('improvement_threshold', DEFAULT_IMPROVEMENT_THRESHOLD)
                    # ★ 追加: 履歴最大数を設定から読み込み (デフォルト値も設定)
                    max_history_events = config.get('limits', {}).get('max_history_events_per_session', 1000)

                    # 既存のベストコード情報を取得
                    existing_best_code_version = row["best_code_version"]
                    existing_best_code_path = row["best_code_path"]

                except (json.JSONDecodeError, TypeError, ValidationError) as load_err:
                    logger.error(f"セッション {session_id} の状態ロード中にエラー: {load_err}", exc_info=True)
                    raise StateManagementError(f"Error loading session state: {load_err}") from load_err

                # 2c. イベントタイプに応じた状態更新ロジック ★★★
                state_updated = False
                best_result_updated = False
                # Pydantic モデルのフィールドを直接更新
                current_cycle_state_dict = current_cycle_state.model_dump() # 更新用に辞書に戻す

                # Determine the primary metric for comparison
                primary_metric_path = config.get('evaluation', {}).get('primary_metric', 'overall.f_measure')
                primary_metric_keys = primary_metric_path.split('.') # e.g., ['overall', 'f_measure']

                def get_metric_value(metrics_dict, keys):
                    val = metrics_dict
                    try:
                        for key in keys:
                            val = val[key]
                        return float(val) if isinstance(val, (int, float)) else None
                    except (KeyError, TypeError, ValueError):
                        return None

                if event_type == "evaluation_complete":
                    # validated_event_data_dict は EvaluationCompleteData.model_dump() のはず
                    new_metrics = validated_event_data_dict.get("metrics", {})
                    new_primary_metric_value = get_metric_value(new_metrics, primary_metric_keys)
                    new_code_version = validated_event_data_dict.get("code_version")
                    new_code_path = validated_event_data_dict.get("code_path")

                    if new_primary_metric_value is not None:
                        current_best_f = get_metric_value(current_best_metrics_dict or {}, primary_metric_keys)
                        logger.info(f"Session {session_id}: New {primary_metric_path}={new_primary_metric_value:.4f}, Best={current_best_f or 'N/A'}")

                        if current_best_f is None or new_primary_metric_value > (current_best_f + improvement_threshold):
                            logger.info(f"Session {session_id}: Primary metric improved! Updating best metrics and resetting stagnation.")
                            updated_best_metrics_dict = validated_event_data_dict.get("metrics", {})
                            updated_best_metrics_json = json.dumps(updated_best_metrics_dict, cls=NumpyEncoder)
                            updated_best_code_version = new_code_version
                            updated_best_code_path = new_code_path
                            best_result_updated = True
                            current_cycle_state.stagnation_count = 0
                            state_updated = True
                            log_msg_suffix += f", ベストスコア更新 ({primary_metric_path})"
                        else:
                            logger.info(f"Session {session_id}: Primary metric did not improve significantly. Incrementing stagnation.")
                            current_cycle_state.stagnation_count += 1
                            state_updated = True
                    else:
                         logger.warning(f"Session {session_id}: Evaluation complete event missing primary metric '{primary_metric_path}' in results. Cannot update stagnation.")

                # ★ 失敗イベントでの停滞カウント更新を有効化 ★
                elif event_type in ["evaluation_failed", "cycle_failed", "improve_code_failed", "analyze_evaluation_failed", 
                                   "parameter_suggestion_failed"] or event_type.endswith("_failed") or event_type == "cycle_error":
                    logger.warning(f"Session {session_id}: Event '{event_type}' occurred. Incrementing stagnation count.")
                    current_cycle_state.stagnation_count += 1
                    state_updated = True

                # 2d. cycle_state_update 引数による更新を適用 (Pydanticモデルに対して)
                if validated_update_dict:
                    logger.debug(f"Applying cycle_state_update: {validated_update_dict}")
                    try:
                        # ★ update_forward_refs は Pydantic v2 では不要 or 別の方法?
                        #    update 辞書を使ってモデルを更新する (フィールドが存在するかは検証済み)
                        updated_state = current_cycle_state.model_copy(update=validated_update_dict)
                        # Check if stagnation count was manually set to 0 (e.g., after successful param opt)
                        # Avoid overriding reset based on evaluation comparison if external update also resets
                        if 'stagnation_count' in validated_update_dict and best_result_updated:
                            logger.debug("Stagnation reset by both evaluation improvement and external update. Using external update.")
                            # The model_copy already applied the external value.
                            
                        current_cycle_state = updated_state # 更新されたモデルを代入
                        state_updated = True
                        if "cycle_state 更新" not in log_msg_suffix: log_msg_suffix += ", cycle_state 更新"
                    except Exception as update_err:
                         # 通常は発生しないはず (キー検証済みのため)
                         logger.warning(f"Failed to apply cycle_state_update {validated_update_dict} to model: {update_err}")

                # 2e. 最終的な履歴と状態をJSONに変換
                current_history.append(new_event)

                # ★ 追加: 履歴が最大数を超えていたら古いものから削除
                if len(current_history) > max_history_events:
                    num_to_remove = len(current_history) - max_history_events
                    logger.warning(f"Session {session_id} history exceeds limit ({max_history_events}). Removing {num_to_remove} oldest events.")
                    current_history = current_history[num_to_remove:]

                updated_history_json = json.dumps(current_history, cls=NumpyEncoder)

                # ★ 更新された Pydantic モデルから JSON 文字列を生成
                updated_cycle_state_json = current_cycle_state.model_dump_json() if state_updated else row["cycle_state"]

                # 2f. DB 更新実行
                sql_parts = ["UPDATE sessions SET history = ?, last_update = ?"]
                params: List[Any] = [updated_history_json, current_timestamp]

                if state_updated:
                    sql_parts.append("cycle_state = ?")
                    params.append(updated_cycle_state_json)
                # best_result_updated フラグが True の場合のみ best_* カラムを更新
                if best_result_updated:
                    if updated_best_metrics_json is not None: # None チェックを追加
                         sql_parts.append("best_metrics = ?")
                         params.append(updated_best_metrics_json)
                    if updated_best_code_version is not None:
                         sql_parts.append("best_code_version = ?")
                         params.append(updated_best_code_version)
                    if updated_best_code_path is not None:
                         sql_parts.append("best_code_path = ?")
                         params.append(updated_best_code_path)

                sql = ", ".join(sql_parts) + " WHERE session_id = ?"
                params.append(session_id)

                await conn.execute(sql, tuple(params)) # パラメータをタプルに変換
                await conn.commit() # トランザクションをコミット
                logger.info(f"セッション {session_id}: {log_msg_suffix} 完了")

        except (ValueError, StateManagementError) as known_err:
            # これらのエラーはキャッチしてロールバック後、再度raise
            if conn: await conn.rollback()
            logger.warning(f"[add_history] Rolling back transaction for {session_id} due to: {known_err}")
            raise known_err
        except aiosqlite.Error as db_err: # aiosqlite.Error をキャッチ (OperationalError を含む)
            # DBロックなどの可能性
            if conn: await conn.rollback()
            logger.error(f"Database error updating history for {session_id}: {db_err}", exc_info=True)
            raise StateManagementError(f"Database error: {db_err}") from db_err
        except Exception as e_db:
            if conn: await conn.rollback()
            logger.error(f"Unexpected database error updating history for {session_id}: {e_db}", exc_info=True)
            raise StateManagementError(f"Unexpected DB error during history update: {e_db}") from e_db
        finally:
            if conn:
                await conn.close()
            logger.debug(f"[add_history] Released DB lock for {session_id}")

# --- Helper function to get schema class --- #
_EVENT_TYPE_TO_SCHEMA = {
    # Evaluation Tool Events
    "evaluation_started": schemas.HistoryEventBaseData, # Generic for now
    "evaluation_complete": schemas.HistoryEventBaseData, # Generic for now (EvaluationCompleteData not defined)
    "evaluation_failed": schemas.HistoryEventBaseData, # TODO: Define EvaluationFailedData
    # Grid Search Tool Events
    "grid_search_started": schemas.HistoryEventBaseData, # Generic for now
    "grid_search_complete": schemas.HistoryEventBaseData, # Reuse? Or create specific history schema?
    "grid_search_failed": schemas.HistoryEventBaseData, # TODO: Define GridSearchFailedData
    # Code Tool Events
    "code_save_started": schemas.HistoryEventBaseData, # Generic for now
    "code_save_complete": schemas.HistoryEventBaseData, # Reuse?
    "code_save_failed": schemas.HistoryEventBaseData,
    # LLM Tool Events (Parameter Suggestion)
    "parameter_suggestion_started": schemas.HistoryEventBaseData,
    "parameter_suggestion_complete": schemas.HistoryEventBaseData,
    "parameter_suggestion_failed": schemas.HistoryEventBaseData,
    # LLM Tool Events (Code Improvement)
    "improve_code_started": schemas.HistoryEventBaseData,
    "improve_code_complete": schemas.HistoryEventBaseData,
    "improve_code_failed": schemas.HistoryEventBaseData,
    # LLM Tool Events (Analysis)
    "analyze_evaluation_started": schemas.HistoryEventBaseData,
    "analyze_evaluation_complete": schemas.HistoryEventBaseData,
    "analyze_evaluation_failed": schemas.HistoryEventBaseData,
    # LLM Tool Events (Strategy/Hypothesis)
    "strategy_suggested": schemas.HistoryEventBaseData,
    "hypotheses_generated": schemas.HistoryEventBaseData,
    # Session Management Events
    "session_started": schemas.HistoryEventBaseData,
    "session_resumed": schemas.HistoryEventBaseData,
    "session_paused": schemas.HistoryEventBaseData,
    "session_stopped": schemas.HistoryEventBaseData,
    "session_completed": schemas.HistoryEventBaseData,
    "session_timeout": schemas.HistoryEventBaseData,
    "session_error": schemas.HistoryEventBaseData,
    # Cycle Specific Events (from old improve loop logic, potentially reused by server)
    "cycle_completed": schemas.HistoryEventBaseData,
    "cycle_failed": schemas.HistoryEventBaseData,
    "cycle_error": schemas.HistoryEventBaseData,
    "stopped_due_to_stagnation": schemas.HistoryEventBaseData,
    "stopped_by_strategy": schemas.HistoryEventBaseData,
    "analysis_failed": schemas.HistoryEventBaseData,
    "suggestion_failed": schemas.HistoryEventBaseData,
    "code_improvement_failed": schemas.HistoryEventBaseData,
    # プロンプト生成関連のイベントタイプを追加
    "improve_code_prompt_generation_started": schemas.HistoryEventBaseData,
    "improve_code_prompt_generation_complete": schemas.HistoryEventBaseData,
    "improve_code_prompt_generation_failed": schemas.HistoryEventBaseData,
    
    "parameter_suggestion_prompt_generation_started": schemas.HistoryEventBaseData,
    "parameter_suggestion_prompt_generation_complete": schemas.HistoryEventBaseData,
    "parameter_suggestion_prompt_generation_failed": schemas.HistoryEventBaseData,
    
    "analyze_evaluation_prompt_generation_started": schemas.HistoryEventBaseData,
    "analyze_evaluation_prompt_generation_complete": schemas.HistoryEventBaseData,
    "analyze_evaluation_prompt_generation_failed": schemas.HistoryEventBaseData,
    
    "strategy_suggestion_prompt_generation_started": schemas.HistoryEventBaseData,
    "strategy_suggestion_prompt_generation_complete": schemas.HistoryEventBaseData,
    "strategy_suggestion_prompt_generation_failed": schemas.HistoryEventBaseData,
    
    "hypotheses_generation_prompt_generation_started": schemas.HistoryEventBaseData,
    "hypotheses_generation_prompt_generation_complete": schemas.HistoryEventBaseData,
    "hypotheses_generation_prompt_generation_failed": schemas.HistoryEventBaseData,
    
    "assess_improvement_prompt_generation_started": schemas.HistoryEventBaseData,
    "assess_improvement_prompt_generation_complete": schemas.HistoryEventBaseData,
    "assess_improvement_prompt_generation_failed": schemas.HistoryEventBaseData,
    # Add other mappings as needed
}

def _get_schema_for_event_type(event_type: str) -> Optional[Type[BaseModel]]:
    """Returns the Pydantic schema class corresponding to the event type."""
    return _EVENT_TYPE_TO_SCHEMA.get(event_type)

# --- Cleanup Task ---

async def cleanup_old_sessions(config: Dict[str, Any]):
    """古いセッションや最大数を超えたセッションをクリーンアップする (非同期)"""
    db_path = Path(config['paths']['db']) / db_utils.DB_FILENAME
    session_timeout_seconds = config.get('cleanup', {}).get('session_timeout_seconds', 86400)
    max_sessions_count = config.get('cleanup', {}).get('max_sessions_count', 100)

    if session_timeout_seconds <= 0 and max_sessions_count <= 0:
        logger.info("セッションのタイムアウトと最大数が無効なため、クリーンアップをスキップします。")
        return

    logger.info(f"古いセッションのクリーンアップを開始 (タイムアウト: {session_timeout_seconds}秒, 最大: {max_sessions_count}件)")
    deleted_timeout_count = 0
    deleted_excess_count = 0

    # --- DB操作は非同期関数 (db_utils.db_*) を使う --- #
    try:
        # タイムアウトしたセッションを削除
        if session_timeout_seconds > 0:
            timeout_threshold = get_timestamp() - session_timeout_seconds
            # 非同期関数を使用
            deleted_timeout_count_result = await db_utils.db_execute_commit_async(db_path, "DELETE FROM sessions WHERE last_update < ?", (timeout_threshold,))
            # execute_commit_async は lastrowid (ここではNone) または影響行数を返さないので、ログのみ
            # if deleted_timeout_count_result:
            #     logger.info(f"{deleted_timeout_count_result} 件のタイムアウトしたセッションを削除しました。")
            logger.info(f"タイムアウトしたセッション削除試行完了 (last_update < {timeout_threshold})。") # 削除件数は正確には取れない

        # 最大数を超えたセッションを削除
        if max_sessions_count > 0:
            # 非同期関数を使用
            row = await db_utils.db_fetch_one_async(db_path, "SELECT COUNT(*) FROM sessions")
            current_session_count = row[0] if row else 0
            if current_session_count > max_sessions_count:
                sessions_to_delete_count = current_session_count - max_sessions_count
                logger.info(f"セッション数が最大保持数 {max_sessions_count} を超過 ({current_session_count} 件)。古い {sessions_to_delete_count} 件を削除します。")
                # 非同期関数を使用
                # execute_commit_async は削除件数を返さないので、ログのみ
                await db_utils.db_execute_commit_async(db_path, "DELETE FROM sessions WHERE session_id IN (SELECT session_id FROM sessions ORDER BY last_update ASC LIMIT ?)", (sessions_to_delete_count,))
                logger.info(f"{sessions_to_delete_count} 件の超過セッション削除試行完了。")
            else:
                 logger.info(f"現在のセッション数: {current_session_count} (最大: {max_sessions_count}) - 超過削除は不要。")

        logger.info(f"セッションクリーンアップ完了。") # 正確な削除件数はログのみで判断

    except StateManagementError as sme:
        logger.error(f"セッションクリーンアップ中にDBエラー: {sme}", exc_info=False)
    except Exception as e:
        logger.error(f"セッションクリーンアップ中に予期せぬエラー: {e}", exc_info=True)

# --- New function for Phase 2 --- #
async def get_session_summary_for_prompt(
    config: dict,
    db_path: Path,
    session_id: str,
    max_summary_length: int = 2000 # Example limit
) -> str:
    """Fetches and summarizes the session history for LLM prompts.

    Args:
        config: Application configuration.
        db_path: Path to the SQLite database.
        session_id: The ID of the session to summarize.
        max_summary_length: Approximate maximum character length for the summary.

    Returns:
        A string summarizing the session history, focusing on evaluations,
        LLM interactions, and errors. Returns an empty string if no history
        or on error.
    """
    logger.debug(f"Summarizing history for session {session_id}")
    try:
        session_data = await get_session_info(config, db_path, session_id)
        history = session_data.get('history', [])
        if not history:
            logger.debug(f"No history found for session {session_id}")
            return ""

        # --- Summarization Logic (Placeholder) --- #
        # TODO: Implement more sophisticated summarization.
        #       - Prioritize recent events.
        #       - Extract key evaluation metrics (trends).
        #       - Summarize LLM suggestions and their outcomes.
        #       - Note significant errors.

        summary_parts = []
        current_length = 0

        # Iterate in reverse to prioritize recent events
        for event in reversed(history):
            event_str = json.dumps(event, cls=NumpyEncoder, indent=2)
            event_len = len(event_str)

            if current_length + event_len > max_summary_length and summary_parts:
                summary_parts.append("... (history truncated)")
                break

            summary_parts.append(event_str)
            current_length += event_len

        # Reverse back to chronological order for the final string
        summary = "\n".join(reversed(summary_parts))
        logger.debug(f"Generated history summary for session {session_id} (length: {len(summary)})" )
        return summary

    except Exception as e:
        logger.error(f"Error summarizing history for session {session_id}: {e}", exc_info=True)
        # Return empty string on error to avoid breaking the LLM prompt
        return ""

async def update_session_status_internal(
    config: Dict[str, Any],
    db_path: Path,
    session_id: str,
    new_status: str, # TODO: Use SessionStatus enum?
    error_details: Optional[Dict[str, Any]] = None
) -> None:
    """内部的にセッションステータスを更新するための関数"""
    current_timestamp = get_timestamp()
    sql_parts = ["UPDATE sessions SET status = ?, last_update = ?"]
    params: List[Any] = [new_status, current_timestamp]

    # Optionally update last_error if provided
    # TODO: Add a last_error column (TEXT) to sessions table in init_database
    # if error_details:
    #     try:
    #         error_json = json.dumps(error_details, cls=NumpyEncoder)
    #         sql_parts.append("last_error = ?")
    #         params.append(error_json)
    #     except TypeError as e:
    #         logger.warning(f"Failed to serialize error_details for session {session_id}: {e}")

    sql = ", ".join(sql_parts) + " WHERE session_id = ?"
    params.append(session_id)

    try:
        # Use the atomic DB update function
        await db_utils.db_execute_commit_async(db_path, sql, tuple(params))
        logger.info(f"セッション {session_id}: ステータスを '{new_status}' に更新しました")
    except StateManagementError as e:
        logger.error(f"Failed to update session status for {session_id} in DB: {e}")
        # Re-raise or handle appropriately
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating session status for {session_id}: {e}", exc_info=True)
        raise StateManagementError(f"Unexpected error updating session status: {e}") from e

# --- Tool Registration --- #

def register_session_tools(mcp: FastMCP, config: Dict[str, Any], db_path: Path):
    """セッション管理関連のMCPツールを登録"""
    logger.info("Registering session management tools...")

    # MCPツールとして登録するために、必要な引数 (config, db_path) を部分適用
    mcp.tool("start_session")(
        lambda base_algorithm="Unknown": start_session(config, db_path, base_algorithm)
    )
    mcp.tool("get_session_info")(
        lambda session_id: get_session_info(config, db_path, session_id)
    )
    # add_session_history は内部利用のみを想定し、ツールとしては公開しない (Phase 2.4 参照)
    # mcp.tool("add_session_history")(
    #     lambda session_id, event_type, event_data, cycle_state_update=None:
    #         add_session_history(config, db_path, session_id, event_type, event_data, cycle_state_update)
    # )

    logger.info("Session management tools registered.")

# --- Import Guard --- #
# This module should not be run directly
if __name__ == "__main__":
    print("This module is intended for import by the main MCP server script.") 