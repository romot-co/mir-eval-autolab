import typer
from typing import Optional, List
import asyncio
from pathlib import Path
import os
import sys
import logging
from datetime import datetime
import yaml
import time # monitor_job_status で使用
import requests # call_mcp_api で使用
import json

# --- PYTHONPATH設定 ---
# スクリプトの場所に基づいてプロジェクトルートを決定
SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = SCRIPT_DIR # このスクリプトがプロジェクトルートにある前提
# src/cli/ に移動したので、プロジェクトルートは2つ上
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# src/ は PROJECT_ROOT を追加すれば不要なはず
# if str(PROJECT_ROOT / 'src') not in sys.path: # srcも追加
#     sys.path.append(str(PROJECT_ROOT / 'src'))

# --- モジュールインポート ---
try:
    from src.utils.path_utils import (
        get_project_root, load_environment_variables, get_workspace_dir,
        get_dataset_paths, get_output_dir, ensure_dir,
        validate_path_within_allowed_dirs
    )
    # exception_utils から必要な例外をインポート
    from src.utils.exception_utils import (
        log_exception, EvaluationError, FileError, ConfigError,
        format_exception_message, wrap_exceptions, MirexError, GridSearchError # GridSearchErrorを追加
    )
    from src.evaluation.evaluation_runner import run_evaluation as run_eval_func
    from src.evaluation.grid_search.core import run_grid_search as run_grid_func
    # 修正: visualize_session を直接インポート
    from src.visualization.plots import plot_detection_results # 可視化関数 (仮、必要に応じて変更)
    # MCPクライアントやAutolabクライアントは現状未使用のためコメントアウト or 削除
    # from src.autolab.autolab_client import AutoLabClient
    # AutoImprover クラスをインポート
    from src.cli.auto_improver import AutoImprover

except ImportError as e:
    print(f"エラー: 必要なモジュールのインポートに失敗しました: {e}", file=sys.stderr)
    print(f"PYTHONPATH: {sys.path}", file=sys.stderr)
    print("プロジェクト構造とPYTHONPATHを確認してください。", file=sys.stderr)
    sys.exit(1)

# --- アプリケーション設定 ---
app = typer.Typer(help="アルゴリズム改善と評価のためのCLIツール")

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ImproverCLI") # ロガー名を変更

# --- 設定読み込み ---
load_environment_variables()

def load_default_config(config_path: Path) -> dict:
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"config.yaml の読み込みに失敗しました: {e}")
            return {}
    else:
        logger.warning(f"config.yaml が見つかりません: {config_path}")
        return {}

# プロジェクトルートを再取得 (path_utilsロード後)
PROJECT_ROOT = get_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'config.yaml'
DEFAULT_CONFIG = load_default_config(DEFAULT_CONFIG_PATH)
CONFIG = DEFAULT_CONFIG.copy()

# --- Define allowed directories for client-side input validation ---
# クライアント側で読み取りアクセスを検証する基本ディレクトリ
# (サーバー側でもさらに厳密な検証が行われる)
try:
    CLIENT_ALLOWED_READ_DIRS = [
        PROJECT_ROOT,
        get_workspace_dir(config=CONFIG) # workspaceも許可
        # 必要に応じてデータセットベースなども追加可能だが、まずはシンプルに
    ]
except Exception as e:
    logger.error(f"クライアントの許可ディレクトリ設定中にエラー: {e}. パス検証が機能しない可能性があります。")
    CLIENT_ALLOWED_READ_DIRS = [] # エラー時は空リストにフォールバック

# パス設定 (環境変数 > .env > デフォルト)
CONFIG['paths'] = CONFIG.get('paths', {})
CONFIG['paths'].update({
    'workspace_dir': str(get_workspace_dir()), # 文字列に変換
    'evaluation_results_dir': str(get_evaluation_results_dir()),
    'grid_search_results_dir': str(get_grid_search_results_dir()),
    'improved_versions_dir': str(get_improved_versions_dir()),
    'audio_dir': str(get_audio_dir()),
    'reference_dir': str(get_label_dir()),
    'data_dir': str(get_workspace_dir() / "data")
})

# --- MCPサーバー通信 ヘルパー関数 (簡易実装) ---
# 実際のMCPサーバーURLは環境変数や設定ファイルから取得するべき
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:5002")

class RequestError(Exception):
    """MCPサーバーとの通信エラーを表す例外"""
    pass

def call_mcp_api(endpoint: str, payload: Optional[dict] = None, method: str = "POST") -> dict:
    """MCPサーバーのAPIを呼び出す"""
    # mcp_server.py で / にマウントし、ツール呼び出しパスが /<ツール名> であると仮定
    # endpoint 引数がツール名そのものであると想定 (例: 'start_session')
    url = f"{MCP_SERVER_URL}/{endpoint}"
    try:
        if method.upper() == "POST":
            response = requests.post(url, json=payload, timeout=60)
        elif method.upper() == "GET":
            response = requests.get(url, params=payload, timeout=60)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status() # HTTPエラーがあれば例外発生
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RequestError(f"MCP API呼び出しエラー ({url}): {e}") from e
    except json.JSONDecodeError as e:
        raise RequestError(f"MCP API応答のJSONデコードエラー ({url}): {e}. Response: {response.text[:200]}...")

def monitor_job_status(job_id: str, poll_interval: int = 5, timeout: int = 600) -> str:
    """MCPジョブの完了をポーリングする"""
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f"ジョブ {job_id} のポーリングがタイムアウトしました ({timeout}秒)")

        try:
            status_info = call_mcp_api(f"job/status/{job_id}", method="GET") # エンドポイント修正
            status = status_info.get("status")
            logger.info(f"ジョブ {job_id} ステータス: {status} (経過時間: {elapsed_time:.1f}秒)")

            if status == "completed":
                return "completed"
            elif status == "failed":
                error_msg = status_info.get("error", "不明なエラー")
                raise RuntimeError(f"MCPジョブ {job_id} が失敗しました: {error_msg}")
            elif status in ["pending", "running", "queued"]: # queuedも考慮
                time.sleep(poll_interval)
            else:
                logger.warning(f"不明なジョブステータス: {status}")
                time.sleep(poll_interval)

        except RequestError as e:
            logger.error(f"ジョブステータス取得エラー: {e}")
            time.sleep(poll_interval * 2) # エラー時は少し長めに待つ
        except Exception as e:
             logger.error(f"ジョブポーリング中に予期せぬエラー: {e}", exc_info=True)
             raise # 予期せぬエラーは再発生させる

# --- Typer Commands ---

@app.command()
async def run(
    detector_name: str = typer.Argument(..., help="使用する検出器の名前。"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="改善されたパラメータと結果の出力先ディレクトリ。", resolve_path=True),
    audio_path: Optional[Path] = typer.Option(None, "--audio-path", help="評価に使用する単一のオーディオファイルパス。--dataset とは併用不可。", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    ref_path: Optional[Path] = typer.Option(None, "--ref-path", help="評価に使用する単一の参照ファイルパス。--dataset とは併用不可。", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    dataset_name: Optional[str] = typer.Option(None, "--dataset", help="評価に使用するデータセット名。--audio-path/--ref-path とは併用不可。"),
    max_iterations: int = typer.Option(10, "--max-iterations", "-i", help="最大改善イテレーション数。"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="追加設定ファイルのパス。", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="再開するセッションID"),
):
    """
    指定された検出器のパラメータ自動改善を実行します。
    auto_improver.py の AutoImprover クラスを直接呼び出します。
    --dataset または --audio-path/--ref-path のいずれかを指定する必要があります。
    """
    logger.info(f"\'{detector_name}\' の改善プロセスを開始します (AutoImprover 呼び出し)...")

    # 引数のバリデーション (AutoImprover側でも行うかもしれないが、CLIでもチェック)
    if dataset_name and (audio_path or ref_path):
        logger.error("--dataset オプションと --audio-path/--ref-path オプションは同時に指定できません。")
        typer.echo("エラー: --dataset オプションと --audio-path/--ref-path オプションは同時に指定できません。", err=True)
        raise typer.Exit(code=1)
    if not dataset_name and not (audio_path and ref_path):
        logger.error("--dataset または (--audio-path と --ref-path の両方) のいずれかを指定する必要があります。")
        typer.echo("エラー: --dataset または (--audio-path と --ref-path の両方) のいずれかを指定する必要があります。", err=True)
        raise typer.Exit(code=1)
    if (audio_path and not ref_path) or (not audio_path and ref_path):
         logger.error("--audio-path と --ref-path は両方指定する必要があります。")
         typer.echo("エラー: --audio-path と --ref-path は両方指定する必要があります。", err=True)
         raise typer.Exit(code=1)

    typer.echo(f"検出器: {detector_name}")
    if dataset_name:
        typer.echo(f"データセット: {dataset_name}")
    else:
        typer.echo(f"オーディオパス: {audio_path}")
        typer.echo(f"参照パス: {ref_path}")
    typer.echo(f"最大イテレーション数: {max_iterations}")
    if session_id:
        typer.echo(f"再開セッションID: {session_id}")
    if config_file:
        typer.echo(f"追加設定ファイル: {config_file}")

    # output_dir は AutoImprover が config やデフォルト値に基づいて決定する想定
    # ensure_dir(output_dir) は AutoImprover 内で行う

    try:
        # AutoImprover インスタンスを作成 (config_file を渡す)
        # server_url は AutoImprover 内で環境変数や config から読み込まれる
        improver = AutoImprover(config_path=config_file)

        typer.echo("改善サイクルを実行します...")
        # run_improvement_cycle を呼び出す
        result = await improver.run_improvement_cycle(
            detector_name=detector_name,
            dataset_name=dataset_name,
            audio_path=str(audio_path) if audio_path else None,
            ref_path=str(ref_path) if ref_path else None,
            iterations=max_iterations,
            session_id=session_id
        )

        # 結果を表示
        typer.echo("--- 改善サイクル完了 --- ")
        typer.echo(f"最終ステータス: {result.get('status', '不明')}")
        typer.echo(f"セッションID: {result.get('session_id')}")
        typer.echo(f"完了イテレーション数: {result.get('iterations_completed', 0)}")
        best_f = result.get('best_f_measure')
        best_f_str = f"{best_f:.4f}" if isinstance(best_f, float) else "N/A"
        typer.echo(f"最終ベストF値: {best_f_str} (Version: {result.get('best_code_version', 'N/A')})" )
        typer.echo(f"総所要時間: {result.get('total_duration_seconds', 0):.2f} 秒")

        if result.get('status') not in ['completed', 'timeout']:
            typer.echo("エラーが発生したか、予期せず終了しました。ログを確認してください。", err=True)
            # エラーの詳細を表示 (final_cycle_state から取得できるか？)
            error_msg = result.get('final_cycle_state', {}).get('error_message', '詳細不明')
            typer.echo(f"エラー詳細: {error_msg}", err=True)
            raise typer.Exit(code=1)

    except Exception as e:
        # AutoImprover の実行時エラーなど
        logger.error(f"改善プロセスの実行中にエラーが発生しました: {e}", exc_info=True)
        typer.echo(f"エラー: 改善プロセスの実行中にエラーが発生しました: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def visualize(
    session_id: str = typer.Argument(..., help="視覚化するセッションID"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="グラフ画像の保存先ディレクトリ。省略時はサーバー側で決定。", resolve_path=True),
):
    """改善セッションの軌跡を視覚化します。"""
    typer.echo(f"セッション {session_id} の改善軌跡を視覚化しています...")

    payload = {"session_id": session_id}

    # ユーザー指定の出力ディレクトリがあればペイロードに追加
    # (visualize系ツールも output_dir を受け付けるように修正されている前提)
    if output_dir:
        payload['output_dir'] = str(output_dir)

    try:
        # 可視化ツール呼び出し (仮のツール名: visualize_session_history)
        # 実際のツール名に応じて変更してください
        vis_tool_name = "visualize_session_history"
        response = call_mcp_api(vis_tool_name, payload)

        job_id = response.get('job_id')
        if job_id:
            typer.echo(f"可視化ジョブを開始しました。Job ID: {job_id}")
            typer.echo("ジョブの完了を待機します...")
            monitor_job_status(job_id)
            typer.echo("可視化ジョブが完了しました。")
            # 完了後、結果のパスなどを表示（サーバーからの応答による）
            # 例: final_status = call_mcp_api(f"job/status/{job_id}", method="GET")
            #     output_files = final_status.get('result', {}).get('output_files', [])
            #     if output_files:
            #         typer.echo("生成されたファイル:")
            #         for f in output_files:
            #             typer.echo(f"- {f}")
        else:
            typer.echo(f"エラー: ジョブの開始に失敗しました。応答: {response}", err=True)
            raise typer.Exit(code=1)

    except RequestError as e:
        logger.error(f"可視化リクエストエラー: {e}")
        typer.echo(f"エラー: サーバーへのリクエストに失敗しました: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        # RequestError, TimeoutError, RuntimeError from monitor_job_status etc.
        logger.error(f"可視化ジョブ実行中にエラー: {e}", exc_info=True)
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
async def evaluate(
    detector_name: str = typer.Option(..., "--detector", "-d", help="使用する検出器の名前"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="評価結果の出力先ディレクトリ (省略時はサーバーが自動生成)", resolve_path=True),
    audio_path: Optional[Path] = typer.Option(None, "--audio-path", help="評価対象のオーディオファイルまたはディレクトリのパス（dataset と排他）", exists=True, resolve_path=True),
    ref_path: Optional[Path] = typer.Option(None, "--ref-path", help="参照ファイルまたはディレクトリのパス（dataset と排他）", exists=True, resolve_path=True),
    dataset_name: Optional[str] = typer.Option(None, "--dataset", "-D", help="使用するデータセット名 (config.yaml で定義)。audio-path/ref-path と排他。"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="追加設定ファイルのパス。", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    num_procs: Optional[int] = typer.Option(None, "--num-procs", "-p", help="並列処理に使用するプロセス数 (サーバー側で解釈)"),
    save_plots: bool = typer.Option(True, "--plot/--no-plot", help="結果のプロットを保存するかどうか (サーバー側で解釈)"),
    save_json: bool = typer.Option(True, "--json/--no-json", help="結果をJSONファイルに保存するかどうか (サーバー側で解釈)"),
    code_version: Optional[str] = typer.Option(None, "--version", "-v", help="評価するコードのバージョン (省略時は最新)")
):
    """
    指定された検出器をデータセットまたは単一ファイルで評価します。
    MCPサーバーの `run_evaluation` ツールを呼び出します。
    """
    logger.info(f"検出器 '{detector_name}' の評価を開始します (MCPサーバー呼び出し)...")

    # --- 入力パス検証 (クライアント側) ---
    validated_audio_path: Optional[str] = None
    validated_ref_path: Optional[str] = None
    try:
        if audio_path:
            validated_audio_path = str(validate_path_within_allowed_dirs(
                audio_path, CLIENT_ALLOWED_READ_DIRS, check_existence=True, check_is_file=True
            ))
        if ref_path:
            validated_ref_path = str(validate_path_within_allowed_dirs(
                ref_path, CLIENT_ALLOWED_READ_DIRS, check_existence=True, check_is_file=True
            ))
    except (ValueError, FileError, ConfigError) as e:
        logger.error(f"入力パスの検証に失敗しました: {e}")
        typer.echo(f"エラー: 入力パスの検証に失敗しました: {e}", err=True)
        raise typer.Exit(code=1)

    # --- 引数バリデーション (dataset vs audio/ref) ---
    if not dataset_name and not (audio_path and ref_path):
        logger.error("--dataset または (--audio-path と --ref-path の両方) のいずれかを指定する必要があります。")
        typer.echo("エラー: --dataset または (--audio-path と --ref-path の両方) のいずれかを指定する必要があります。", err=True)
        raise typer.Exit(code=1)
    if dataset_name and (audio_path or ref_path):
        logger.error("--dataset オプションと --audio-path/--ref-path オプションは同時に指定できません。")
        typer.echo("エラー: --dataset オプションと --audio-path/--ref-path オプションは同時に指定できません。", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"検出器: {detector_name}" + (f" (バージョン: {code_version})" if code_version else " (最新バージョン)"))
    if dataset_name:
        typer.echo(f"データセット: {dataset_name}")
    else:
        typer.echo(f"オーディオパス: {audio_path}")
        typer.echo(f"参照パス: {ref_path}")
    typer.echo(f"出力ディレクトリ (サーバー側): {output_dir}")
    if config_file:
        typer.echo(f"追加設定ファイル (サーバー側): {config_file}")

    # 出力ディレクトリ (サーバー側で処理されるため、CLI側での準備は不要)
    # ensure_dir(output_dir)

    # MCPサーバー呼び出し
    try:
        typer.echo("MCPサーバーに評価ジョブをリクエストしています...")

        # run_evaluation ツールの引数を準備
        tool_payload = {
            "detector_name": detector_name,
            "dataset_name": dataset_name,
            "audio_path": validated_audio_path,
            "ref_path": validated_ref_path,
            "output_dir": str(output_dir.resolve()) if output_dir else None,
            "code_version": code_version,
            # サーバー側のツールが解釈するオプション
            "num_procs": num_procs,
            "save_plots": save_plots,
            "save_json": save_json,
            "config_override_path": str(config_file.resolve()) if config_file else None # サーバー側での設定ファイル上書き用?
        }
        # None の値を除去
        tool_payload = {k: v for k, v in tool_payload.items() if v is not None}

        # run_evaluation ツールを呼び出す
        job_start_info = call_mcp_api("run_evaluation", payload=tool_payload, method="POST")
        job_id = job_start_info.get("job_id")

        if not job_id:
            error_msg = job_start_info.get('error', 'サーバーからジョブIDが返されませんでした')
            typer.echo(f"エラー: 評価ジョブの開始に失敗しました: {error_msg}", err=True)
            logger.error(f"評価ジョブの開始失敗: {error_msg}, Payload: {tool_payload}")
            raise typer.Exit(code=1)

        typer.echo(f"評価ジョブが開始されました。Job ID: {job_id}")
        logger.info(f"評価ジョブ開始成功: job_id={job_id}")

        # ジョブの完了を待機
        typer.echo("ジョブの完了を待っています...")
        # TODO: タイムアウト値を設定可能にする
        job_status = await asyncio.to_thread(monitor_job_status, job_id, timeout=1800) # monitor_job_statusは同期的のため別スレッドで実行

        if job_status == "completed":
            typer.echo("ジョブは正常に完了しました。最終結果を取得します...")
            final_job_info = call_mcp_api(f"job/status/{job_id}", method="GET")
            logger.info(f"最終ジョブ情報取得完了: job_id={job_id}, status={final_job_info.get('status')}")

            typer.echo("\n--- 評価結果 --- ")
            typer.echo(f"Job ID: {job_id}")
            result_data = final_job_info.get('result', {})
            evaluation_results = result_data.get('evaluation_results') # サーバーの応答形式に依存

            if evaluation_results:
                 typer.echo(json.dumps(evaluation_results, indent=2, ensure_ascii=False))
                 # TODO: サマリー表示など追加？
                 # print_summary_statistics(evaluation_results)
            else:
                 typer.echo("評価結果データが見つかりませんでした。サーバーの応答を確認してください。")
                 typer.echo(f"サーバー応答: {final_job_info}")

            # サーバー側の出力ディレクトリを表示 (もしあれば)
            server_output_dir = result_data.get('output_dir')
            if server_output_dir:
                typer.echo(f"\n結果はサーバー上の '{server_output_dir}' に保存されている可能性があります。")
        else:
            # monitor_job_status がエラーを raise するので、ここには到達しないはずだが念のため
            typer.echo(f"ジョブは予期せず終了しました (Status: {job_status})。", err=True)
            final_job_info = call_mcp_api(f"job/status/{job_id}", method="GET")
            typer.echo(f"最終ジョブ情報: {final_job_info}", err=True)
            raise typer.Exit(code=1)

    except RequestError as e:
        logger.error(f"MCPサーバーとの通信エラー: {e}", exc_info=True)
        typer.echo(f"エラー: MCPサーバーとの通信に失敗しました: {e}", err=True)
        raise typer.Exit(code=1)
    except TimeoutError as e:
        logger.error(f"ジョブのポーリングがタイムアウトしました: {e}")
        typer.echo(f"エラー: ジョブ待機がタイムアウトしました: {e}", err=True)
        raise typer.Exit(code=1)
    except RuntimeError as e:
        # monitor_job_status がジョブ失敗時に発生させる
        logger.error(f"MCPジョブが失敗しました: {e}")
        typer.echo(f"エラー: MCPジョブが失敗しました: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"評価コマンド実行中に予期せぬエラーが発生しました: {e}", exc_info=True)
        typer.echo(f"予期せぬエラー: {e}", err=True)
        raise typer.Exit(code=1)

@app.command(name="grid-search")
async def grid_search(
    config_path: Path = typer.Argument(..., help="グリッドサーチ設定ファイル (YAML) のパス。", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="グリッドサーチ結果の出力先ディレクトリ (省略時はサーバーが自動生成)", resolve_path=True),
    num_procs: Optional[int] = typer.Option(None, "--num-procs", "-p", help="並列処理に使用するプロセス数。省略時は自動検出 (サーバー側)。"),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="既存の結果をスキップする (サーバー側)。"),
    best_metric: str = typer.Option("note.f_measure", "--best-metric", "-b", help="最適化対象とするメトリック名 (例: 'note.f_measure', 'pitch.accuracy')。")
):
    """
    設定ファイルに基づいてパラメータのグリッドサーチを実行します。
    MCPサーバーの `run_grid_search` ツールを呼び出します。
    """
    logger.info(f"グリッドサーチを開始します (設定ファイル: {config_path}, MCPサーバー呼び出し)...")

    # --- 入力パス検証 (クライアント側) ---
    validated_config_path: str
    try:
        validated_config_path = str(validate_path_within_allowed_dirs(
            config_path, CLIENT_ALLOWED_READ_DIRS, check_existence=True, check_is_file=True
        ))
    except (ValueError, FileError, ConfigError) as e:
        logger.error(f"グリッドサーチ設定ファイルのパス検証に失敗しました: {e}")
        typer.echo(f"エラー: 設定ファイルのパス検証に失敗しました: {e}", err=True)
        raise typer.Exit(code=1)

    # --- 出力ディレクトリの決定ロジックは維持 (サーバーに渡す情報として) ---
    final_output_dir: Optional[Path] = None # Optional に変更
    if output_dir:
        final_output_dir = output_dir
    else:
        # detector_name を設定ファイルから読み取る試み (ベストエフォート)
        detector_name_from_config = "unknown"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                grid_config_data = yaml.safe_load(f)
                if isinstance(grid_config_data, dict):
                    detector_name_from_config = grid_config_data.get('detector_name', 'unknown')
        except Exception as e:
            logger.warning(f"設定ファイルからの検出器名読み込みに失敗: {e}")

        grid_results_base_dir = Path(CONFIG.get('paths', {}).get('grid_search_results_dir', './results/grid_search'))
        config_stem = config_path.stem
        # output_dir が None の場合、サーバー側でデフォルトの命名規則を使うことを期待する
        # もしクライアント側で生成したい場合は、ここで final_output_dir を設定する
        # unique_suffix = f"{detector_name_from_config}_{config_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # final_output_dir = get_output_dir(grid_results_base_dir, unique_suffix)
        # logger.info(f"出力ディレクトリが指定されていないため、サーバー側のデフォルトを使用します。")
        pass # サーバー側に任せる

    if final_output_dir:
         typer.echo(f"出力ディレクトリ (サーバー側): {final_output_dir}")
         # ensure_dir はサーバー側で行うため不要
    else:
         typer.echo(f"出力ディレクトリ: サーバー側のデフォルトを使用")

    # MCPサーバー呼び出し
    try:
        typer.echo("MCPサーバーにグリッドサーチジョブをリクエストしています...")

        # run_grid_search ツールの引数を準備
        tool_payload = {
            "config_path": validated_config_path,
            "output_dir": str(final_output_dir.resolve()) if final_output_dir else None,
            "num_procs": num_procs,
            "skip_existing": skip_existing,
            "best_metric": best_metric,
        }
        # None の値を除去
        tool_payload = {k: v for k, v in tool_payload.items() if v is not None}

        # run_grid_search ツールを呼び出す
        job_start_info = call_mcp_api("run_grid_search", payload=tool_payload, method="POST")
        job_id = job_start_info.get("job_id")

        if not job_id:
            error_msg = job_start_info.get('error', 'サーバーからジョブIDが返されませんでした')
            typer.echo(f"エラー: グリッドサーチジョブの開始に失敗しました: {error_msg}", err=True)
            logger.error(f"グリッドサーチジョブの開始失敗: {error_msg}, Payload: {tool_payload}")
            raise typer.Exit(code=1)

        typer.echo(f"グリッドサーチジョブが開始されました。Job ID: {job_id}")
        logger.info(f"グリッドサーチジョブ開始成功: job_id={job_id}")

        # ジョブの完了を待機
        typer.echo("ジョブの完了を待っています...")
        # TODO: タイムアウト値を設定可能にする
        job_status = await asyncio.to_thread(monitor_job_status, job_id, timeout=3600) # タイムアウトを長め(1時間)に設定

        if job_status == "completed":
            typer.echo("ジョブは正常に完了しました。最終結果を取得します...")
            final_job_info = call_mcp_api(f"job/status/{job_id}", method="GET")
            logger.info(f"最終ジョブ情報取得完了: job_id={job_id}, status={final_job_info.get('status')}")

            typer.echo("\n--- グリッドサーチ結果 --- ")
            typer.echo(f"Job ID: {job_id}")
            result_data = final_job_info.get('result', {}) # サーバーの応答形式に依存

            if result_data:
                 # サーバーからの結果を整形して表示
                 best_params = result_data.get("best_params")
                 best_metrics = result_data.get("best_metrics") # 辞書の可能性
                 server_output_dir = result_data.get('output_dir')

                 typer.echo("\n--- グリッドサーチ結果サマリー ---")
                 if best_params:
                     typer.echo(f"  最適パラメータ ({best_metric} 基準):")
                     typer.echo(json.dumps(best_params, indent=4))
                 else:
                     typer.echo("  最適パラメータが見つかりませんでした。")

                 if best_metrics:
                     # best_metric で指定された値を探す
                     metric_val = None
                     if isinstance(best_metrics, dict):
                         # ネストされた辞書の場合 (例: {"note": {"f_measure": 0.8}})
                         keys = best_metric.split('.')
                         current_val = best_metrics
                         try:
                             for k in keys:
                                 current_val = current_val[k]
                             metric_val = current_val
                         except (KeyError, TypeError):
                             logger.warning(f"指定された best_metric '{best_metric}' が結果内に見つかりません。")
                             # best_metrics 辞書全体を表示する
                             typer.echo(f"  最高メトリクス (全体):")
                             typer.echo(json.dumps(best_metrics, indent=4))

                     # 見つかった場合、または best_metrics が単一の値の場合
                     if metric_val is not None:
                          typer.echo(f"  最適メトリック ({best_metric}): {metric_val:.4f}" if isinstance(metric_val, float) else f"  最適メトリック ({best_metric}): {metric_val}")
                     elif not isinstance(best_metrics, dict):
                          typer.echo(f"  最高メトリクス: {best_metrics}") # 辞書でない場合そのまま表示
                 else:
                     typer.echo("  最高メトリクス情報が見つかりませんでした。")

                 if server_output_dir:
                     typer.echo(f"\n詳細な結果はサーバー上の '{server_output_dir}' ディレクトリに保存されました。")
                 else:
                      typer.echo(f"\n詳細な結果はサーバー側のデフォルトディレクトリに保存されました。")
            else:
                 typer.echo("グリッドサーチ結果データが見つかりませんでした。サーバーの応答を確認してください。")
                 typer.echo(f"サーバー応答: {final_job_info}")
        else:
             # monitor_job_status がエラーを raise するので、ここには到達しないはず
             typer.echo(f"ジョブは予期せず終了しました (Status: {job_status})。", err=True)
             final_job_info = call_mcp_api(f"job/status/{job_id}", method="GET")
             typer.echo(f"最終ジョブ情報: {final_job_info}", err=True)
             raise typer.Exit(code=1)

    except RequestError as e:
        logger.error(f"MCPサーバーとの通信エラー: {e}", exc_info=True)
        typer.echo(f"エラー: MCPサーバーとの通信に失敗しました: {e}", err=True)
        raise typer.Exit(code=1)
    except TimeoutError as e:
        logger.error(f"ジョブのポーリングがタイムアウトしました: {e}")
        typer.echo(f"エラー: ジョブ待機がタイムアウトしました: {e}", err=True)
        raise typer.Exit(code=1)
    except RuntimeError as e:
        # monitor_job_status がジョブ失敗時に発生させる
        logger.error(f"MCPジョブが失敗しました: {e}")
        typer.echo(f"エラー: MCPジョブが失敗しました: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"グリッドサーチコマンド実行中に予期せぬエラーが発生しました: {e}", exc_info=True)
        typer.echo(f"予期せぬエラー: {e}", err=True)
        raise typer.Exit(code=1)


# --- Added Commands --- #

@app.command()
def start_session(
    base_algorithm: str = typer.Option(..., "--base-algorithm", help="改善対象のベースアルゴリズム名"),
    config_str: Optional[str] = typer.Option(None, "--config-json", help="セッションの初期設定 (JSON文字列)"),
):
    """新しい改善セッションを開始します。"""
    typer.echo(f"Starting new session for base algorithm: {base_algorithm}...")
    payload = {"base_algorithm": base_algorithm}
    if config_str:
        try:
            payload["config"] = json.loads(config_str)
        except json.JSONDecodeError as e:
            typer.echo(f"Error: Invalid JSON format for --config-json: {e}", err=True)
            raise typer.Exit(code=1)

    try:
        # サーバー側のツール名 'start_session' を呼び出す想定
        # MCPの標準的なツール呼び出しエンドポイントを使用 (/call/ プレフィックスなし)
        result = call_mcp_api("start_session", payload=payload, method="POST")
        typer.echo("Session started successfully:")
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    except RequestError as e:
        typer.echo(f"Error starting session: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        logger.error("Unexpected error in start_session", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def get_session(
    session_id: str = typer.Option(..., "--session-id", help="情報を取得するセッションID")
):
    """指定されたセッションの情報を取得します。"""
    typer.echo(f"Getting information for session: {session_id}...")
    payload = {"session_id": session_id}

    try:
        # サーバー側のツール名 'get_session_info' を呼び出す想定
        result = call_mcp_api("get_session_info", payload=payload, method="POST")
        typer.echo("Session information:")
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    except RequestError as e:
        typer.echo(f"Error getting session info: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        logger.error("Unexpected error in get_session", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def get_job(
    job_id: str = typer.Option(..., "--job-id", help="状態を確認するジョブID")
):
    """指定されたジョブの状態と結果を取得します。"""
    typer.echo(f"Getting status for job: {job_id}...")
    payload = {"job_id": job_id}

    try:
        # サーバー側のツール名 'get_job_status' を呼び出す想定
        result = call_mcp_api("get_job_status", payload=payload, method="POST")
        typer.echo("Job status:")
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    except RequestError as e:
        typer.echo(f"Error getting job status: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        logger.error("Unexpected error in get_job", exc_info=True)
        raise typer.Exit(code=1)


# --- Main Execution Guard --- #
if __name__ == "__main__":
    app()