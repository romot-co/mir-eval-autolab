#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Desktop 設定ツール

Claude Desktopアプリケーションに、このMIRプロジェクトのMCPサーバーと
連携するための設定を自動的に適用します。

これにより、Claude Desktopから直接MCPサーバーのツール（評価実行、コード改善など）を
呼び出したり、関連するプロンプトテンプレートを利用したりできるようになります。

使用方法:
  python setup_claude_integration.py [--open-claude] [--config-path <path>]

オプション:
  --open-claude : 設定適用後にClaude Desktopアプリケーションを起動します。
  --config-path : Claude Desktopの設定ファイル(config.json)のパスを指定します。
                  省略した場合、OSに応じたデフォルトパスが使用されます。
"""

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("claude_setup")

# --- 定数 ---


def get_default_claude_config_path() -> Optional[str]:
    """OSに応じてClaude Desktopのデフォルト設定ファイルパスを取得"""
    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        return str(home / "Library/Application Support/Claude/storage/config.json")
    elif system == "Windows":
        # Windowsの正確なパスはAppData\Roamingなどになる可能性があります
        # ここでは一般的な場所を仮定しますが、ユーザー指定を推奨します
        app_data = os.environ.get("APPDATA")
        if app_data:
            return str(Path(app_data) / "Claude/storage/config.json")
        else:
            logger.warning(
                "WindowsのAPPDATA環境変数が見つかりません。config.jsonのパスを特定できませんでした。"
            )
            return None
    elif system == "Linux":
        # Linuxのパスは .config などになる可能性があります
        return str(home / ".config/Claude/storage/config.json")
    else:
        logger.error(f"未対応のOSです: {system}")
        return None


# 環境変数またはデフォルトパスを使用
DEFAULT_CONFIG_FILE = get_default_claude_config_path()
CLAUDE_CONFIG_PATH = os.environ.get("CLAUDE_CONFIG_PATH", DEFAULT_CONFIG_FILE)

# MIREXプロジェクトのルートディレクトリを特定
MIREX_PROJECT_ROOT = Path(__file__).resolve().parent


# --- メイン関数 ---
def setup_claude_desktop_config(config_path_override: Optional[str] = None):
    """Claude Desktopの設定ファイルを更新"""
    logger.info("Claude Desktop設定ツールを開始します")

    config_path = config_path_override or CLAUDE_CONFIG_PATH
    if not config_path:
        logger.error(
            "Claude設定ファイル(config.json)のパスを特定できませんでした。--config-pathで指定してください。"
        )
        return False

    config_path_obj = Path(config_path)
    config_dir = config_path_obj.parent

    logger.info(f"対象の設定ファイル: {config_path_obj}")

    # 設定ファイルの存在確認とディレクトリ作成
    if not config_path_obj.exists():
        logger.warning(f"設定ファイルが見つかりません: {config_path_obj}")
        try:
            logger.info(f"設定ディレクトリを作成します: {config_dir}")
            config_dir.mkdir(parents=True, exist_ok=True)
            # 空の設定ファイルを作成
            with open(config_path_obj, "w", encoding="utf-8") as f:
                f.write("{}")
            logger.info(f"空の設定ファイルを作成しました: {config_path_obj}")
        except Exception as e:
            logger.error(f"設定ディレクトリまたは空ファイルの作成に失敗しました: {e}")
            return False
    elif not os.access(config_path_obj, os.W_OK) or not os.access(config_dir, os.W_OK):
        logger.error(
            f"設定ファイルまたはディレクトリへの書き込み権限がありません: {config_path_obj}"
        )
        return False

    # 現在の設定をバックアップ
    backup_path = config_dir / f"config.json.backup_{int(time.time())}"
    try:
        if config_path_obj.exists():
            shutil.copy2(config_path_obj, backup_path)
            logger.info(f"設定ファイルをバックアップしました: {backup_path}")
    except Exception as e:
        logger.error(f"バックアップ作成に失敗しました: {e}")
        return False

    # 現在の設定を読み込む
    current_config = {}
    try:
        if config_path_obj.exists() and config_path_obj.stat().st_size > 0:
            with open(config_path_obj, "r", encoding="utf-8") as f:
                current_config = json.load(f)
        else:
            logger.warning("設定ファイルが存在しないか空です。新しい設定で開始します。")
    except json.JSONDecodeError:
        logger.warning(
            "設定ファイルが有効なJSONではありません。新しい設定で開始します。"
        )
    except Exception as e:
        logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
        return False

    # MIREX プロジェクトパス
    mirex_path_str = str(MIREX_PROJECT_ROOT)
    logger.info(f"MIREXプロジェクトパス: {mirex_path_str}")

    # 設定内容を更新
    try:
        new_config = update_claude_config(current_config, mirex_path_str)
    except Exception as e:
        logger.error(f"設定の更新中にエラーが発生しました: {e}", exc_info=True)
        return False

    # 設定を保存
    try:
        with open(config_path_obj, "w", encoding="utf-8") as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info("Claude設定を更新しました")
        return True
    except Exception as e:
        logger.error(f"設定ファイルの保存に失敗しました: {e}")
        # バックアップから復元
        try:
            if backup_path.exists():
                shutil.copy2(backup_path, config_path_obj)
                logger.info("バックアップから設定を復元しました")
        except Exception as restore_error:
            logger.error(f"バックアップからの復元に失敗しました: {restore_error}")
        return False


def update_claude_config(
    current_config: Dict[str, Any], mirex_path_str: str
) -> Dict[str, Any]:
    """Claudeの設定を更新し、MIREXプロジェクト用の設定を追加"""
    config = json.loads(json.dumps(current_config))  # ディープコピー

    # --- 1. プロンプトテンプレートの設定 ---
    setup_prompt_templates(config, mirex_path_str)

    # --- 2. チート機能の設定 (削除) ---
    # setup_cheat_functions(config, mirex_path_str) # MCP v1.5+ では不要

    # --- 3. UI設定 ---
    config["ui"] = config.get("ui", {})
    config["ui"]["showAdvancedSettings"] = True
    config["ui"]["showModelPicker"] = True
    config["ui"]["theme"] = config["ui"].get("theme", "system")  # 既存設定を尊重

    # --- 4. アプリ設定 ---
    config["app"] = config.get("app", {})
    config["app"]["clearTempFolderOnStartup"] = False  # デフォルトでクリアしない

    # --- 5. カスタム指示の設定 ---
    setup_custom_instructions(config, mirex_path_str)

    # --- 6. プロジェクト固有設定 (tool設定は削除) ---
    # setup_project_specific_config(config, mirex_path_str) # tool["mcp"] は不要
    # workflows と resources は Desktop 固有機能として維持 (必要なら)
    config["workflows"] = config.get("workflows", {})  # workflowsセクションの存在を確認
    setup_workflows(
        config["workflows"], mirex_path_str
    )  # ワークフロー設定関数を呼び出し

    config["resources"] = config.get("resources", {})  # resourcesセクションの存在を確認
    setup_resources(config["resources"], mirex_path_str)  # リソース設定関数を呼び出し

    # --- 7. MCPサーバー設定 ---
    setup_mcp_servers(config, mirex_path_str)

    return config


def setup_prompt_templates(config: Dict[str, Any], mirex_path_str: str):
    """プロンプトテンプレートの設定"""
    if "promptTemplates" not in config:
        config["promptTemplates"] = []
    existing_templates = {
        template.get("name"): template for template in config["promptTemplates"]
    }
    # mirex_templates リストの定義 (変更なし、前回のコードを参照)
    mirex_templates = [
        {
            "name": "MIREX開発モード",
            "description": "MIREXプロジェクトの開発者モード",
            "isDefault": True,
            "text": f"MIREXアルゴリズム開発プロジェクトについての質問に答えてください。\nあなたは音楽情報抽出技術の専門家です。\nMIR（Music Information Retrieval）の技術について詳しく説明してください。\n\n作業ディレクトリ: {mirex_path_str}",
        },
        {
            "name": "MIREX検出器改善",
            "description": "検出器アルゴリズムの改善依頼用",
            "text": f"以下の音楽検出器アルゴリズムを改善したいと思います。\n問題点を分析して、具体的な改善案を提案してください。\n特に、[問題点]のセクションに対処する方法を教えてください。\n\n作業ディレクトリ: {mirex_path_str}\n\n検出器名: {{detector_name}}\n\n問題点:\n{{problem_description}}\n\n現在のコード:\n```python\n{{current_code}}\n```",
        },
        {
            "name": "MIREX評価結果分析",
            "description": "検出器の評価結果を分析",
            "text": f"以下の音楽検出器の評価結果を分析し、パフォーマンスの問題点と改善案を提案してください。\n評価メトリクスの意味と、それらから読み取れる問題点を詳しく説明してください。\n\n作業ディレクトリ: {mirex_path_str}\n\n検出器名: {{detector_name}}\n\n評価結果:\n```\n{{evaluation_result}}\n```",
        },
        {
            "name": "音楽信号処理Q&A",
            "description": "音楽信号処理に関する質問",
            "text": "音楽信号処理、特に以下の質問について専門的に回答してください:\n\n質問: {{question}}\n\n具体的なコード例やアルゴリズムの説明もあると助かります。",
        },
        {
            "name": "グリッドサーチ結果分析",
            "description": "パラメータグリッドサーチ結果の分析",
            "text": f"以下の検出器パラメータのグリッドサーチ結果を分析してください。\n最適なパラメータ組み合わせとその理由、およびパラメータ変化によるパフォーマンスへの影響を説明してください。\n\n作業ディレクトリ: {mirex_path_str}\n\n検出器名: {{detector_name}}\n\nグリッドサーチ結果:\n```\n{{grid_search_result}}\n```",
        },
        # 他のテンプレートも同様に追加...
    ]
    new_templates = []
    processed_names = set()
    for template in mirex_templates:
        name = template["name"]
        if name in existing_templates:
            existing = existing_templates[name]
            existing["text"] = template["text"]
            existing["description"] = template["description"]
            if "isDefault" in template:
                existing["isDefault"] = template["isDefault"]
            new_templates.append(existing)
        else:
            new_templates.append(template)
        processed_names.add(name)
    for name, template in existing_templates.items():
        if name not in processed_names:
            new_templates.append(template)
    config["promptTemplates"] = new_templates
    logger.info("プロンプトテンプレートを設定しました。")


# def setup_cheat_functions(...): # この関数は削除


def setup_custom_instructions(config: Dict[str, Any], mirex_path_str: str):
    """カスタム指示の設定"""
    if "customInstructions" not in config:
        config["customInstructions"] = {}
    # 修正: コマンド例をMCPツール呼び出し形式に
    custom_instructions = f"""
# MIREX アルゴリズム開発プロジェクト

**作業ディレクトリ:** {mirex_path_str}

## 主要ツール一覧 (MCPサーバー接続時に利用可能)

- `@tool health_check()`: サーバー動作確認
- `@tool start_session(base_algorithm="検出器名")`: 改善セッション開始
- `@tool get_session_info(session_id="...")`: セッション情報取得
- `@tool add_session_history(...)`: セッション履歴追加 (主に内部用)
- `@tool get_job_status(job_id="...")`: 非同期ジョブ状態確認
- `@tool get_code(detector_name="...")`: 検出器コード取得
- `@tool save_code(detector_name="...", code="...", version="...")`: 改善コード保存
- `@tool improve_code(prompt="...", session_id="...")`: (LLM) コード改善依頼
- `@tool run_evaluation(detector_name="...", dataset_name="...")`: 評価実行
- `@tool run_grid_search(config_path="...")`: グリッドサーチ実行
- `@tool analyze_evaluation_results(evaluation_results_json="...")`: (LLM) 評価結果分析
- `@tool generate_hypotheses(session_id="...")`: (LLM) 仮説生成
- `@tool analyze_code_segment(code_segment="...", question="...")`: (LLM) コード断片分析
- `@tool suggest_parameters(source_code="...", context="...")`: (LLM) パラメータ提案
- `@tool optimize_parameters(detector_name="...", auto_suggest=True)`: パラメータ最適化
- `@tool suggest_exploration_strategy(session_id="...", current_performance=0.8)`: 次の戦略提案
- `@tool visualize_improvement_trajectory(session_id="...")`: 改善軌跡グラフ生成

ワークフローの例は `@prompt workflow-guide` を参照してください。
""".strip()
    config["customInstructions"][
        "text"
    ] = custom_instructions  # 常に上書きする形でシンプル化
    config["customInstructions"]["alwaysEnabled"] = True
    logger.info("カスタム指示を設定しました。")


# def setup_project_specific_config(...): # tools["mcp"] は不要になったため、この関数を分割
def setup_workflows(workflows_config: Dict[str, Any], mirex_path_str: str):
    """ワークフロー設定 (Claude Desktop固有機能の可能性)"""
    # MIREXワークフロー定義 (変更なし)
    workflows_config["mirex_improve_algorithm"] = {
        "name": "アルゴリズム改善",
        "description": "音楽情報検索アルゴリズムの自動改善ワークフロー",
        "steps": [
            {
                "id": "start_session",
                "name": "セッション開始",
                "description": "改善セッションを開始します",
                "tool": "mcp",
                "toolName": "start_session",
                "parameters": {"base_algorithm": "{detector_name}"},
            },
            {
                "id": "get_code",
                "name": "コード取得",
                "description": "検出器のコードを取得します",
                "tool": "mcp",
                "toolName": "get_code",
                "parameters": {"detector_name": "{detector_name}"},
            },
            {
                "id": "run_evaluation",
                "name": "初期評価",
                "description": "検出器の初期評価を実行します",
                "tool": "mcp",
                "toolName": "run_evaluation",
                "parameters": {"detector_name": "{detector_name}"},
            },
            {
                "id": "improve_code",
                "name": "コード改善",
                "description": "AIを使ってコードを改善します",
                "tool": "mcp",
                "toolName": "improve_code",
                "parameters": {
                    "prompt": "以下の検出器を改善してください: {improvement_goal}",
                    "session_id": "{session_id}",
                },
            },
            {
                "id": "evaluate_improved",
                "name": "改善評価",
                "description": "改善されたコードを評価します",
                "tool": "mcp",
                "toolName": "run_evaluation",
                "parameters": {"detector_name": "{detector_name}_improved"},
            },
        ],
    }
    logger.info("ワークフロー設定を更新しました。")


def setup_resources(resources_config: Dict[str, Any], mirex_path_str: str):
    """リソース設定 (Claude Desktop固有機能の可能性)"""
    # MIREXリソース定義 (変更なし)
    resources_config["mirex"] = {
        "detectors": f"{mirex_path_str}/src/detectors",
        "evaluation_results": f"{mirex_path_str}/evaluation_results",
        "grid_search_results": f"{mirex_path_str}/grid_search_results",
    }
    logger.info("リソース設定を更新しました。")


def setup_mcp_servers(config: Dict[str, Any], mirex_path_str: str):
    """MCPサーバー設定を追加/更新"""
    # Python実行ファイルのパスを決定 (仮想環境優先)
    venv_python_path = Path(mirex_path_str) / "venv" / "bin" / "python"  # Linux/macOS
    venv_python_path_win = (
        Path(mirex_path_str) / "venv" / "Scripts" / "python.exe"
    )  # Windows
    python_executable = sys.executable  # デフォルトは現在のPython

    if venv_python_path.exists():
        python_executable = str(venv_python_path)
    elif venv_python_path_win.exists():
        python_executable = str(venv_python_path_win)
    else:
        logger.warning(
            f"仮想環境のPythonが見つかりません ({venv_python_path} or {venv_python_path_win})。現在のPython ({sys.executable}) を使用します。"
        )

    # デフォルトのMCPポート
    mcp_port = os.environ.get("MCP_PORT", "5002")

    # メインワークスペースパス
    main_workspace = os.environ.get(
        "MIREX_WORKSPACE", str(MIREX_PROJECT_ROOT / "mcp_workspace")
    )  # Pathを文字列に

    # MCPサーバー設定
    config["mcpServers"] = config.get("mcpServers", {})  # 既存設定を保持
    config["mcpServers"]["mirex-auto-improver"] = {  # サーバー名を指定
        "command": python_executable,
        "args": [
            str(MIREX_PROJECT_ROOT / "mcp_server.py"),  # Pathを文字列に
            "--port",
            mcp_port,
            # 必要に応じて他のサーバー引数を追加 (e.g., --log-level)
        ],
        "cwd": mirex_path_str,  # サーバーの作業ディレクトリをプロジェクトルートに設定
        "env": {
            "ANYIO_BACKEND": "asyncio",  # FastMCPが必要とする場合がある
            "MIREX_WORKSPACE": main_workspace,  # サーバーが参照するワークスペース
            "PYTHONPATH": f"{str(MIREX_PROJECT_ROOT / 'src')}{os.pathsep}{str(MIREX_PROJECT_ROOT)}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",  # srcとプロジェクトルートをPYTHONPATHに追加
            # MCP_SERVER_URL は不要
            # APIキーは .env から読み込まれる想定
            "ANTHROPIC_API_KEY": os.environ.get(
                "ANTHROPIC_API_KEY", ""
            ),  # 念のため渡す
        },
        "enable": True,  # サーバーを有効にする
    }
    # "mirex-improver" (代替) の設定は必要であれば残すが、通常は1つで良い場合が多い
    # if "mirex-improver" in config["mcpServers"]: del config["mcpServers"]["mirex-improver"]

    logger.info(
        f"MCPサーバー設定を更新/追加しました: mirex-auto-improver (Port: {mcp_port})"
    )
    logger.info(f"  Python Executable: {python_executable}")
    logger.info(f"  Workspace: {main_workspace}")


def open_claude_desktop():
    """Claude Desktopアプリを起動"""
    logger.info("Claude Desktopを起動中...")
    try:
        if platform.system() == "Darwin":
            subprocess.run(["open", "-a", "Claude"], check=True)
        elif platform.system() == "Windows":
            # Windowsでの正確な起動方法は環境による可能性あり
            subprocess.run(["start", "", "Claude"], shell=True, check=True)
        elif platform.system() == "Linux":
            subprocess.run(
                ["xdg-open", "claude://"], check=True
            )  # claude:// URI スキームを試す
        else:
            logger.error(f"未対応のOS: {platform.system()}")
            return False
        logger.info("Claude Desktopを起動しました（または起動試行しました）")
        return True
    except FileNotFoundError:
        logger.error(
            "Claude Desktopアプリケーションが見つかりません。パスが通っているか確認してください。"
        )
        return False
    except Exception as e:
        logger.error(f"Claude Desktopの起動中にエラーが発生しました: {e}")
        return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Claude Desktop設定ツール")
    parser.add_argument(
        "--open-claude", action="store_true", help="設定後にClaudeを起動"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help=f"Claude設定ファイル(config.json)のパス (デフォルト: {DEFAULT_CONFIG_FILE or 'OS依存'})",
    )
    args = parser.parse_args()

    success = setup_claude_desktop_config(config_path_override=args.config_path)

    if success and args.open_claude:
        open_claude_desktop()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
