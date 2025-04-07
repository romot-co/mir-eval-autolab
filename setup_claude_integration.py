#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Desktop設定ツール

Claude DesktopアプリにMIREXプロジェクトの設定を自動的に適用します。
Claude DesktopアプリでMIREXの機能を使いやすくするための環境設定を行います。

使用方法:
python setup_claude_integration.py --open-claude  # 設定後にClaude Desktopを起動
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('claude_setup')

# --- 定数 ---
# 環境変数からパスを取得（設定されていない場合はデフォルト値）
CLAUDE_CONFIG_PATH = os.environ.get(
    "CLAUDE_CONFIG_PATH",
    os.path.expanduser("~/Library/Application Support/Claude/storage/config.json")
)
TEMPLATE_DIR = os.environ.get("MIREX_TEMPLATE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))

# --- メイン関数 ---
def setup_claude_desktop_config():
    """Claude Desktopの設定ファイルを更新"""
    logger.info("Claude Desktop設定ツールを開始します")
    
    # 設定ファイルの存在確認
    if not os.path.exists(CLAUDE_CONFIG_PATH):
        create_dir = input(f"Claude設定ファイルが見つかりません: {CLAUDE_CONFIG_PATH}\nディレクトリを作成しますか？ (y/n): ")
        if create_dir.lower() == 'y':
            os.makedirs(os.path.dirname(CLAUDE_CONFIG_PATH), exist_ok=True)
            # 空の設定ファイルを作成
            with open(CLAUDE_CONFIG_PATH, 'w') as f:
                f.write('{}')
            logger.info(f"空の設定ファイルを作成しました: {CLAUDE_CONFIG_PATH}")
        else:
            logger.error("設定ファイルがないため、処理を中止します")
            return False
    
    # 現在の設定をバックアップ
    backup_path = f"{CLAUDE_CONFIG_PATH}.backup_{int(time.time())}"
    try:
        shutil.copy2(CLAUDE_CONFIG_PATH, backup_path)
        logger.info(f"設定ファイルをバックアップしました: {backup_path}")
    except Exception as e:
        logger.error(f"バックアップ作成に失敗しました: {e}")
        return False
    
    # 現在の設定を読み込む
    try:
        with open(CLAUDE_CONFIG_PATH, 'r') as f:
            current_config = json.load(f)
    except json.JSONDecodeError:
        logger.warning("設定ファイルが空か、有効なJSONではありません。新しい設定で開始します。")
        current_config = {}
    except Exception as e:
        logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
        return False
    
    # MIREX プロジェクトパスを取得
    mirex_path = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"MIREXプロジェクトパス: {mirex_path}")
    
    # 更新する設定
    new_config = update_claude_config(current_config, mirex_path)
    
    # 設定を保存
    try:
        with open(CLAUDE_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info("Claude設定を更新しました")
        return True
    except Exception as e:
        logger.error(f"設定ファイルの保存に失敗しました: {e}")
        # バックアップから復元
        try:
            shutil.copy2(backup_path, CLAUDE_CONFIG_PATH)
            logger.info("バックアップから設定を復元しました")
        except Exception as restore_error:
            logger.error(f"バックアップからの復元に失敗しました: {restore_error}")
        return False

def update_claude_config(current_config: Dict[str, Any], mirex_path: str) -> Dict[str, Any]:
    """Claudeの設定を更新し、MIREXプロジェクト用の設定を追加"""
    # 新しい設定を作成 (ディープコピー)
    config = json.loads(json.dumps(current_config))
    
    # プロンプトテンプレートの設定
    setup_prompt_templates(config, mirex_path)
    
    # チート機能の設定
    setup_cheat_functions(config, mirex_path)
    
    # UI設定
    if "ui" not in config:
        config["ui"] = {}
    
    # 高度な設定を表示
    config["ui"]["showAdvancedSettings"] = True
    config["ui"]["showModelPicker"] = True
    
    # 追加のUIカスタマイズ（テーマなど）
    config["ui"]["theme"] = "system"  # "light", "dark", "system"のいずれか
    
    # tempディレクトリのクリア不要
    if "app" not in config:
        config["app"] = {}
    config["app"]["clearTempFolderOnStartup"] = False
    
    # カスタム説明の追加
    setup_custom_instructions(config, mirex_path)
    
    # プロジェクト固有の設定
    setup_project_specific_config(config, mirex_path)
    
    # MCPサーバー設定
    setup_mcp_servers(config, mirex_path)
    
    return config

def setup_prompt_templates(config: Dict[str, Any], mirex_path: str):
    """プロンプトテンプレートの設定"""
    if "promptTemplates" not in config:
        config["promptTemplates"] = []
    
    # 既存のテンプレートを保持
    existing_templates = {template.get("name"): template for template in config["promptTemplates"]}
    
    # MIREXプロジェクト用のテンプレート
    mirex_templates = [
        {
            "name": "MIREX開発モード",
            "description": "MIREXプロジェクトの開発者モード",
            "isDefault": True,
            "text": f"""
MIREXアルゴリズム開発プロジェクトについての質問に答えてください。
あなたは音楽情報抽出技術の専門家です。
MIR（Music Information Retrieval）の技術について詳しく説明してください。

作業ディレクトリ: {mirex_path}
""".strip()
        },
        {
            "name": "MIREX検出器改善",
            "description": "検出器アルゴリズムの改善依頼用",
            "text": f"""
以下の音楽検出器アルゴリズムを改善したいと思います。
問題点を分析して、具体的な改善案を提案してください。
特に、[問題点]のセクションに対処する方法を教えてください。

作業ディレクトリ: {mirex_path}

検出器名: {{detector_name}}

問題点:
{{problem_description}}

現在のコード:
```python
{{current_code}}
```
""".strip()
        },
        {
            "name": "MIREX評価結果分析",
            "description": "検出器の評価結果を分析",
            "text": f"""
以下の音楽検出器の評価結果を分析し、パフォーマンスの問題点と改善案を提案してください。
評価メトリクスの意味と、それらから読み取れる問題点を詳しく説明してください。

作業ディレクトリ: {mirex_path}

検出器名: {{detector_name}}

評価結果:
```
{{evaluation_result}}
```
""".strip()
        },
        {
            "name": "音楽信号処理Q&A",
            "description": "音楽信号処理に関する質問",
            "text": """
音楽信号処理、特に以下の質問について専門的に回答してください:

質問: {{question}}

具体的なコード例やアルゴリズムの説明もあると助かります。
""".strip()
        },
        {
            "name": "グリッドサーチ結果分析",
            "description": "パラメータグリッドサーチ結果の分析",
            "text": f"""
以下の検出器パラメータのグリッドサーチ結果を分析してください。
最適なパラメータ組み合わせとその理由、およびパラメータ変化によるパフォーマンスへの影響を説明してください。

作業ディレクトリ: {mirex_path}

検出器名: {{detector_name}}

グリッドサーチ結果:
```
{{grid_search_result}}
```
""".strip()
        },
        {
            "name": "AUGコード生成",
            "description": "Audio Utility Graph生成・解析ツール",
            "text": f"""
以下の音楽処理タスクのためのAudio Utility Graph (AUG)を設計してください。
グラフ形式で表現された音声処理パイプラインを考え、mermaid形式で出力してください。

作業ディレクトリ: {mirex_path}

タスク: {{task_description}}

次のような出力が必要です:
```mermaid
graph TD
    A[Input Audio] --> B[前処理]
    B --> C[特徴抽出]
    C --> D[分析]
    D --> E[出力]
```

さらに、各ノードの詳細な説明も提供してください。
""".strip()
        },
        {
            "name": "PyTorch改善案",
            "description": "PyTorchモデル最適化アドバイス",
            "text": """
以下のPyTorchコードを分析し、パフォーマンスと精度を改善するための提案をしてください。
特にモデルアーキテクチャ、損失関数、最適化手法に焦点を当ててください。

```python
{{pytorch_code}}
```

特に、次の点について改善案を提供してください:
1. モデル構造の最適化
2. バッチ処理とデータローディングの効率化
3. 学習率とオプティマイザの設定
4. 正則化技術
5. その他のパフォーマンス向上テクニック
""".strip()
        },
        {
            "name": "論文レビュー",
            "description": "MIR研究論文のレビュー",
            "text": """
以下のMIR（音楽情報検索）研究論文の要約とクリティカルレビューを提供してください。

論文タイトル: {{paper_title}}
著者: {{paper_authors}}

主要な貢献、手法、結果に焦点を当て、強みと弱み、および将来の研究方向を特定してください。

論文の概要:
{{paper_abstract}}
""".strip()
        },
        {
            "name": "バグ修正ヘルパー",
            "description": "コードバグを特定して修正",
            "text": f"""
以下のコードに含まれるバグを特定し、修正案を提示してください。
エラーメッセージや予期しない動作も記載します。

作業ディレクトリ: {mirex_path}

バグのある部分:
```python
{{buggy_code}}
```

エラーまたは問題の説明:
{{error_description}}
""".strip()
        },
        {
            "name": "テスト生成",
            "description": "ユニットテストとテストケース生成",
            "text": f"""
以下のコードに対するテストケースとユニットテストを生成してください。
境界条件や特殊なケースも含めて、網羅的なテストを提案してください。

作業ディレクトリ: {mirex_path}

テスト対象のコード:
```python
{{code_to_test}}
```

テストするべき特定の観点:
{{test_focus}}
""".strip()
        }
    ]
    
    # テンプレートを更新または追加
    new_templates = []
    for template in mirex_templates:
        # 既存のテンプレートがあれば保持し、新しい内容で更新
        if template["name"] in existing_templates:
            existing = existing_templates[template["name"]]
            # 基本情報はそのまま残し、テキスト内容を更新
            existing["text"] = template["text"]
            existing["description"] = template["description"]
            # isDefaultが指定されている場合は更新
            if "isDefault" in template:
                existing["isDefault"] = template["isDefault"]
            new_templates.append(existing)
        else:
            # 新規追加
            new_templates.append(template)
    
    # 他の既存テンプレートも残して追加
    for name, template in existing_templates.items():
        if not any(t["name"] == name for t in new_templates):
            new_templates.append(template)
    
    # 更新したテンプレートを設定
    config["promptTemplates"] = new_templates

def setup_cheat_functions(config: Dict[str, Any], mirex_path: str):
    """チート機能（外部ツール）の設定"""
    if "cheatFunctions" not in config:
        config["cheatFunctions"] = []
    
    # 既存のチート機能を保持
    existing_functions = {func.get("name"): func for func in config["cheatFunctions"]}
    
    # デフォルトのMCPポート
    mcp_port = os.environ.get("MCP_PORT", "5002")
    
    # MIREXプロジェクト用のチート機能
    mirex_functions = [
        {
            "name": "start_session",
            "description": "改善セッションを開始します",
            "parameters": {
                "base_algorithm": {
                    "description": "改善したいアルゴリズム名（例: PZSTDDetector）",
                    "type": "string",
                    "required": True
                }
            },
            "url": f"http://localhost:{mcp_port}/api/session/start",
            "method": "POST"
        },
        {
            "name": "get_code",
            "description": "検出器のソースコードを取得します",
            "parameters": {
                "detector_name": {
                    "description": "検出器名",
                    "type": "string",
                    "required": True
                }
            },
            "url": f"http://localhost:{mcp_port}/api/detector/code",
            "method": "GET"
        },
        {
            "name": "run_evaluation",
            "description": "検出器の評価を実行します",
            "parameters": {
                "detector_name": {
                    "description": "評価する検出器名",
                    "type": "string",
                    "required": True
                },
                "audio_dir": {
                    "description": "音声ファイルディレクトリ（省略時はデフォルト）",
                    "type": "string",
                    "required": False
                },
                "reference_dir": {
                    "description": "正解ラベルディレクトリ（省略時はデフォルト）",
                    "type": "string",
                    "required": False
                }
            },
            "url": f"http://localhost:{mcp_port}/api/evaluate",
            "method": "POST"
        },
        {
            "name": "improve_code",
            "description": "LLMを使用してコードを改善します",
            "parameters": {
                "prompt": {
                    "description": "LLMに送信するプロンプト",
                    "type": "string",
                    "required": True
                },
                "session_id": {
                    "description": "セッションID（省略可）",
                    "type": "string",
                    "required": False
                }
            },
            "url": f"http://localhost:{mcp_port}/api/improve",
            "method": "POST"
        },
        {
            "name": "run_grid_search",
            "description": "パラメータグリッドサーチを実行します",
            "parameters": {
                "detector_name": {
                    "description": "検出器名",
                    "type": "string",
                    "required": True
                },
                "grid_params_json": {
                    "description": "グリッドサーチパラメータ（JSON文字列）",
                    "type": "string",
                    "required": True
                },
                "audio_dir": {
                    "description": "音声ファイルディレクトリ（省略時はデフォルト）",
                    "type": "string",
                    "required": False
                },
                "reference_dir": {
                    "description": "正解ラベルディレクトリ（省略時はデフォルト）",
                    "type": "string",
                    "required": False
                }
            },
            "url": f"http://localhost:{mcp_port}/api/grid_search",
            "method": "POST"
        },
        {
            "name": "get_job_status",
            "description": "非同期ジョブのステータスを取得します",
            "parameters": {
                "job_id": {
                    "description": "ジョブID",
                    "type": "string",
                    "required": True
                }
            },
            "url": f"http://localhost:{mcp_port}/api/job/status/{{job_id}}",
            "method": "GET"
        }
    ]
    
    # チート機能を更新
    updated_functions = []
    
    # 既存の機能を保持（MIREXプロジェクト関連以外）
    for func in config["cheatFunctions"]:
        if not func.get("name").startswith("mirex_") and not any(func.get("name") == mirex_func.get("name") for mirex_func in mirex_functions):
            updated_functions.append(func)
    
    # 新しい機能を追加
    updated_functions.extend(mirex_functions)
    
    # 設定を更新
    config["cheatFunctions"] = updated_functions

def setup_custom_instructions(config: Dict[str, Any], mirex_path: str):
    """カスタム説明の設定"""
    if "customInstructions" not in config:
        config["customInstructions"] = {}
    
    # MIREXプロジェクトのカスタム説明
    custom_instructions = f"""
# MIREX アルゴリズム開発プロジェクト

**作業ディレクトリ:** {mirex_path}

## 主要コマンド一覧

- **run_evaluation**: 検出器の評価を実行
- **run_grid_search**: パラメータグリッドサーチを実行
- **run_improvement**: 検出器の自動改善を実行
- **view_code**: 検出器コードを表示
- **create_detector**: 新しい検出器を作成
- **list_detectors**: 利用可能な検出器一覧を表示
- **compare_detectors**: 2つの検出器を比較
- **open_mirex_tools**: MIREX関連ツールを一覧表示
- **run_tests**: テストを実行
- **explore_datasets**: データセット情報を表示
- **help_mirex**: MIREX使用方法を表示

引数付きでコマンドを使用：例 `run_evaluation PZSTDDetector`
""".strip()
    
    # 既存の説明を保持
    existing_instructions = config["customInstructions"].get("text", "")
    
    # MIREXの説明が含まれていなければ追加
    if "MIREX アルゴリズム開発プロジェクト" not in existing_instructions:
        if existing_instructions:
            config["customInstructions"]["text"] = existing_instructions + "\n\n" + custom_instructions
        else:
            config["customInstructions"]["text"] = custom_instructions
    
    # Always enabledにする
    config["customInstructions"]["alwaysEnabled"] = True

def setup_project_specific_config(config: Dict[str, Any], mirex_path: str):
    """プロジェクト固有の設定"""
    # デフォルトのMCPポート
    mcp_port = os.environ.get("MCP_PORT", "5002")
    
    # MCP設定
    if "tools" not in config:
        config["tools"] = {}
    
    # MCPツール設定
    config["tools"]["mcp"] = {
        "url": f"http://localhost:{mcp_port}",
        "auth": None
    }
    
    # ワークフロー設定
    if "workflows" not in config:
        config["workflows"] = {}
    
    # MIREXワークフロー
    config["workflows"]["mirex_improve_algorithm"] = {
        "name": "アルゴリズム改善",
        "description": "音楽情報検索アルゴリズムの自動改善ワークフロー",
        "steps": [
            {
                "id": "start_session",
                "name": "セッション開始",
                "description": "改善セッションを開始します",
                "tool": "mcp",
                "toolName": "start_session",
                "parameters": {
                    "base_algorithm": "{detector_name}"
                }
            },
            {
                "id": "get_code",
                "name": "コード取得",
                "description": "検出器のコードを取得します",
                "tool": "mcp",
                "toolName": "get_code",
                "parameters": {
                    "detector_name": "{detector_name}"
                }
            },
            {
                "id": "run_evaluation",
                "name": "初期評価",
                "description": "検出器の初期評価を実行します",
                "tool": "mcp",
                "toolName": "run_evaluation",
                "parameters": {
                    "detector_name": "{detector_name}"
                }
            },
            {
                "id": "improve_code",
                "name": "コード改善",
                "description": "AIを使ってコードを改善します",
                "tool": "mcp",
                "toolName": "improve_code",
                "parameters": {
                    "prompt": "以下の検出器を改善してください: {improvement_goal}",
                    "session_id": "{session_id}"
                }
            },
            {
                "id": "evaluate_improved",
                "name": "改善評価",
                "description": "改善されたコードを評価します",
                "tool": "mcp",
                "toolName": "run_evaluation",
                "parameters": {
                    "detector_name": "{detector_name}_improved"
                }
            }
        ]
    }
    
    # リソース設定
    if "resources" not in config:
        config["resources"] = {}
    
    # MIREXリソース
    config["resources"]["mirex"] = {
        "detectors": f"{mirex_path}/src/detectors",
        "evaluation_results": f"{mirex_path}/evaluation_results",
        "grid_search_results": f"{mirex_path}/grid_search_results"
    }

def setup_mcp_servers(config: Dict[str, Any], mirex_path: str):
    """MCPサーバー設定を追加"""
    # Python仮想環境のパス
    venv_python = os.environ.get("MIREX_VENV_PYTHON", os.path.join(mirex_path, "venv", "bin", "python"))
    
    # デフォルトのMCPポート
    mcp_port = os.environ.get("MCP_PORT", "5002")
    mcp_alt_port = os.environ.get("MCP_ALT_PORT", "5003")
    
    # メインワークスペースパス
    main_workspace = os.environ.get("MIREX_WORKSPACE", os.path.join(mirex_path, "mcp_workspace"))
    alt_workspace = os.environ.get("MIREX_ALT_WORKSPACE", os.path.expanduser("~/Documents/mirex_workspace"))
    
    # MCPサーバー設定
    config["mcpServers"] = {
        "mirex-auto-improver": {
            "command": venv_python,
            "args": [
                os.path.join(mirex_path, "mcp_server.py"),
                "--port",
                mcp_port
            ],
            "env": {
                "ANYIO_BACKEND": "asyncio",
                "MCP_SERVER_URL": f"http://localhost:{mcp_port}",
                "MIREX_WORKSPACE": main_workspace
            }
        },
        "mirex-improver": {
            "command": venv_python,
            "args": [
                os.path.join(mirex_path, "mcp_server.py"),
                "--port",
                mcp_alt_port
            ],
            "cwd": alt_workspace,
            "env": {
                "ANYIO_BACKEND": "asyncio",
                "MCP_SERVER_URL": f"http://localhost:{mcp_alt_port}",
                "MIREX_WORKSPACE": alt_workspace
            }
        }
    }
    
    logger.info(f"MCPサーバー設定を追加しました: メインポート={mcp_port}, 代替ポート={mcp_alt_port}")
    logger.info(f"ワークスペース: メイン={main_workspace}, 代替={alt_workspace}")

def open_claude_desktop():
    """Claude Desktopアプリを起動"""
    logger.info("Claude Desktopを起動中...")
    
    if platform.system() == "Darwin":  # macOS
        subprocess.run(["open", "-a", "Claude"])
    elif platform.system() == "Windows":
        subprocess.run(["start", "Claude"], shell=True)
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", "claude"])
    else:
        logger.error(f"未対応のOS: {platform.system()}")
        return False
    
    logger.info("Claude Desktopを起動しました")
    return True

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Claude Desktop設定ツール")
    parser.add_argument("--open-claude", action="store_true", help="設定後にClaudeを起動")
    parser.add_argument("--config-path", type=str, help="Claude設定ファイルのパス（デフォルト: ~/Library/Application Support/Claude/storage/config.json）")
    
    args = parser.parse_args()
    
    # 設定ファイルのパスを更新（引数で指定された場合）
    if args.config_path:
        global CLAUDE_CONFIG_PATH
        CLAUDE_CONFIG_PATH = args.config_path
    
    # Claude Desktopの設定を更新
    success = setup_claude_desktop_config()
    
    if success and args.open_claude:
        # 設定成功後、Claudeを起動
        open_claude_desktop()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
    
    # 使用方法の例
    # python setup_claude_integration.py                # 設定のみ
    # python setup_claude_integration.py --open-claude  # 設定後にClaude Desktopを起動 