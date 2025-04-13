#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIREXワークスペース管理スクリプト

.mcp_server_dataディレクトリの作成と消去を行います

使用例:
    # ワークスペースを作成
    python manage_workspace.py --create
    
    # ワークスペースを削除
    python manage_workspace.py --delete
    
    # ワークスペースをリセット
    python manage_workspace.py --reset
    
    # ワークスペース構造を表示
    python manage_workspace.py --show
"""

import os
import sys
import argparse
import shutil
from pathlib import Path


# ANSI色コード
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[0;33m'
    NC = '\033[0m'  # No Color


def get_project_root():
    """プロジェクトルートディレクトリを取得"""
    return Path(__file__).resolve().parent.parent


def get_workspace_dir():
    """ワークスペースディレクトリを取得"""
    return get_project_root() / '.mcp_server_data'


# サブディレクトリの構造定義
SUBDIRS = [
    "code/src/detectors",
    "code/src/templates",
    "code/improved_versions",
    "datasets/synthesized_v1/audio",
    "datasets/synthesized_v1/labels",
    "datasets/synthetic_v1/audio",
    "datasets/synthetic_v1/labels",
    "datasets/synthetic_poly_v1/audio",
    "datasets/synthetic_poly_v1/labels",
    "output/evaluation_results",
    "output/grid_search_results",
    "output/visualizations",
    "output/scientific_output",
    "db"
]


def create_workspace(workspace_dir):
    """ワークスペースディレクトリとサブディレクトリを作成"""
    if workspace_dir.exists():
        print(f"{Colors.YELLOW}ワークスペースはすでに存在します: {workspace_dir}{Colors.NC}")
        return
    
    print(f"{Colors.GREEN}ワークスペースを作成しています: {workspace_dir}{Colors.NC}")
    workspace_dir.mkdir(exist_ok=True, parents=True)
    
    for subdir in SUBDIRS:
        subdir_path = workspace_dir / subdir
        subdir_path.mkdir(exist_ok=True, parents=True)
        print(f"{Colors.GREEN}ディレクトリを作成しました: {subdir_path}{Colors.NC}")
    
    print(f"{Colors.GREEN}ワークスペースの作成が完了しました。{Colors.NC}")


def delete_workspace(workspace_dir, force=False):
    """ワークスペースディレクトリを削除"""
    if not workspace_dir.exists():
        print(f"{Colors.YELLOW}ワークスペースが存在しません: {workspace_dir}{Colors.NC}")
        return
    
    if not force:
        print(f"{Colors.RED}警告: 次のディレクトリを削除します: {workspace_dir}{Colors.NC}")
        response = input("続行しますか？ [y/N] ")
        if not response.lower() in ['y', 'yes']:
            print("中止しました。")
            return
    
    print(f"{Colors.RED}ワークスペースを削除しています: {workspace_dir}{Colors.NC}")
    shutil.rmtree(workspace_dir)
    print(f"{Colors.GREEN}ワークスペースの削除が完了しました。{Colors.NC}")


def show_workspace(workspace_dir):
    """ワークスペース構造を表示"""
    if not workspace_dir.exists():
        print(f"{Colors.YELLOW}ワークスペースが存在しません: {workspace_dir}{Colors.NC}")
        return
    
    print(f"{Colors.GREEN}ワークスペース構造:{Colors.NC}")
    
    # ディレクトリ構造を表示
    for path in sorted(workspace_dir.glob('**')):
        if path.is_dir():
            rel_path = path.relative_to(workspace_dir.parent)
            print(rel_path)


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='MIREXワークスペース管理スクリプト')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-c', '--create', action='store_true', help='ワークスペースディレクトリを作成')
    group.add_argument('-d', '--delete', action='store_true', help='ワークスペースディレクトリを削除')
    group.add_argument('-r', '--reset', action='store_true', help='ワークスペースディレクトリをリセット（削除して再作成）')
    group.add_argument('-s', '--show', action='store_true', help='現在のワークスペース構造を表示')
    parser.add_argument('--force', action='store_true', help='確認なしで実行する')
    
    return parser.parse_args()


def main():
    """メイン関数"""
    args = parse_args()
    workspace_dir = get_workspace_dir()
    
    if args.create:
        create_workspace(workspace_dir)
    elif args.delete:
        delete_workspace(workspace_dir, args.force)
    elif args.reset:
        delete_workspace(workspace_dir, args.force)
        if not workspace_dir.exists():  # 削除が成功した場合のみ作成
            create_workspace(workspace_dir)
    elif args.show:
        show_workspace(workspace_dir)


if __name__ == "__main__":
    main() 