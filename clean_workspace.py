#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ワークスペースクリーニングツール

このスクリプトは以下のディレクトリや一時ファイルをクリーンアップします：
- evaluation_results: 評価結果の保存先
- grid_search_results: グリッドサーチ結果の保存先
- mcp_workspace: MCPワークスペース（セッションデータなど）
- home_workspace: ホームディレクトリのワークスペース

使用例:
    # すべてのディレクトリをクリーンアップ
    python clean_workspace.py --all
    
    # 評価結果のみをクリーンアップ
    python clean_workspace.py --evaluation-results
    
    # 7日以上経過したファイルのみクリーンアップ
    python clean_workspace.py --older-than 7
    
    # バックアップを作成してからクリーンアップ
    python clean_workspace.py --all --backup
"""

import os
import sys
import shutil
import argparse
import logging
import time
import glob
import datetime
import json
import zipfile
import traceback

# プロジェクト内モジュールをインポート
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # src.utils.path_utils モジュールをインポート
    from src.utils.path_utils import get_project_root, get_workspace_dir, get_evaluation_results_dir, get_grid_search_results_dir, get_improved_versions_dir
    use_path_utils = True
except ImportError:
    # インポートに失敗した場合は元のコード実装を使用
    use_path_utils = False

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('clean_workspace')

if not use_path_utils:
    # path_utilsが使えない場合の代替実装
    logger.warning("src.utils.path_utils のインポートに失敗しました。内部実装を使用します。")
    
    def get_project_root():
        """プロジェクトのルートディレクトリを特定します"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = script_dir
        
        # 最大5階層まで遡る
        for _ in range(5):
            # プロジェクトのマーカーファイルが存在するか確認
            if os.path.exists(os.path.join(current_dir, 'mcp_server.py')) or \
               os.path.exists(os.path.join(current_dir, 'src', 'detectors')):
                return current_dir
            
            # 親ディレクトリへ移動
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # ルートディレクトリに到達した場合
                break
            current_dir = parent_dir
        
        # マーカーが見つからない場合はスクリプトのディレクトリを返す
        return script_dir

def get_workspace_dirs():
    """クリーンアップ対象のディレクトリパスを取得します"""
    if use_path_utils:
        # src.utils.path_utils の関数を使用
        dirs = {
            'evaluation_results': str(get_evaluation_results_dir()),
            'grid_search_results': str(get_grid_search_results_dir()),
            'mcp_workspace': str(get_workspace_dir()),
            'improved_versions': str(get_improved_versions_dir()),
            'home_workspace': os.path.join(os.path.expanduser("~"), '.mcp_workspace')
        }
    else:
        # path_utils が使えない場合は従来の実装
        project_root = get_project_root()
        home_dir = os.path.expanduser("~")
        
        dirs = {
            'evaluation_results': os.path.join(project_root, 'evaluation_results'),
            'grid_search_results': os.path.join(project_root, 'grid_search_results'),
            'mcp_workspace': os.path.join(project_root, 'mcp_workspace'),
            'improved_versions': os.path.join(project_root, 'improved_versions'),
            'home_workspace': os.path.join(home_dir, '.mcp_workspace')
        }
    
    # 環境変数の確認
    if 'MIREX_WORKSPACE' in os.environ:
        mirex_workspace = os.environ['MIREX_WORKSPACE']
        dirs['mirex_workspace'] = mirex_workspace
    
    return dirs

def is_older_than(file_path, days):
    """ファイルが指定日数よりも古いかどうかを判定します"""
    if not os.path.exists(file_path):
        return False
    
    file_time = os.path.getmtime(file_path)
    current_time = time.time()
    return (current_time - file_time) > (days * 24 * 60 * 60)

def create_backup(target_dirs, backup_dir=None):
    """指定したディレクトリのバックアップを作成します"""
    if not backup_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(get_project_root(), f'backups_{timestamp}')
    
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    logger.info(f"バックアップを作成しています: {backup_dir}")
    
    backup_info = {}
    
    for name, path in target_dirs.items():
        if os.path.exists(path):
            try:
                # ディレクトリ別のZIPファイルを作成
                zip_path = os.path.join(backup_dir, f'{name}_backup.zip')
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, path)
                            zipf.write(file_path, arcname)
                
                backup_info[name] = {
                    'original_path': path,
                    'backup_path': zip_path,
                    'size': os.path.getsize(zip_path),
                    'timestamp': time.time()
                }
                logger.info(f"ディレクトリ {name} のバックアップを作成しました: {zip_path}")
            
            except Exception as e:
                logger.error(f"ディレクトリ {name} のバックアップ作成中にエラーが発生しました: {str(e)}")
                logger.error(traceback.format_exc())
    
    # バックアップ情報を保存
    info_path = os.path.join(backup_dir, 'backup_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(backup_info, f, indent=2, ensure_ascii=False)
    
    return backup_dir

def clean_directory(directory, older_than_days=None, dry_run=False):
    """指定したディレクトリをクリーンアップします"""
    if not os.path.exists(directory):
        logger.info(f"ディレクトリが存在しません: {directory}")
        return False
    
    logger.info(f"ディレクトリをクリーンアップしています: {directory}")
    
    try:
        # 古いファイルのみを削除する場合
        if older_than_days is not None:
            count = 0
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    if is_older_than(file_path, older_than_days):
                        if not dry_run:
                            os.remove(file_path)
                        logger.debug(f"ファイルを削除しました: {file_path}")
                        count += 1
                
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    # ディレクトリが空かどうかを確認
                    if not os.listdir(dir_path):
                        if not dry_run:
                            os.rmdir(dir_path)
                        logger.debug(f"空のディレクトリを削除しました: {dir_path}")
            
            logger.info(f"{count}個のファイルを削除しました（{older_than_days}日以上経過したもの）")
        
        # ディレクトリ全体を削除
        else:
            if not dry_run:
                # ディレクトリ内のすべてのコンテンツを削除
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                
                # ディレクトリ自体は保持
                logger.info(f"ディレクトリの内容をすべて削除しました: {directory}")
            else:
                logger.info(f"[ドライラン] ディレクトリの内容をすべて削除: {directory}")
        
        return True
    
    except Exception as e:
        logger.error(f"クリーンアップ中にエラーが発生しました: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description='ワークスペースクリーニングツール')
    
    # クリーンアップの対象を指定するオプション
    target_group = parser.add_argument_group('クリーンアップ対象')
    target_group.add_argument('--all', action='store_true', help='すべてのディレクトリをクリーンアップ')
    target_group.add_argument('--evaluation-results', action='store_true', help='評価結果をクリーンアップ')
    target_group.add_argument('--grid-search-results', action='store_true', help='グリッドサーチ結果をクリーンアップ')
    target_group.add_argument('--mcp-workspace', action='store_true', help='MCPワークスペースをクリーンアップ')
    target_group.add_argument('--improved-versions', action='store_true', help='改善バージョンをクリーンアップ')
    target_group.add_argument('--home-workspace', action='store_true', help='ホームディレクトリのワークスペースをクリーンアップ')
    
    # その他のオプション
    parser.add_argument('--older-than', type=int, help='指定日数より古いファイルのみをクリーンアップ')
    parser.add_argument('--backup', action='store_true', help='クリーンアップ前にバックアップを作成')
    parser.add_argument('--dry-run', action='store_true', help='ドライラン（実際には削除しない）')
    parser.add_argument('--verbose', action='store_true', help='詳細なログを表示')
    
    args = parser.parse_args()
    
    # 詳細ログの設定
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # ワークスペースディレクトリの取得
    workspace_dirs = get_workspace_dirs()
    
    # クリーンアップ対象の決定
    target_dirs = {}
    if args.all:
        target_dirs = workspace_dirs
    else:
        if args.evaluation_results and 'evaluation_results' in workspace_dirs:
            target_dirs['evaluation_results'] = workspace_dirs['evaluation_results']
        if args.grid_search_results and 'grid_search_results' in workspace_dirs:
            target_dirs['grid_search_results'] = workspace_dirs['grid_search_results']
        if args.mcp_workspace and 'mcp_workspace' in workspace_dirs:
            target_dirs['mcp_workspace'] = workspace_dirs['mcp_workspace']
        if args.improved_versions and 'improved_versions' in workspace_dirs:
            target_dirs['improved_versions'] = workspace_dirs['improved_versions']
        if args.home_workspace and 'home_workspace' in workspace_dirs:
            target_dirs['home_workspace'] = workspace_dirs['home_workspace']
    
    # 対象が指定されていない場合は終了
    if not target_dirs:
        parser.print_help()
        logger.error("クリーンアップ対象が指定されていません。--allまたは個別のディレクトリオプションを指定してください。")
        return 1
    
    logger.info("===== ワークスペースクリーニングツール =====")
    logger.info(f"プロジェクトルート: {get_project_root()}")
    
    # バックアップの作成
    if args.backup:
        backup_dir = create_backup(target_dirs)
        logger.info(f"バックアップを作成しました: {backup_dir}")
    
    # クリーンアップの実行
    success_count = 0
    for name, path in target_dirs.items():
        logger.info(f"--- {name} のクリーンアップを開始 ---")
        if clean_directory(path, args.older_than, args.dry_run):
            success_count += 1
    
    # 結果の表示
    if args.dry_run:
        logger.info(f"ドライラン完了: {len(target_dirs)}個のディレクトリが対象でした")
    else:
        logger.info(f"クリーンアップ完了: {success_count}/{len(target_dirs)}個のディレクトリを処理しました")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 