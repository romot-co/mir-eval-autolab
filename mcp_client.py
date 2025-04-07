#!/usr/bin/env python3
"""
MCPサーバークライアントライブラリ

このモジュールは、MCPサーバーへのAPIリクエストを行うためのクライアントを提供します。
"""

import os
import sys
import json
import time
import requests
import logging
from typing import Dict, Any, List, Optional
from prettytable import PrettyTable
import argparse

# 環境変数からの設定取得
SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:5002")
DEFAULT_POLL_INTERVAL = int(os.environ.get("MCP_POLL_INTERVAL", "5"))  # seconds
DEFAULT_TIMEOUT = int(os.environ.get("MCP_REQUEST_TIMEOUT", "60"))  # 1分

class MCPClient:
    """MCPサーバーへのアクセスを提供するクライアントクラス"""
    
    def __init__(self, server_url=None):
        self.server_url = server_url or os.environ.get("MCP_SERVER_URL", SERVER_URL)
        self.session_id = None
        self.timeout = DEFAULT_TIMEOUT
        
    def create_session(self) -> str:
        """新しいセッションを作成し、セッションIDを返す"""
        endpoint = f"{self.server_url}/api/session/start"
        response = requests.post(endpoint, timeout=self.timeout)
        
        if response.status_code == 200:
            data = response.json()
            self.session_id = data.get('id')
            return self.session_id
        else:
            raise Exception(f"セッション作成エラー: {response.status_code} {response.text}")
    
    def poll_job_status(self, job_id: str, max_retries=30, interval=None) -> Dict[str, Any]:
        """非同期ジョブの完了を待機し、結果を返す"""
        if interval is None:
            interval = DEFAULT_POLL_INTERVAL
            
        retries = 0
        while retries < max_retries:
            endpoint = f"{self.server_url}/api/async/status/{job_id}"
            response = requests.get(endpoint, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                
                if status == 'completed':
                    return data
                elif status == 'failed':
                    error = data.get('error', 'Unknown error')
                    raise Exception(f"ジョブ失敗: {error}")
                
                # Pending/running - continue polling
                time.sleep(interval)
                retries += 1
            else:
                raise Exception(f"ジョブステータス取得エラー: {response.status_code} {response.text}")
        
        raise Exception(f"最大試行回数 ({max_retries}) を超えました")
    
    def improve_code(self, detector_name: str, prompt: str, session_id=None) -> Dict[str, Any]:
        """LLMに改善コードを生成させる"""
        if not session_id and not self.session_id:
            raise ValueError("セッションIDが必要です。先にcreate_session()を呼び出すか、session_idを指定してください")
            
        used_session_id = session_id or self.session_id
        
        endpoint = f"{self.server_url}/api/improve_code"
        payload = {
            "prompt": prompt,
            "session_id": used_session_id
        }
        
        response = requests.post(endpoint, json=payload, timeout=self.timeout)
        
        if response.status_code == 200:
            data = response.json()
            job_id = data.get('job_id')
            
            if not job_id:
                raise ValueError("ジョブIDが返されませんでした")
                
            # ジョブの完了を待つ
            result = self.poll_job_status(job_id)
            return {
                'status': 'success',
                'code': result.get('result', {}).get('code'),
                'message': result.get('result', {}).get('message')
            }
        else:
            return {
                'status': 'error', 
                'error': f"コード改善リクエストエラー: {response.status_code} {response.text}"
            }
            
    def evaluate_detector(self, detector_name: str, params: Dict[str, Any], session_id=None) -> Dict[str, Any]:
        """検出器を評価し、性能メトリクスを返す"""
        used_session_id = session_id or self.session_id
        
        endpoint = f"{self.server_url}/api/job/evaluate"
        payload = {
            **params,
            "session_id": used_session_id
        }
        
        if 'detectors' not in payload and detector_name:
            payload['detectors'] = [detector_name]
            
        response = requests.post(endpoint, json=payload, timeout=self.timeout)
        
        if response.status_code == 200:
            data = response.json()
            job_id = data.get('job_id')
            
            if not job_id:
                raise ValueError("ジョブIDが返されませんでした")
                
            # ジョブの完了を待つ
            result = self.poll_job_status(job_id)
            return {
                'status': 'success',
                **result.get('result', {})
            }
        else:
            return {
                'status': 'error', 
                'error': f"評価リクエストエラー: {response.status_code} {response.text}"
            }
    
    def add_history_event(self, event_type: str, data: Dict[str, Any], session_id=None) -> bool:
        """セッション履歴にイベントを追加する"""
        if not session_id and not self.session_id:
            raise ValueError("セッションIDが必要です")
            
        used_session_id = session_id or self.session_id
        
        endpoint = f"{self.server_url}/api/session/{used_session_id}/history"
        payload = {
            "type": event_type,
            **data
        }
        
        response = requests.post(endpoint, json=payload, timeout=self.timeout)
        
        return response.status_code == 200
    
    def get_session_data(self, session_id=None) -> Dict[str, Any]:
        """セッション情報を取得する"""
        if not session_id and not self.session_id:
            raise ValueError("セッションIDが必要です")
            
        used_session_id = session_id or self.session_id
        
        endpoint = f"{self.server_url}/api/session/{used_session_id}"
        response = requests.get(endpoint, timeout=self.timeout)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"セッション情報取得エラー: {response.status_code} {response.text}")
            
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """指定したセッションの履歴を取得する"""
        endpoint = f"{self.server_url}/api/session/{session_id}"
        response = requests.get(endpoint, timeout=self.timeout)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('history', [])
        else:
            raise Exception(f"セッション履歴取得エラー: {response.status_code} {response.text}")
    
    def get_detector_code(self, detector_name: str, session_id=None) -> Dict[str, Any]:
        """検出器のコードを取得する（特定のセッションで最新の改良バージョンを取得）"""
        used_session_id = session_id or self.session_id
        
        endpoint = f"{self.server_url}/api/detector/code"
        params = {"name": detector_name}
        
        if used_session_id:
            params["session_id"] = used_session_id
            
        response = requests.get(endpoint, params=params, timeout=self.timeout)
        
        if response.status_code == 200:
            return {
                'status': 'success',
                'code': response.text
            }
        else:
            return {
                'status': 'error', 
                'error': f"検出器コード取得エラー: {response.status_code} {response.text}"
            }
    
    def run_grid_search(self, params: Dict[str, Any], session_id=None) -> Dict[str, Any]:
        """グリッドサーチを実行する"""
        used_session_id = session_id or self.session_id
        
        endpoint = f"{self.server_url}/api/job/grid_search"
        payload = {
            **params,
            "session_id": used_session_id
        }
        
        response = requests.post(endpoint, json=payload, timeout=self.timeout)
        
        if response.status_code == 200:
            data = response.json()
            job_id = data.get('job_id')
            
            if not job_id:
                raise ValueError("ジョブIDが返されませんでした")
                
            # ジョブの完了を待つ - グリッドサーチは時間がかかるので長めのタイムアウト
            result = self.poll_job_status(job_id, max_retries=120)
            return {
                'status': 'success',
                **result.get('result', {})
            }
        else:
            return {
                'status': 'error', 
                'error': f"グリッドサーチリクエストエラー: {response.status_code} {response.text}"
            }
    
    def set_detector_params(self, detector_name: str, params: Dict[str, Any], session_id=None) -> Dict[str, Any]:
        """検出器のパラメータを設定する"""
        used_session_id = session_id or self.session_id
        
        endpoint = f"{self.server_url}/api/detector/params"
        payload = {
            "name": detector_name,
            "params": params,
            "session_id": used_session_id
        }
        
        response = requests.post(endpoint, json=payload, timeout=self.timeout)
        
        if response.status_code == 200:
            return {
                'status': 'success',
                'message': "パラメータを正常に設定しました"
            }
        else:
            return {
                'status': 'error', 
                'error': f"パラメータ設定エラー: {response.status_code} {response.text}"
            }

    def add_algorithm(self, code_file, description=""):
        """新しいアルゴリズムバージョンを追加"""
        if not self.session_id:
            print("エラー: 先にセッションを作成してください")
            return None
        
        # コードファイル読み込み
        with open(code_file, 'r') as f:
            code = f.read()
        
        # LLMにコード改善をリクエスト（内部ストレージとして使用）
        endpoint = f"{self.server_url}/api/improve_code"
        payload = {
            "session_id": self.session_id,
            "prompt": f"# 保存用のプロンプト（実際のLLM呼び出しはしません）\nここから下のコードを保存してください。\n\n```python\n{code}\n```"
        }
        
        response = requests.post(endpoint, json=payload)
        if response.status_code in [200, 201, 202]:
            job_data = response.json()
            job_id = job_data.get("job_id")
            
            # ジョブの完了を待機
            result = self.poll_job_status(job_id)
            if result:
                # コード改善結果が返されるので、イベントを記録
                self.add_history_event("コード登録", {"version": description, "code": code})
                return {"version": description, "code": code}
        else:
            print(f"エラー: {response.status_code} - {response.text}")
        
        return None

    def evaluate_algorithm(self):
        """現在のアルゴリズムバージョンを評価"""
        if not self.session_id:
            print("エラー: 先にセッションを作成してください")
            return None
        
        # 評価ジョブをスタート
        endpoint = f"{self.server_url}/api/job/evaluate"
        payload = {
            "session_id": self.session_id,
            "detector_names": [self.session_id]
        }
        
        response = requests.post(endpoint, json=payload)
        if response.status_code in [200, 201, 202]:
            job_data = response.json()
            job_id = job_data.get("job_id")
            
            print("評価処理を開始しました。結果を待機中...")
            
            # ジョブの完了を待機
            result = self.poll_job_status(job_id)
            if result:
                if "error" in result:
                    print(f"評価エラー: {result['error']}")
                    return {"success": False, "error": result["error"]}
                
                print("評価が正常に完了しました")
                self._print_evaluation_summary(result)
                
                # セッション履歴に評価結果を追加
                self.add_history_event("評価完了", result)
                
                return {"success": True, "evaluation_result": result}
            else:
                print("評価処理が失敗しました")
                return {"success": False, "error": "評価ジョブが失敗しました"}
        else:
            print(f"エラー: {response.status_code} - {response.text}")
            return None

    def _print_evaluation_summary(self, result):
        """評価結果のサマリーを表示"""
        if "error" in result:
            print(f"評価エラー: {result['error']}")
            return
        
        # 全体指標の表示
        overall_metrics = result.get("overall_metrics", {})
        if overall_metrics:
            print("\n======= 全体評価指標 =======")
            metrics_table = PrettyTable()
            metrics_table.field_names = ["指標", "値"]
            
            # F値を最初に表示
            if "note" in overall_metrics and "f_measure" in overall_metrics["note"]:
                metrics_table.add_row(["Note F-measure", f"{overall_metrics['note']['f_measure']:.4f}"])
            
            # その他の指標を表示
            for category, metrics in overall_metrics.items():
                for metric, value in metrics.items():
                    if category == "note" and metric == "f_measure":
                        continue  # すでに表示済み
                    metrics_table.add_row([f"{category}.{metric}", f"{value:.4f}"])
            
            print(metrics_table)
        
        # 問題ファイルの表示
        problem_files = result.get("problem_files", [])
        if problem_files:
            print(f"\n======= 問題があるファイル ({len(problem_files)}件) =======")
            files_table = PrettyTable()
            files_table.field_names = ["ファイル名", "Note F-measure", "問題"]
            
            for file_info in problem_files:
                file_name = file_info["file_name"]
                f_measure = file_info["metrics"].get("note", {}).get("f_measure", 0)
                
                # 問題の種類を特定
                problem = "不明"
                metrics = file_info["metrics"].get("note", {})
                if metrics.get("precision", 1.0) < 0.5 and metrics.get("recall", 1.0) > 0.7:
                    problem = "過検出（False Positive多い）"
                elif metrics.get("recall", 1.0) < 0.5 and metrics.get("precision", 1.0) > 0.7:
                    problem = "検出漏れ（False Negative多い）"
                elif metrics.get("precision", 1.0) < 0.7 and metrics.get("recall", 1.0) < 0.7:
                    problem = "検出精度と網羅性の両方に問題"
                
                files_table.add_row([file_name, f"{f_measure:.4f}", problem])
            
            print(files_table)
        else:
            print("\n問題のあるファイルはありません")

def main():
    parser = argparse.ArgumentParser(description="MCPサーバークライアントツール")
    parser.add_argument("--server-url", default=SERVER_URL, help="MCPサーバーのURL")
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")
    
    # create-session コマンド
    create_parser = subparsers.add_parser("create-session", help="新しいセッションを作成")
    
    # load-session コマンド
    load_parser = subparsers.add_parser("load-session", help="既存のセッションをロード")
    load_parser.add_argument("--session-id", required=True, help="セッションID")
    
    # add-algorithm コマンド
    add_parser = subparsers.add_parser("add-algorithm", help="新しいアルゴリズムバージョンを追加")
    add_parser.add_argument("--code-file", required=True, help="アルゴリズムのPythonコードファイル")
    add_parser.add_argument("--description", default="", help="バージョンの説明")
    add_parser.add_argument("--session-id", help="セッションID（ロード済みの場合は不要）")
    
    # evaluate コマンド
    eval_parser = subparsers.add_parser("evaluate", help="現在のアルゴリズムバージョンを評価")
    eval_parser.add_argument("--session-id", help="セッションID（ロード済みの場合は不要）")
    
    # get-detector コマンド
    get_detector_parser = subparsers.add_parser("get-detector", help="検出器情報を取得")
    get_detector_parser.add_argument("name", help="検出器名")
    get_detector_parser.add_argument("--session-id", help="セッションID（オプション）")
    
    args = parser.parse_args()
    
    # クライアントインスタンス作成
    client = MCPClient(server_url=args.server_url)
    
    # コマンド処理
    if args.command == "create-session":
        try:
            session_id = client.create_session()
            print(f"セッションを作成しました: {session_id}")
        except Exception as e:
            print(f"エラー: {e}")
            sys.exit(1)
    
    elif args.command == "load-session":
        client.session_id = args.session_id
    
    elif args.command == "add-algorithm":
        session_id = args.session_id or client.session_id
        if not session_id:
            print("エラー: セッションIDが必要です。--session-idオプションを指定するか、事前にセッションをロードしてください。")
            return
        
        if args.session_id:
            client.session_id = args.session_id
        
        result = client.add_algorithm(args.code_file, args.description)
        if result:
            print(f"アルゴリズムバージョンを追加しました: {result['version']}")
    
    elif args.command == "evaluate":
        session_id = args.session_id or client.session_id
        if not session_id:
            print("エラー: セッションIDが必要です。--session-idオプションを指定するか、事前にセッションをロードしてください。")
            return
        
        if args.session_id:
            client.session_id = args.session_id
        
        client.evaluate_algorithm()
    
    elif args.command == "get-detector":
        try:
            result = client.get_detector_code(args.name, args.session_id)
            if result['status'] == 'success':
                print(result['code'])
            else:
                print(f"エラー: {result['error']}")
                sys.exit(1)
        except Exception as e:
            print(f"エラー: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 