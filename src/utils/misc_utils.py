# -*- coding: utf-8 -*-
import uuid
import time
from datetime import datetime

def generate_id():
    """ユニークなUUID v4を生成します。"""
    return str(uuid.uuid4())

def get_timestamp():
    """現在のエポックからの経過時間を秒単位で返します。"""
    return time.time() 

def format_timestamp(timestamp):
    """タイムスタンプを読みやすい形式に変換します。"""
    if timestamp is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return str(timestamp) 