# -*- coding: utf-8 -*-
import uuid
import time

def generate_id():
    """ユニークなUUID v4を生成します。"""
    return str(uuid.uuid4())

def get_timestamp():
    """現在のエポックからの経過時間を秒単位で返します。"""
    return time.time() 