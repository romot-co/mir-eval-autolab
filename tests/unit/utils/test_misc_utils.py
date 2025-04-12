# -*- coding: utf-8 -*-
"""
ユーティリティ関数のテスト
"""
import re
import time
import pytest

from src.utils.misc_utils import generate_id, get_timestamp

def test_generate_id_default():
    """デフォルト設定でIDが生成されることを確認"""
    generated_id = generate_id()
    assert isinstance(generated_id, str)
    # UUIDのフォーマットは8-4-4-4-12の形式（ハイフン含めて36文字）
    assert len(generated_id) == 36
    # UUIDv4フォーマットに従っているか確認
    assert re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', generated_id.lower())

def test_generate_id_uniqueness():
    """生成されるIDがユニークであることを確認"""
    # 複数のIDを生成して一意性をチェック
    ids = [generate_id() for _ in range(100)]
    # セットに変換した場合も同じ長さなら重複なし
    assert len(ids) == len(set(ids))

def test_get_timestamp_returns_float():
    """タイムスタンプが浮動小数点数として返されることを確認"""
    timestamp = get_timestamp()
    assert isinstance(timestamp, float)
    # 現在時刻として妥当な値であることを確認
    current_time = time.time()
    # 10秒以内の誤差であること
    assert abs(timestamp - current_time) < 10 