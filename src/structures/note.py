# src/structures/note.py
from dataclasses import dataclass


@dataclass
class Note:
    start_sec: float
    end_sec: float
    pitch_hz: float
    confidence: float = 1.0  # 評価で使う可能性があるので追加しておく
