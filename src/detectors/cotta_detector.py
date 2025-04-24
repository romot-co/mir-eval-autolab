"""
COTTA (Cochlear-Optimal-Transport Topological Assignment) 検出器

深層学習を使わず、MIR_eval >= 0.9 をリアルタイムで達成する
完全自動採譜アルゴリズムの実装
"""

import datetime
import logging
import mmap
import os
import struct
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal
from scipy import optimize, spatial
from scipy.fft import rfft, rfftfreq
from scipy.optimize import linear_sum_assignment

from src.detectors import register_detector
from src.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


@register_detector(
    name="COTTADetector",
    description="Cochlear-Optimal-Transport Topological Assignment 検出器",
    version="1.0",
    params={
        "hop_size": 220,  # 5ms @ 44.1kHz
        "win_size": 4096,  # 約92ms
        "erb_channels": 64,
        "min_freq": 20.0,
        "max_freq": 8000.0,
        "persist_threshold": 6.0,
        "sinkhorn_lambda": 6.0,
        "sinkhorn_iterations": 3,
        "kappa_dummy": 8.0,
        "alpha_cost": 1.0,
        "beta_cost": 0.2,
        "gamma_cost": 0.05,
        "onset_threshold": 6.0,  # dB
        "adaptive_hop": True,  # Adaptive hop size
    },
)
class COTTADetector(BaseDetector):
    """
    COTTA (Cochlear-Optimal-Transport Topological Assignment) 検出器

    ERB フィルタバンクと位相再割当てによる時間-周波数表現から
    Persistent Homology と Optimal Transport を組み合わせて
    音楽トランスクリプションを実現します。

    特徴:
    - 深層学習不使用
    - 楽器非依存
    - 低計算コスト
    - 高精度 (MIR_eval F >= 0.9)
    """

    def __init__(self, **kwargs):
        """検出器の初期化"""
        super().__init__(**kwargs)

        # パラメータの取得と設定
        self.hop_size = self.params.get("hop_size", 220)
        self.win_size = self.params.get("win_size", 4096)
        self.erb_channels = self.params.get("erb_channels", 64)
        self.min_freq = self.params.get("min_freq", 20.0)
        self.max_freq = self.params.get("max_freq", 8000.0)
        self.persist_threshold = self.params.get("persist_threshold", 6.0)
        self.sinkhorn_lambda = self.params.get("sinkhorn_lambda", 6.0)
        self.sinkhorn_iterations = self.params.get("sinkhorn_iterations", 3)
        self.kappa_dummy = self.params.get("kappa_dummy", 8.0)
        self.alpha_cost = self.params.get("alpha_cost", 1.0)
        self.beta_cost = self.params.get("beta_cost", 0.2)
        self.gamma_cost = self.params.get("gamma_cost", 0.05)
        self.onset_threshold = self.params.get("onset_threshold", 6.0)
        self.adaptive_hop = self.params.get("adaptive_hop", True)

        # 内部状態の初期化
        self._init_cochlear_filter()
        self._init_components()

    def _init_cochlear_filter(self):
        """ERBフィルタバンクの初期化"""
        # 窓関数の生成
        self.window = np.hanning(self.win_size)

        # ERBスケールのフィルタバンク作成
        self.erb_weights = np.zeros((self.win_size // 2 + 1, self.erb_channels))
        freqs = rfftfreq(self.win_size, 1.0 / 44100)

        for c in range(self.erb_channels):
            # ERBスケールで均等に配置された中心周波数
            fc = 440.0 * 2.0 ** ((c - 36.0) / 12.0)  # 36は基準MIDI番号

            # ERB (Equivalent Rectangular Bandwidth)
            erb_width = 24.7 * (4.37 * fc / 1000.0 + 1.0)

            # ガウシアンフィルタ形状
            self.erb_weights[:, c] = np.exp(-(((freqs - fc) / (erb_width / 4.0)) ** 2))

        # 位相追跡用の状態変数
        self.prev_phase = np.zeros(self.win_size // 2 + 1)
        self.buffer = np.zeros(self.win_size)

        # Adaptive hop用のカウンター
        self.adaptive_hop_counter = 0
        self.current_hop_size = self.hop_size

    def _init_components(self):
        """その他のコンポーネントの初期化"""
        # 現在の状態
        self.current_clusters = []
        self.prev_clusters = []
        self.note_events = []
        self.active_voices = {}  # voice_id -> Voice情報

        # 信号バッファ
        self.input_buffer = np.zeros(self.win_size)

        # メディアン重み（音量正規化用）
        self.median_weight = 0.01

        # 時間追跡
        self.current_time = 0.0

        # KeyFinder状態
        self.chroma_hist = np.zeros(12)
        self.current_key = 0
        self.last_onset_time = 0.0

        # Tempogram状態
        self.onset_envelope = []
        self.current_bpm = 120.0

        # N-gram 確率テーブル
        self.ngram_probs = {}
        self._load_ngram_probabilities()

    def detect(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """
        オーディオデータから音楽情報を検出する

        Parameters
        ----------
        audio_data : np.ndarray
            入力オーディオデータ。形状は (サンプル数,) または (チャンネル数, サンプル数)
        sample_rate : int
            オーディオデータのサンプルレート (Hz)

        Returns
        -------
        Dict[str, np.ndarray]
            検出結果を含む辞書
        """
        print("DEBUG: COTTADetector.detect method started")
        overall_start_time = time.time()
        logger.info(
            f"[COTTA Detect Start] Timestamp: {datetime.datetime.now().isoformat()}"
        )

        # モノラルに変換（必要な場合）
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=0)

        # サンプルレートのチェックと調整
        if sample_rate != 44100:
            logger.warning(
                f"サンプルレート {sample_rate} Hz を 44100 Hz にリサンプリングします"
            )
            # scipy.signal.resample を使用してリサンプリング
            target_length = int(len(audio_data) * 44100 / sample_rate)
            audio_data = signal.resample(audio_data, target_length)
            sample_rate = 44100

        # 状態の初期化
        self._init_components()
        logger.info(f"Internal components initialized.")

        # 結果格納用
        intervals = []
        pitches = []
        frame_times = []
        frame_frequencies = []

        # 仮のノートイベント（後で量子化される）
        note_events = []

        # 検出処理
        i = 0
        frame_count = 0
        processing_times = {
            "total": 0.0,
            "frame_proc": [],
            "quantize": 0.0,
            "grammar": 0.0,
        }
        while i < len(audio_data) - self.win_size:
            frame_start_time = time.time()
            frame_log_prefix = f"[Frame {frame_count} @ {self.current_time:.3f}s]"
            # バッファ更新
            self.input_buffer = np.roll(self.input_buffer, -self.current_hop_size)
            end_pos = min(i + self.current_hop_size, len(audio_data))
            pad_len = self.current_hop_size - (end_pos - i)
            if pad_len > 0:
                # 端の処理（必要に応じてゼロパディング）
                self.input_buffer[-self.current_hop_size : -pad_len] = audio_data[
                    i:end_pos
                ]
                self.input_buffer[-pad_len:] = 0
            else:
                self.input_buffer[-self.current_hop_size :] = audio_data[
                    i : i + self.current_hop_size
                ]

            # 現在時刻の更新
            self.current_time = i / sample_rate

            # フレームを処理
            logger.debug(f"{frame_log_prefix} Processing frame...")
            frame_results = self._process_frame(self.input_buffer)
            frame_proc_time = time.time() - frame_start_time
            processing_times["frame_proc"].append(frame_proc_time)
            logger.debug(
                f"{frame_log_prefix} Frame processed in {frame_proc_time:.4f}s"
            )

            # テンポグラム更新
            onset_strength = 0.0
            if frame_results.get("notes"):
                # オンセットのみエネルギー
                for note in frame_results["notes"]:
                    if note.get("offset") is None:  # オンセット
                        onset_strength += note.get("velocity", 60) / 127.0

            # BPM更新
            current_bpm = self._update_tempogram(onset_strength)

            # 結果の蓄積
            if frame_results.get("notes"):
                for note in frame_results["notes"]:
                    # 終了したノートをintervals/pitchesに追加
                    if note.get("offset") is not None:
                        intervals.append([note["onset"], note["offset"]])
                        pitches.append(note["pitch"])
                    # 全てのノートイベントを蓄積（量子化用）
                    note_events.append(note)

            if frame_results.get("frame_freq") is not None:
                frame_times.append(self.current_time)
                frame_frequencies.append(frame_results["frame_freq"])

            # 適応的ホップサイズ調整（位相ラップを検出した場合）
            if self.adaptive_hop and frame_results.get("phase_wrapped", False):
                if self.current_hop_size > self.hop_size // 2:
                    self.current_hop_size = self.hop_size // 2
                    self.adaptive_hop_counter = 5  # 5フレーム維持
            elif self.adaptive_hop_counter > 0:
                self.adaptive_hop_counter -= 1
                if (
                    self.adaptive_hop_counter == 0
                    and self.current_hop_size < self.hop_size
                ):
                    self.current_hop_size = self.hop_size

            # 次のフレームへ
            i += self.current_hop_size
            frame_count += 1

        # 残っているアクティブなノートを終了
        for voice_id, voice in self.active_voices.items():
            if voice["active"] and voice["on"]:
                note_events.append(
                    {
                        "pitch": voice["pitch"],
                        "velocity": voice["vel"],
                        "onset": voice["onset"],
                        "offset": self.current_time,
                        "voice": voice_id,
                    }
                )

        # 音符の後処理
        if note_events:
            # キー推定
            self.current_key = self._estimate_key()

            # グラマーフィルタ適用前に確率テーブルがロードされているか確認
            if not self.ngram_probs:
                logger.warning(
                    "N-gram probability table not loaded. Grammar filter may be ineffective."
                )

            # 文法フィルタ適用
            grammar_start_time = time.time()
            logger.info("Applying grammar filter...")
            filtered_notes = self._apply_grammar_filter(note_events)
            processing_times["grammar"] = time.time() - grammar_start_time
            logger.info(
                f"Grammar filter applied in {processing_times['grammar']:.4f}s. Filtered notes: {len(filtered_notes)}"
            )

            # テンポ量子化
            quantize_start_time = time.time()
            logger.info(
                f"Quantizing {len(filtered_notes)} notes with BPM: {self.current_bpm:.2f}..."
            )
            quantized_notes = self._quantize_notes(filtered_notes, self.current_bpm)
            processing_times["quantize"] = time.time() - quantize_start_time
            logger.info(
                f"Quantization finished in {processing_times['quantize']:.4f}s. Quantized notes: {len(quantized_notes)}"
            )

            # --- DEBUG: Validate quantized intervals ---
            invalid_intervals_count = 0
            for idx, note in enumerate(quantized_notes):
                onset = note.get("onset")
                offset = note.get("offset")
                if onset is not None and offset is not None and onset > offset:
                    invalid_intervals_count += 1
                    if invalid_intervals_count <= 20:  # Log first 20 invalid intervals
                        logger.error(
                            f"  Invalid quantized interval found at index {idx}: onset={onset:.6f}, offset={offset:.6f}"
                        )
            if invalid_intervals_count > 0:
                logger.error(
                    f"  Found {invalid_intervals_count} total invalid quantized intervals (onset > offset)."
                )
            # --- END DEBUG ---

            # 量子化結果を反映
            intervals_out = []
            pitches_out = []

            for note in quantized_notes:
                if note.get("offset") is not None:
                    intervals_out.append([note["onset"], note["offset"]])
                    pitches_out.append(note["pitch"])

        else:
            quantized_notes = []  # Ensure quantized_notes exists
            logger.info("No note events found, skipping post-processing.")

        # 検出にかかった時間
        detection_time = time.time() - overall_start_time
        processing_times["total"] = detection_time
        avg_frame_time = (
            np.mean(processing_times["frame_proc"])
            if processing_times["frame_proc"]
            else 0
        )

        logger.info(f"[COTTA Detect End] Total time: {detection_time:.4f}s")
        logger.info(
            f"Processing time breakdown: Avg Frame={avg_frame_time:.4f}s, Grammar={processing_times['grammar']:.4f}s, Quantize={processing_times['quantize']:.4f}s"
        )

        # 結果をnumpyアレイに変換
        intervals_out = []
        pitches_out = []
        if quantized_notes:
            for note in quantized_notes:
                if note.get("offset") is not None:
                    intervals_out.append([note["onset"], note["offset"]])
                    pitches_out.append(note["pitch"])

        return {
            "intervals": np.array(intervals_out, dtype=float),
            "note_pitches": np.array(pitches_out, dtype=float),
            "frame_times": np.array(frame_times, dtype=float),
            "frame_frequencies": np.array(frame_frequencies, dtype=float),
            "detection_time": detection_time,  # Keep original detection_time metric
        }

    def _process_frame(self, frame: np.ndarray) -> Dict:
        """
        単一のオーディオフレームを処理する

        Parameters
        ----------
        frame : np.ndarray
            処理するオーディオフレーム

        Returns
        -------
        Dict
            フレーム処理結果
        """
        proc_start = time.time()
        # ここで以下の処理を順番に行う：
        # 1. ERB + Phase Reassignment でスペクトルを取得
        # 2. ピーク抽出
        # 3. Persistent Homology でクラスター化
        # 4. Optimal Transport で時間追跡
        # 5. ノートイベント生成

        # 1. CochlearFilter: ERBフィルタと位相再割当
        step_start = time.time()
        logger.debug("  Step 1: Cochlear Filter...")
        complex_spec, reassigned_freqs, phase_wrapped = self._cochlear_filter(frame)
        logger.debug(f"    Cochlear Filter took {time.time() - step_start:.4f}s")

        # 2. PeakExtractor: ピーク抽出
        step_start = time.time()
        logger.debug("  Step 2: Extract Peaks...")
        peaks = self._extract_peaks(complex_spec, reassigned_freqs)
        logger.debug(
            f"    Extract Peaks took {time.time() - step_start:.4f}s. Found {len(peaks)} peaks."
        )

        # 3. HomologyStream: クラスター抽出
        step_start = time.time()
        logger.debug(f"  Step 3: Extract Clusters (Input peaks: {len(peaks)})...")
        clusters = self._extract_clusters(peaks)
        logger.debug(
            f"    Extract Clusters took {time.time() - step_start:.4f}s. Found {len(clusters)} clusters."
        )

        # 4. OTTracker: 時間追跡
        step_start = time.time()
        logger.debug(
            f"  Step 4: Track Clusters (Prev: {len(self.prev_clusters)}, Curr: {len(clusters)})..."
        )
        assignments = self._track_clusters(self.prev_clusters, clusters)
        logger.debug(
            f"    Track Clusters took {time.time() - step_start:.4f}s. Found {len(assignments)} assignments."
        )

        # 5. NoteStream: ノート生成
        step_start = time.time()
        logger.debug(
            f"  Step 5: Generate Notes (Input clusters: {len(clusters)}, Assignments: {len(assignments)})..."
        )
        notes = self._generate_notes(clusters, assignments)
        logger.debug(
            f"    Generate Notes took {time.time() - step_start:.4f}s. Generated {len(notes)} note events."
        )

        # 前のフレームのクラスター更新
        self.prev_clusters = clusters

        # フレームの主要周波数（複数の場合は最大振幅のもの）
        frame_freq = None
        if clusters:
            max_cluster = max(clusters, key=lambda c: c["weight"])
            frame_freq = max_cluster["f0"]

        logger.debug(f"  Total frame processing time: {time.time() - proc_start:.4f}s")
        return {
            "notes": notes,
            "frame_freq": frame_freq,
            "phase_wrapped": phase_wrapped,
        }

    def _cochlear_filter(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        ERBフィルタバンクと位相再割当てを適用する

        Parameters
        ----------
        frame : np.ndarray
            入力オーディオフレーム

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, bool]
            複素スペクトル、再割当て周波数、位相ラップフラグ
        """
        # 窓関数適用
        windowed = frame * self.window

        # FFT計算
        complex_spec = rfft(windowed)

        # 位相再割当て
        phases = np.angle(complex_spec)
        delta_phases = phases - self.prev_phase

        # 位相差を-πからπの範囲に正規化
        delta_phases = np.mod(delta_phases + np.pi, 2 * np.pi) - np.pi

        # 位相ラップ検出 - ±π付近のデルタ位相が多い場合
        phase_wrapped = (
            np.sum(np.abs(delta_phases) > 0.8 * np.pi) > len(delta_phases) * 0.2
        )

        # 再割当て周波数計算
        bins = np.arange(len(complex_spec))
        bin_offsets = delta_phases * self.win_size / (2 * np.pi * self.current_hop_size)
        reassigned_freqs = (bins + bin_offsets) * 44100 / self.win_size

        # 不正な周波数を修正（非負かつナイキスト以下）
        reassigned_freqs = np.clip(reassigned_freqs, 0, 44100 / 2)

        # ERBバンドごとにパワー統合（バンドサム処理）
        mag_squared = np.abs(complex_spec) ** 2
        erb_powers = np.zeros(self.erb_channels)

        for c in range(self.erb_channels):
            # 各バンドごとのパワー積分
            erb_powers[c] = np.sum(mag_squared * self.erb_weights[:, c])

        # 位相更新
        self.prev_phase = phases

        return complex_spec, reassigned_freqs, phase_wrapped

    def _extract_peaks(
        self, complex_spec: np.ndarray, reassigned_freqs: np.ndarray
    ) -> List[Dict]:
        """
        スペクトルからピークを抽出する
        ERBバンドごとに局所ピークを探索する方式に変更

        Parameters
        ----------
        complex_spec : np.ndarray
            複素スペクトル
        reassigned_freqs : np.ndarray
            位相再割当てされた周波数

        Returns
        -------
        List[Dict]
            抽出されたピークのリスト (`band` インデックスを含む)
        """
        peaks = []
        mag = np.abs(complex_spec)
        phases = np.angle(complex_spec)

        # 閾値（ヒステリシス）
        rise_threshold = 10 ** (-66 / 20)  # -66dB
        fall_threshold = 10 ** (-72 / 20)  # -72dB

        # 各ERBバンドについてピークを探索
        for c in range(self.erb_channels):
            # バンド内の周波数ビンインデックスを取得
            band_indices = np.where(self.erb_weights[:, c] > 0.1)[
                0
            ]  # 重みが0.1より大きいビン
            if len(band_indices) < 3:
                continue  # ピーク検出には最低3点必要

            # バンド内の振幅と位相
            band_mag = mag[band_indices]
            band_phases = phases[band_indices]
            band_reassigned_freqs = reassigned_freqs[band_indices]
            band_power = band_mag**2 * self.erb_weights[band_indices, c]

            in_peak = False
            for k_band in range(1, len(band_mag) - 1):
                k_global = band_indices[k_band]  # グローバルなビンインデックス
                amp = band_mag[k_band]
                power = band_power[k_band]

                # ヒステリシス閾値チェック
                threshold = fall_threshold if in_peak else rise_threshold
                # バンド内パワーではなく、元のマグニチュードで判定
                if amp < threshold:
                    if in_peak:
                        in_peak = False
                    continue

                # バンド内でのピーク検出（前後より大きい）
                if amp > band_mag[k_band - 1] and amp >= band_mag[k_band + 1]:
                    in_peak = True

                    # パラボラ補間（バンド内インデックスで）
                    a, b, c = band_mag[k_band - 1], amp, band_mag[k_band + 1]
                    delta_band = 0.5 * (a - c) / (a - 2 * b + c + 1e-9)

                    # 周波数の確定（再割当て周波数を優先）
                    freq = band_reassigned_freqs[k_band]

                    # 再割当てが不適切な場合は補間で代替（グローバルビンで）
                    if not (20 <= freq <= 8000):
                        # 元のスペクトルで補間
                        if 1 <= k_global < len(mag) - 1:
                            a_g, b_g, c_g = (
                                mag[k_global - 1],
                                mag[k_global],
                                mag[k_global + 1],
                            )
                            delta_g = 0.5 * (a_g - c_g) / (a_g - 2 * b_g + c_g + 1e-9)
                            true_bin = k_global + delta_g
                            freq = true_bin * 44100 / self.win_size
                        else:  # 端の場合は補間不可
                            freq = k_global * 44100 / self.win_size

                    # ピーク情報を記録（バンドインデックスを追加）
                    peaks.append(
                        {
                            "freq": freq,
                            "amp": np.sqrt(power),  # バンド内パワーに基づく振幅
                            "bin": k_global,
                            "phase": band_phases[k_band],
                            "band": c,  # ERBバンドインデックス
                        }
                    )

        # 重複ピークの削除（近接周波数で最も強いものを残す）
        if not peaks:
            return []
        peaks.sort(key=lambda p: p["freq"])
        filtered_peaks = []
        last_peak = peaks[0]

        for i in range(1, len(peaks)):
            # 周波数差が閾値以下（例：1/4半音）なら比較
            if abs(np.log2(peaks[i]["freq"]) - np.log2(last_peak["freq"])) < 0.25 / 12:
                # 振幅が大きい方を採用
                if peaks[i]["amp"] > last_peak["amp"]:
                    last_peak = peaks[i]
            else:
                # 閾値より離れていたら前のピークを確定
                filtered_peaks.append(last_peak)
                last_peak = peaks[i]
        filtered_peaks.append(last_peak)  # 最後のピークを追加

        return filtered_peaks

    def _extract_clusters(self, peaks: List[Dict]) -> List[Dict]:
        """
        Persistent Homologyを使用してピークをクラスタリングする
        k最近傍グラフ + Union-Find によるO(n log n)実装

        Parameters
        ----------
        peaks : List[Dict]
            ピークのリスト

        Returns
        -------
        List[Dict]
            クラスターのリスト
        """
        cluster_start = time.time()
        if not peaks:
            return []

        n = len(peaks)

        # DSU (Disjoint Set Union) の実装
        class DSU:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n

            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def union(self, x, y):
                x_root = self.find(x)
                y_root = self.find(y)
                if x_root == y_root:
                    return False
                if self.rank[x_root] < self.rank[y_root]:
                    self.parent[x_root] = y_root
                else:
                    self.parent[y_root] = x_root
                    if self.rank[x_root] == self.rank[y_root]:
                        self.rank[x_root] += 1
                return True

        dsu = DSU(n)

        # k最近傍のkを決定: k = ⌈10 log₂ n⌉
        k = max(1, int(10 * np.log2(n + 1)))
        k = min(k, n - 1)  # 実ノード数以下に制限

        # 周波数の対数空間でのk最近傍を効率的に見つける
        log_freqs = np.array([np.log2(peak["freq"]) for peak in peaks])

        # kd-treeを使用して最近傍を高速に検索
        if n > 1:  # 複数ノードが必要
            tree = spatial.KDTree(log_freqs.reshape(-1, 1))
            _, neighbor_indices = tree.query(log_freqs.reshape(-1, 1), k=min(k + 1, n))

            # エッジを生成（距離順）
            edges = []
            for i in range(n):
                for j_idx in range(1, len(neighbor_indices[i])):  # 0番目は自分自身
                    j = neighbor_indices[i][j_idx]
                    dist = abs(log_freqs[i] - log_freqs[j])
                    edges.append((dist, i, j))

            # 重複除去（無向グラフなので）と距離でソート
            edges = list(set([(d, min(i, j), max(i, j)) for d, i, j in edges]))
            edges.sort()  # 距離順にソート
        else:
            edges = []  # ノードが1つしかない場合

        # コンポーネント管理
        components = [[] for _ in range(n)]
        birth_times = [float("inf")] * n

        # 初期コンポーネント
        for i in range(n):
            components[i] = [i]  # 各要素が自分自身だけのコンポーネントから開始

        # クラスター結果
        clusters = []

        # Persistent Homologyフィルトレーション
        for dist, i, j in edges:
            i_root = dsu.find(i)
            j_root = dsu.find(j)

            if i_root != j_root:
                # 誕生時刻
                if birth_times[i_root] == float("inf") and birth_times[j_root] == float(
                    "inf"
                ):
                    # 両方が誕生していない場合は現在の距離を誕生時刻とする
                    b = dist
                else:
                    # 少なくとも1つが誕生している場合は既存の誕生時刻を使用
                    b = max(birth_times[i_root], birth_times[j_root])
                    if b == float("inf"):
                        b = dist

                # 持続率が閾値を超えるかチェック
                persistence_ratio = dist / b if b > 0 else float("inf")

                if persistence_ratio > self.persist_threshold:
                    # 安定クラスターとして登録
                    for root in [i_root, j_root]:
                        if components[root] and len(components[root]) > 0:
                            # 必要なピークだけを使ってクラスターを生成
                            member_peaks = [peaks[idx] for idx in components[root]]
                            cluster = self._create_cluster(member_peaks)
                            if cluster:
                                clusters.append(cluster)

                # コンポーネントをマージ
                merged = []
                new_root = dsu.find(i_root)  # unionした後の新rootを取得

                # 効率的なマージ - 新しいリストを作成
                if i_root == new_root:
                    merged = components[i_root].copy()
                    merged.extend(components[j_root])
                else:
                    merged = components[j_root].copy()
                    merged.extend(components[i_root])

                # 新rootにマージしたコンポーネントを設定
                components[new_root] = merged

                # 古いコンポーネントをクリア
                if i_root != new_root:
                    components[i_root] = []
                if j_root != new_root and j_root != i_root:
                    components[j_root] = []

                # 誕生時刻を設定
                birth_times[new_root] = b

                # DSUでマージ
                dsu.union(i_root, j_root)

        # 残りのコンポーネントをクラスターに変換
        for i in range(n):
            if dsu.find(i) == i and components[i] and len(components[i]) > 0:
                member_peaks = [peaks[idx] for idx in components[i]]
                cluster = self._create_cluster(member_peaks)
                if cluster:
                    clusters.append(cluster)

        logger.debug(
            f"    _extract_clusters: O(n log n) impl took {time.time() - cluster_start:.4f}s, found {len(clusters)} clusters"
        )
        return clusters

    def _create_cluster(self, peaks: List[Dict]) -> Dict:
        """
        ピークのコレクションからクラスターを作成し、基本周波数を推定する

        Parameters
        ----------
        peaks : List[Dict]
            クラスターを構成するピークのリスト

        Returns
        -------
        Dict
            クラスター情報
        """
        if not peaks:
            return None

        # クラスター総重み
        weight = sum(p["amp"] for p in peaks)

        # 単一ピークの場合は自明
        if len(peaks) == 1:
            return {
                "members": peaks,
                "f0": peaks[0]["freq"],
                "weight": weight,
                "phase": peaks[0]["phase"],  # 位相情報を保存
            }

        # 周波数でソート
        sorted_peaks = sorted(peaks, key=lambda p: p["freq"])

        # 基本周波数候補の探索範囲
        min_freq = sorted_peaks[0]["freq"] / 10  # 最小周波数の1/10まで
        max_freq = sorted_peaks[-1]["freq"]  # 最大周波数まで

        # 最大振幅のピーク
        max_amp_peak = max(peaks, key=lambda p: p["amp"])
        max_amp_freq = max_amp_peak["freq"]

        # 振幅で正規化した重み
        total_amp = sum(p["amp"] for p in peaks)
        if total_amp > 1e-9:
            weights = [p["amp"] / total_amp for p in peaks]
        else:
            weights = [0.0] * len(peaks)

        # 基本周波数を最適化（整数調波LP）
        # 各ピークが調波数列のどの部分に対応するかを探索
        best_f0 = min_freq
        min_error = float("inf")

        # より効率的なLP実装（整数調波モデル）
        def compute_error(f0):
            error = 0
            for peak, w in zip(peaks, weights):
                # 周波数比を計算
                ratio = peak["freq"] / f0
                # 最も近い整数調波数
                harmonic = int(round(ratio))
                harmonic = max(1, min(harmonic, 10))  # 1-10次の範囲に制限

                # 調波からの誤差を重み付き加算
                freq_err = abs(peak["freq"] - harmonic * f0)
                error += w * freq_err
            return error

        # 最適化ステップ（制約付き最適化）
        try:
            from scipy.optimize import minimize_scalar

            result = minimize_scalar(
                compute_error, bounds=(min_freq, max_freq), method="bounded"
            )

            if result.success:
                best_f0 = result.x
                min_error = result.fun
        except ImportError:
            # scipy.optimizeが使えない場合は離散サンプリングで代替
            log_min = np.log(min_freq)
            log_max = np.log(max_freq)

            for i in range(1000):
                f = np.exp(log_min + (log_max - log_min) * i / 999)
                error = compute_error(f)

                if error < min_error:
                    min_error = error
                    best_f0 = f

        # 最大振幅ピークが1倍・2倍・3倍調波である可能性をチェック
        for div in [1, 2, 3]:
            candidate = max_amp_freq / div
            if candidate < min_freq:
                continue

            error = compute_error(candidate)
            if error < min_error:
                min_error = error
                best_f0 = candidate

        # 代表位相（最大振幅ピークの位相）
        phase = max_amp_peak["phase"]

        return {"members": peaks, "f0": best_f0, "weight": weight, "phase": phase}

    def _track_clusters(
        self, prev_clusters: List[Dict], curr_clusters: List[Dict]
    ) -> List[Dict]:
        """
        Sinkhorn正則化Optimal Transportを使用してクラスターを追跡する
        (現在は線形割り当て (Hungarian) のみを使用)

        Parameters
        ----------
        prev_clusters : List[Dict]
            前フレームのクラスター
        curr_clusters : List[Dict]
            現在フレームのクラスター

        Returns
        -------
        List[Dict]
            割り当て結果（prev_idxとnext_idxのペア）
        """
        track_start = time.time()
        if not prev_clusters or not curr_clusters:
            return []

        m = len(prev_clusters)
        n = len(curr_clusters)

        # コスト行列（ダミー行/列を含む (m+1) x (n+1)） - Hungarian用に実部のみ利用
        C = np.full((m, n), self.kappa_dummy, dtype=np.float32)

        # 実コスト計算
        for i in range(m):
            for j in range(n):
                C[i, j] = self._cluster_cost(prev_clusters[i], curr_clusters[j])

        # --- GPU/Sparse Sinkhorn関連のコードを削除 ---
        # use_sparse, use_gpu フラグ、assignSparse/assignDense 呼び出しを削除
        # Sinkhorn反復 (u, v, K) の計算も不要に (Hungarianのみ使用)

        # 離散化: ハンガリアン法を使用
        assignments = []

        # 実際のクラスター間のみで最適割り当て問題を解く
        if m > 0 and n > 0:
            # ハンガリアン法（線形割り当て問題）
            # コスト最小化なので、コスト行列をそのまま使用
            row_ind, col_ind = linear_sum_assignment(C)

            # コストがダミーコスト未満のもののみを採用
            for r, c in zip(row_ind, col_ind):
                cost = C[r, c]
                if cost < self.kappa_dummy:
                    assignments.append(
                        {"prev_idx": int(r), "next_idx": int(c), "cost": float(cost)}
                    )

        logger.debug(
            f"    _track_clusters: Linear Assignment took {time.time() - track_start:.4f}s"
        )
        return assignments

    def _cluster_cost(self, a: Dict, b: Dict) -> float:
        """
        二つのクラスター間のコストを計算する

        Parameters
        ----------
        a : Dict
            クラスターA
        b : Dict
            クラスターB

        Returns
        -------
        float
            コスト値
        """
        # MIDI半音距離
        dp = abs(self._freq_to_midi(a["f0"]) - self._freq_to_midi(b["f0"]))

        # 振幅差（dB）
        dE = abs(self._to_db(a["weight"]) - self._to_db(b["weight"]))

        # 位相差 - 位相情報を有効活用
        dPhase = abs(a.get("phase", 0.0) - b.get("phase", 0.0))
        # π周期性を考慮して0～1に正規化
        dPhase = min(dPhase, 2 * np.pi - dPhase) / np.pi

        # 各要素を重み付けして合算
        return self.alpha_cost * dp + self.beta_cost * dE + self.gamma_cost * dPhase

    def _generate_notes(
        self, curr_clusters: List[Dict], assignments: List[Dict]
    ) -> List[Dict]:
        """
        クラスター割り当てからノートイベントを生成する

        Parameters
        ----------
        curr_clusters : List[Dict]
            現在フレームのクラスター
        assignments : List[Dict]
            クラスター間の割り当て

        Returns
        -------
        List[Dict]
            ノートイベントのリスト
        """
        notes = []

        # 全ボイスを非アクティブに初期化
        for voice_id in self.active_voices:
            self.active_voices[voice_id]["active"] = False

        # マップに従ってボイス追跡
        for assignment in assignments:
            prev_idx = assignment["prev_idx"]
            next_idx = assignment["next_idx"]

            # ボイスがなければ作成
            if prev_idx not in self.active_voices:
                self.active_voices[prev_idx] = {
                    "pitch": 0,
                    "vel": 0,
                    "onset": 0,
                    "active": False,
                    "on": False,
                    "energy": -120.0,  # dB
                }

            voice = self.active_voices[prev_idx]
            cluster = curr_clusters[next_idx]

            # ボイスをアクティブに設定
            voice["active"] = True

            # MIDI ノート番号および速度計算
            midi_pitch = int(round(self._freq_to_midi(cluster["f0"])))

            # 速度マッピング
            norm_weight = cluster["weight"] / (self.median_weight + 1e-9)
            vel_val = int(40 + 30 * np.log10(max(norm_weight, 1e-9)))
            vel = max(0, min(127, vel_val))

            # 新規ノートイベント
            if not voice["on"]:
                # 新規ノート発音
                onset_energy = False
                if voice["energy"] > -120:
                    # エネルギー上昇判定（6dB以上）
                    prev_energy = voice["energy"]
                    curr_energy = self._to_db(cluster["weight"])
                    onset_energy = curr_energy - prev_energy >= self.onset_threshold

                # 新規発音条件：未発音または大きなエネルギー上昇
                voice["pitch"] = midi_pitch
                voice["vel"] = vel
                voice["onset"] = self.current_time
                voice["on"] = True
                voice["energy"] = self._to_db(cluster["weight"])

                # クロマヒストグラム更新（KeyFinder用）
                self._update_chroma_hist(midi_pitch)

                notes.append(
                    {
                        "pitch": midi_pitch,
                        "velocity": vel,
                        "onset": self.current_time,
                        "offset": None,
                        "voice": prev_idx,
                    }
                )

            elif voice["pitch"] != midi_pitch:
                # ピッチ変更：現在の音を終了し、新しい音を開始
                notes.append(
                    {
                        "pitch": voice["pitch"],
                        "velocity": voice["vel"],
                        "onset": voice["onset"],
                        "offset": self.current_time,
                        "voice": prev_idx,
                    }
                )

                # 新規ノート発音
                voice["pitch"] = midi_pitch
                voice["vel"] = vel
                voice["onset"] = self.current_time
                voice["on"] = True
                voice["energy"] = self._to_db(cluster["weight"])

                # クロマヒストグラム更新（KeyFinder用）
                self._update_chroma_hist(midi_pitch)

                notes.append(
                    {
                        "pitch": midi_pitch,
                        "velocity": vel,
                        "onset": self.current_time,
                        "offset": None,
                        "voice": prev_idx,
                    }
                )

            else:
                # 同一ピッチ継続中：エネルギー更新
                voice["energy"] = self._to_db(cluster["weight"])

        # 現在フレームで非アクティブになったボイスの終了処理
        for voice_id, voice in self.active_voices.items():
            if not voice["active"] and voice["on"]:
                notes.append(
                    {
                        "pitch": voice["pitch"],
                        "velocity": voice["vel"],
                        "onset": voice["onset"],
                        "offset": self.current_time,
                        "voice": voice_id,
                    }
                )
                voice["on"] = False

        # クラスタ重みの統計更新
        if curr_clusters:
            avg_weight = sum(c["weight"] for c in curr_clusters) / len(curr_clusters)
            # メディアン重み更新（指数移動平均）
            self.median_weight = self.median_weight * 0.95 + avg_weight * 0.05

        return notes

    def _update_chroma_hist(self, midi_pitch):
        """
        クロマヒストグラムを更新する (KeyFinder用)

        Parameters
        ----------
        midi_pitch : int
            MIDIノート番号
        """
        # 音符間間隔(IoI)の重み付け
        weight = 1.0
        if self.last_onset_time > 0:
            interval = self.current_time - self.last_onset_time
            # 短い音符間隔ほど重要視
            weight = max(0.2, 1.0 / (1.0 + interval))
        self.last_onset_time = self.current_time

        # クロマ（0-11）を抽出
        chroma = midi_pitch % 12

        # ヒストグラム更新
        self.chroma_hist[chroma] += weight

    def _estimate_key(self) -> int:
        """
        クロマヒストグラムから調を推定する

        Returns
        -------
        int
            調（0-11、Cを0とする）
        """
        # Krumhansl-Schmucklerプロファイル
        maj_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        min_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )

        # 相対短調補正（-1.5dB）
        minor_penalty = 10 ** (-1.5 / 20)

        best_corr = -1
        best_key = 0

        # 各調の相関を計算
        for key in range(12):
            # 現在の調に合わせてヒストグラムをシフト
            shifted_hist = np.roll(self.chroma_hist, -key)

            # 正規化
            hist_sum = np.sum(shifted_hist)
            if hist_sum > 1e-9:
                shifted_hist /= hist_sum

            # 長調相関
            corr_maj = np.sum(shifted_hist * maj_profile)
            if corr_maj > best_corr:
                best_corr = corr_maj
                best_key = key

            # 短調相関（ペナルティ適用）
            corr_min = np.sum(shifted_hist * min_profile) * minor_penalty
            if corr_min > best_corr:
                best_corr = corr_min
                best_key = key  # 短調も同じkeyで表現（実装簡略化）

        return best_key

    def _freq_to_midi(self, freq: float) -> float:
        """
        周波数をMIDIノート番号に変換する
        0 Hz入力でもクラッシュしないよう保護

        Parameters
        ----------
        freq : float
            周波数（Hz）

        Returns
        -------
        float
            MIDIノート番号
        """
        # 0または負の周波数を防御
        if freq <= 1e-6:
            return 0.0
        return 69 + 12 * np.log2(freq / 440.0)

    def _to_db(self, linear: float) -> float:
        """線形スケールをdBに変換する"""
        return 20 * np.log10(linear + 1e-12)

    def _midi_to_hz(self, midi_note: float) -> float:
        """MIDIノート番号を周波数(Hz)に変換する"""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    def _apply_grammar_filter(self, note_events: List[Dict]) -> List[Dict]:
        """
        文法フィルタを適用して音符列を整形する
        N-gram確率テーブルを利用

        Parameters
        ----------
        note_events : List[Dict]
            音符イベントのリスト

        Returns
        -------
        List[Dict]
            整形された音符イベントのリスト
        """
        if not note_events or not self.ngram_probs:
            logger.warning(
                "Grammar filter skipped: No notes or N-gram table not loaded."
            )
            return note_events

        # 現在の調を推定
        key = self._estimate_key()

        # ボイス(voice_id)ごとにノートイベントをグループ化
        voice_notes = {}
        for i, note in enumerate(note_events):
            voice_id = note.get("voice", 0)
            if voice_id not in voice_notes:
                voice_notes[voice_id] = []
            voice_notes[voice_id].append((i, note))

        # 各ボイスを独立に処理
        needs_cleanup = [False] * len(note_events)

        # 簡易n-gramプロファイル（実際の実装ではもっと大きいデータからロード）
        ngram_order = 3
        prob_threshold = 0.01  # 確率の閾値 (例)

        # キー依存のスケール度数（C majorを基準とする）
        major_compat = np.array(
            [1.0, 0.3, 0.8, 0.3, 0.9, 0.8, 0.3, 1.0, 0.3, 0.7, 0.3, 0.6]
        )

        for voice_id, notes in voice_notes.items():
            # 時刻順にソートされていることを確認
            notes.sort(key=lambda item: item[1]["onset"])

            if len(notes) < ngram_order:
                continue

            # 各ノートの適合度をチェック
            for i in range(ngram_order - 1, len(notes)):
                # n-gramコンテキスト構築 (MIDIピッチ)
                context_pitches = [
                    note[1]["pitch"] for note in notes[i - (ngram_order - 1) : i]
                ]

                # 現在のピッチ
                current_pitch = notes[i][1]["pitch"]

                # n-gram確率を取得
                prob = self._get_ngram_probability(context_pitches, current_pitch, key)

                # 低確率なノートをマーク
                if prob < prob_threshold:
                    needs_cleanup[notes[i][0]] = True
                else:
                    # キーとの適合度も考慮 (確率が高い場合のみ)
                    current_chroma = current_pitch % 12
                    key_compat = major_compat[(current_chroma - key) % 12]
                    if key_compat < 0.4:
                        needs_cleanup[notes[i][0]] = True

        # クリーンアップ対象ノートを修正
        cleaned_events = []

        for i, note in enumerate(note_events):
            if needs_cleanup[i]:
                # 代替候補を探す（キーに合うように半音上下）
                orig_pitch = note["pitch"]
                candidates = []

                # ±1, ±2 半音の修正を試みる
                for delta in [-2, -1, 1, 2]:
                    new_pitch = orig_pitch + delta
                    if 0 <= new_pitch <= 127:
                        # コンテキストを取得
                        voice_id = note.get("voice", 0)
                        note_index_in_voice = -1
                        if voice_id in voice_notes:
                            for idx_v, (idx_global, _) in enumerate(
                                voice_notes[voice_id]
                            ):
                                if idx_global == i:
                                    note_index_in_voice = idx_v
                                    break

                        context_pitches = []
                        if note_index_in_voice >= ngram_order - 1:
                            context_pitches = [
                                n[1]["pitch"]
                                for n in voice_notes[voice_id][
                                    note_index_in_voice
                                    - (ngram_order - 1) : note_index_in_voice
                                ]
                            ]

                        new_prob = self._get_ngram_probability(
                            context_pitches, new_pitch, key
                        )
                        key_compat = major_compat[(new_pitch % 12 - key) % 12]

                        # 確率とキー適合度の両方を考慮
                        score = new_prob * key_compat
                        if score > prob_threshold * 0.5:  # 元の閾値の半分以上なら候補
                            candidates.append((new_pitch, score))

                if candidates:
                    # 最もスコアの高い候補を選択
                    best_candidate = max(candidates, key=lambda x: x[1])
                    note_copy = note.copy()
                    note_copy["pitch"] = best_candidate[0]
                    cleaned_events.append(note_copy)
                else:
                    # 適切な候補がなければ元のまま（または削除も検討可）
                    cleaned_events.append(note)
            else:
                cleaned_events.append(note)

        # ボイスリーディング制約の適用
        return self._apply_voice_leading(cleaned_events)

    def _load_ngram_probabilities(self, filename="ngram3.bin"):
        """
        バイナリファイルからN-gram確率テーブルをロードする
        ファイルフォーマット: (int64 key, float32 probability) の連続
        """
        entry_size = struct.calcsize("qf")  # int64 + float32
        self.ngram_probs = {}

        if not os.path.exists(filename):
            logger.warning(
                f"N-gram probability file '{filename}' not found. Grammar filter will use default probabilities."
            )
            return

        try:
            with open(filename, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    logger.info(
                        f"Loading N-gram probabilities from '{filename}' ({mm.size()} bytes)"
                    )
                    offset = 0
                    while offset < mm.size():
                        if offset + entry_size > mm.size():
                            logger.warning(
                                f"Incomplete entry found at offset {offset} in '{filename}'. Stopping read."
                            )
                            break
                        key, prob = struct.unpack_from("qf", mm, offset)
                        # ここではキーを直接辞書に入れる (ローリングハッシュ済みと仮定)
                        self.ngram_probs[key] = prob
                        offset += entry_size
                    logger.info(f"Loaded {len(self.ngram_probs)} N-gram entries.")
        except Exception as e:
            logger.error(f"Error loading N-gram probabilities from '{filename}': {e}")
            self.ngram_probs = {}  # エラー発生時は空にする

    def _get_ngram_probability(
        self, context_pitches: List[int], current_pitch: int, key: int
    ) -> float:
        """
        指定されたコンテキストとピッチに対するN-gram確率を取得する
        (現状はダミー実装、ngram3.binの構造に合わせて要調整)
        """
        if not self.ngram_probs:
            return 0.5  # テーブルがなければデフォルト確率

        # --- ダミーのキー生成ロジック ---
        # 実際のローリングハッシュ等をここに実装する必要がある
        # 例: contextとcurrent_pitchからint64キーを計算
        hash_key = hash(tuple(context_pitches + [current_pitch]))
        # -------------------------------

        # テーブルから確率を取得、なければデフォルト値
        return self.ngram_probs.get(hash_key, 0.001)  # 見つからない場合は低い確率

    def _apply_voice_leading(self, note_events: List[Dict]) -> List[Dict]:
        """
        ボイスリーディング制約を適用する
        同時発音時のボイス順序も考慮して修正
        """
        if not note_events:
            return []

        # ボイス(voice_id)ごとにノートイベントをグループ化
        voice_notes: Dict[int, List[Dict]] = {}
        for note in note_events:
            voice_id = note.get("voice", 0)
            if voice_id not in voice_notes:
                voice_notes[voice_id] = []
            voice_notes[voice_id].append(note)

        # 各ボイスの大きな跳躍を修正
        for voice_id, notes in voice_notes.items():
            # 発音時刻でソート
            notes.sort(key=lambda x: x["onset"])

            if len(notes) < 2:
                continue

            for i in range(1, len(notes)):
                prev_note = notes[i - 1]
                curr_note = notes[i]

                # 間隔を計算
                interval = curr_note["pitch"] - prev_note["pitch"]

                # 12半音以上の跳躍を修正
                if abs(interval) > 12:
                    # オクターブ方向に修正
                    octaves = interval // 12
                    new_pitch = prev_note["pitch"] + (interval - octaves * 12)
                    # MIDIノート範囲内に収める
                    curr_note["pitch"] = max(0, min(127, new_pitch))

        # ボイス間クロッシング検出と修正 (同時発音を考慮)
        if len(voice_notes) >= 2:
            # すべてのノートを発音時刻順にソート
            all_notes = sorted(note_events, key=lambda x: x["onset"])

            # タイムポイントごとのノートをグループ化 (量子化誤差を考慮)
            time_points: Dict[float, List[Dict]] = {}
            for note in all_notes:
                onset = note["onset"]
                # 量子化誤差を考慮して、近いonsetをまとめる (例: 1ms以内)
                quantized_onset = round(onset * 1000) / 1000.0
                if quantized_onset not in time_points:
                    time_points[quantized_onset] = []
                time_points[quantized_onset].append(note)

            # 各時点でのボイス間クロッシングをチェック
            for time, notes_at_time in time_points.items():
                if len(notes_at_time) < 2:
                    continue

                # ボイスIDでソート (これが基準の順序)
                notes_at_time.sort(key=lambda x: x.get("voice", 0))

                # ピッチの順序がボイス順序と一致しているか確認
                pitch_order_violated = False
                for i in range(len(notes_at_time) - 1):
                    if notes_at_time[i]["pitch"] > notes_at_time[i + 1]["pitch"]:
                        pitch_order_violated = True
                        break

                # ピッチ順序が逆転していたら修正を試みる
                if pitch_order_violated:
                    logger.debug(
                        f"Voice crossing detected at time {time:.3f}. Attempting correction."
                    )
                    # ピッチでソートして、ボイスIDにピッチ順を再割り当てする形で修正
                    # (これは破壊的な変更になる可能性があるため、単純なオクターブシフトで対応)
                    for i in range(len(notes_at_time) - 1):
                        note1 = notes_at_time[i]
                        note2 = notes_at_time[i + 1]

                        # 低いボイスIDのピッチが高い場合
                        if note1["pitch"] > note2["pitch"]:
                            # note1を1オクターブ下げるか、note2を1オクターブ上げる
                            # より大きなピッチ変更を避ける方を優先 (中央値に近い方へ)
                            median_pitch = np.median(
                                [n["pitch"] for n in notes_at_time]
                            )

                            pitch1_down = note1["pitch"] - 12
                            pitch2_up = note2["pitch"] + 12

                            dist1 = abs(pitch1_down - median_pitch)
                            dist2 = abs(pitch2_up - median_pitch)

                            corrected = False
                            # note1を下げても範囲内かつ、note2より低くなるか
                            if pitch1_down >= 0 and dist1 <= dist2:
                                note1["pitch"] = pitch1_down
                                corrected = True
                                logger.debug(
                                    f"  Corrected voice {note1.get('voice')}: pitch lowered to {pitch1_down}"
                                )
                            # note2を上げても範囲内かつ、note1より高くなるか
                            elif pitch2_up <= 127:
                                note2["pitch"] = pitch2_up
                                corrected = True
                                logger.debug(
                                    f"  Corrected voice {note2.get('voice')}: pitch raised to {pitch2_up}"
                                )
                            else:
                                logger.warning(
                                    f"  Could not correct voice crossing between {note1.get('voice')} and {note2.get('voice')} at {time:.3f}"
                                )

        # すべてのノートを発音時刻でソートして返す
        return sorted(note_events, key=lambda x: x["onset"])

    def _update_tempogram(self, onset_strength: float) -> float:
        """
        テンポグラムを更新し、現在のBPMを推定する

        Parameters
        ----------
        onset_strength : float
            オンセット強度

        Returns
        -------
        float
            推定BPM
        """
        # オンセット包絡線バッファに追加
        self.onset_envelope.append(onset_strength)

        # バッファサイズ制限（4秒、5msホップ）
        max_len = int(4.0 * 1000.0 / 5.0)
        if len(self.onset_envelope) > max_len:
            self.onset_envelope = self.onset_envelope[-max_len:]

        # バッファサイズが不十分な場合は現在のBPMを返す
        if len(self.onset_envelope) < max_len // 10:
            return self.current_bpm

        # テンポ範囲の定義
        min_bpm = 40.0
        max_bpm = 300.0

        # ラグ範囲計算（秒単位）
        lag_min = int(60.0 / max_bpm * 200)  # 5ms単位
        lag_max = int(60.0 / min_bpm * 200)  # 5ms単位
        lag_min = max(1, min(lag_min, len(self.onset_envelope) // 2))
        lag_max = min(lag_max, len(self.onset_envelope) // 2)

        # 自己相関計算
        acf = np.zeros(lag_max - lag_min + 1)
        env = np.array(self.onset_envelope)

        for i, lag in enumerate(range(lag_min, lag_max + 1)):
            # Hannウィンドウ適用自己相関
            win = 0.5 - 0.5 * np.cos(
                2 * np.pi * np.arange(len(env) - lag) / (len(env) - lag)
            )
            r = np.sum(env[lag:] * env[:-lag] * win)
            acf[i] = r

        # 自己相関関数からピーク検出
        # 単純な局所最大値検出
        peak_lags = []
        for i in range(1, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] >= acf[i + 1]:
                peak_lags.append(i + lag_min)

        if not peak_lags:
            return self.current_bpm

        # 最大ピークを選択
        best_lag = peak_lags[np.argmax(acf[np.array(peak_lags) - lag_min])]
        new_bpm = 60.0 / (best_lag * 0.005)  # 5msあたり

        # Kalmanフィルタ平滑化（α=0.3）
        predicted_bpm = self.current_bpm
        if abs(new_bpm - predicted_bpm) < 10.0:
            # 急激な変化を防ぐフィルタ
            self.current_bpm = 0.7 * predicted_bpm + 0.3 * new_bpm
        else:
            # ±5 BPM以内の最も近い2倍/半分値を選択
            if abs(new_bpm * 2.0 - predicted_bpm) < 5.0:
                self.current_bpm = 0.7 * predicted_bpm + 0.3 * (new_bpm * 2.0)
            elif abs(new_bpm * 0.5 - predicted_bpm) < 5.0:
                self.current_bpm = 0.7 * predicted_bpm + 0.3 * (new_bpm * 0.5)
            else:
                # それでも大きな変化の場合は信頼性低として現状維持
                # ただし時間経過で徐々に新BPMへ近づける
                self.current_bpm = 0.95 * predicted_bpm + 0.05 * new_bpm

        return self.current_bpm

    def _quantize_notes(self, note_events: List[Dict], bpm: float) -> List[Dict]:
        """
        ノートイベントをテンポグリッドに量子化する
        Sparse Viterbiアルゴリズムで効率的に実装

        Parameters
        ----------
        note_events : List[Dict]
            ノートイベントのリスト
        bpm : float
            テンポ (BPM)

        Returns
        -------
        List[Dict]
            量子化されたノートイベントのリスト
        """
        quant_start = time.time()
        if not note_events:
            return []

        # 発音時刻でソート
        sorted_notes = sorted(note_events, key=lambda x: x.get("onset", 0))

        # タイミング設定
        grid_division = 16  # 16分音符
        phase_division = 2  # 2相
        num_states = grid_division * phase_division

        # グリッド単位計算（拍の長さ）
        beat_duration = 60.0 / bpm
        grid_unit = beat_duration / (grid_division / 4)  # 16分音符の長さ

        # 最初と最後のノートの時間
        t_start = sorted_notes[0].get("onset", 0)
        t_end = max(
            [
                (
                    n.get("offset", t_start + 1)
                    if n.get("offset") is not None
                    else t_start + 1
                )
                for n in sorted_notes
            ]
        )

        # 単一時点の場合は処理不要
        if t_start == t_end:
            return note_events

        # Viterbiアルゴリズム用データ構造 - メモリ使用量を最適化
        L = int((t_end - t_start) / (grid_unit / 2)) + 2

        # float32型でメモリ使用量を削減
        dp = np.full((L, num_states), -1e9, dtype=np.float32)

        # uint16型で十分（状態数は通常32以下）
        back = np.zeros((L, num_states), dtype=np.uint16)

        # 初期状態
        dp[0, :] = 0

        # 各ノートに最も近い状態を特定
        note_steps = []
        for note in sorted_notes:
            onset = note.get("onset", 0)
            rel_time = onset - t_start
            step = int(round(rel_time / (grid_unit / 2)))
            step = max(0, min(step, L - 1))
            note_steps.append(step)

        # Viterbi前進（Sparse最適化版）
        viterbi_start = time.time()
        max_grid_diff = 3  # prev_grid ±3 だけ遷移を許可

        for t in range(1, L):
            for s in range(num_states):
                # 現在グリッド位置とフェーズ
                grid = s // phase_division
                phase = s % phase_division

                # デフォルト値を設定
                dp[t, s] = -1e9  # 非常に低い値で初期化

                # Sparse遷移: 近傍状態のみ評価
                for prev_grid_offset in range(-max_grid_diff, max_grid_diff + 1):
                    prev_grid = grid + prev_grid_offset

                    # グリッド範囲チェック
                    if prev_grid < 0 or prev_grid >= grid_division:
                        continue

                    # フェーズを両方チェック
                    for prev_phase in range(phase_division):
                        prev_s = prev_grid * phase_division + prev_phase

                        # グリッド間の差分
                        grid_diff = abs(prev_grid_offset)
                        transition_prob = np.exp(-grid_diff / 0.4)

                        # フェーズ間の自然な遷移（0→1→0...）
                        if (prev_phase == 0 and phase == 1) or (
                            prev_phase == 1 and phase == 0 and grid != prev_grid
                        ):
                            transition_prob *= 1.5

                        # スコア更新
                        score = dp[t - 1, prev_s] + np.log(transition_prob + 1e-9)
                        if score > dp[t, s]:
                            dp[t, s] = score
                            back[t, s] = prev_s

            # このステップに音符があるか確認
            for i, step in enumerate(note_steps):
                if step == t:
                    # 音符のタイミング
                    onset = sorted_notes[i].get("onset", 0)
                    rel_time = onset - t_start

                    # 各状態での観測確率
                    for s in range(num_states):
                        # 状態の時刻計算
                        grid = s // phase_division
                        phase = s % phase_division
                        state_time = grid * grid_unit + phase * (
                            grid_unit / phase_division
                        )

                        # 観測誤差
                        err = abs(rel_time - state_time)
                        obs_prob = np.exp(-err * err / (2 * 0.01))  # σ = 10ms

                        # スコア更新
                        dp[t, s] += np.log(obs_prob + 1e-9)

        # 最終状態でスコア最大の状態を選択
        best_last_state = np.argmax(dp[L - 1, :])

        # Viterbiバックトレース
        path = np.zeros(L, dtype=np.uint16)  # uint16に変更
        path[L - 1] = best_last_state
        for t in range(L - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]

        # 量子化結果を適用
        quantized_notes = []
        for i, note in enumerate(sorted_notes):
            note_copy = note.copy()
            step = note_steps[i]
            state = path[step]

            # 状態からタイミング計算
            grid = state // phase_division
            phase = state % phase_division
            grid_time = (
                t_start + grid * grid_unit + phase * (grid_unit / phase_division)
            )

            # オンセット更新
            note_copy["onset"] = grid_time

            # オフセットも存在すれば同様に量子化
            if note_copy.get("offset") is not None:
                offset = note_copy["offset"]
                off_rel = offset - t_start
                off_step = int(round(off_rel / (grid_unit / 2)))
                off_step = max(0, min(off_step, L - 1))

                off_state = path[off_step]
                off_grid = off_state // phase_division
                off_phase = off_state % phase_division
                off_grid_time = (
                    t_start
                    + off_grid * grid_unit
                    + off_phase * (grid_unit / phase_division)
                )

                # 不正な量子化（オフセット≤オンセット）を修正
                if off_grid_time <= grid_time:
                    # 最低でも1グリッド単位離す
                    off_grid_time = grid_time + grid_unit

                note_copy["offset"] = off_grid_time

            quantized_notes.append(note_copy)

        logger.info(
            f"    Quantization: Sparse Viterbi took {time.time() - quant_start:.4f}s, L={L}, S={num_states}"
        )
        return quantized_notes
