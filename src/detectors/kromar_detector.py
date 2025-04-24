"""
KROMAR（Kalman-Ridge-Optimized Moment Analysis Resolution）検出器

このモジュールは、高度な信号処理技術を使用して音楽ノートを検出する
KROMARアルゴリズムの実装を提供します。

アルゴリズムの特徴:
- Hilbert拡張とモーメント写像
- 時間-周波数解析（Morlet + Hermite fCWT、Sliding DFT）
- シンプレクティック散乱と群平均化
- Persistent Homologyによる位相リッジ抽出
- Diophantine Pitch Solver
- 整数カルマン-KZスムージング
"""

import itertools
import logging
import os
import resource
import sys
import time
import traceback
import warnings
from typing import Dict, List, Optional, Tuple, Union

# OpenMP スレッド数を制限（GUDHI/CGAL安定化）
os.environ["OMP_NUM_THREADS"] = "1"

# マルチプロセス起動方法をspawnに設定（リソーストラッカのセマフォリーク防止）
try:
    import multiprocessing

    # すでに設定されている場合を考慮
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
except (RuntimeError, AttributeError):
    # すでに設定されている場合や、古いPythonバージョンでは無視
    pass

# デバッグやクラッシュ診断用のfaulthandlerを有効化
import faulthandler

faulthandler.enable(all_threads=True)

# macOSの場合、GUDHIとfpylllのメモリ制限を設定
if sys.platform == "darwin":
    try:
        # 現在の制限を取得
        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_DATA)
        # 安全な値を設定（現在の制限を超えないようにする）
        new_limit = min(2**30, current_hard)
        resource.setrlimit(resource.RLIMIT_DATA, (new_limit, current_hard))
        logging.warning(
            f"macOSのためGUDHI/fpylllのメモリ制限を{new_limit/(2**20):.1f}MBに設定しました（セグメンテーションフォルト防止）"
        )
    except (ValueError, OSError) as e:
        logging.warning(
            f"メモリ制限の設定に失敗しました: {e}。デフォルト値を使用します。"
        )

import numpy as np
import osqp
import scipy.sparse as sparse
from scipy import fft, signal
from scipy.interpolate import interp1d

# 環境チェック - macOS + Python 3.10以上の場合はfpylllをスキップ
if sys.platform == "darwin" and sys.version_info >= (3, 10):
    HAS_FPYLLL = False
    logging.warning(
        "macOS + Python 3.10以上環境のため、fpylllを無効化します（セグメンテーションフォルト防止）"
    )
else:
    # P0: Diophantine Pitch Solver のためのライブラリ
    try:
        import fpylll

        HAS_FPYLLL = True
    except ImportError:
        HAS_FPYLLL = False
        logging.error(
            "fpylllライブラリが見つかりません。KROMARディテクターの高度なピッチ解決機能には必須のライブラリです。"
        )

# P1: Persistent Homology Ridge Extractor のためのライブラリ
try:
    import gudhi

    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False
    logging.error(
        "gudhiライブラリが見つかりません。KROMARディテクターの実行には必須のライブラリです。"
    )
    raise ImportError(
        "gudhiライブラリがインストールされていません。KROMARディテクターの実行には必須のライブラリです。"
    )

# GPU 処理のためのライブラリ（オプション）
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    logging.warning("cupyがインポートできませんでした。CPU処理にフォールバックします。")

from src.detectors import register_detector
from src.detectors.base_detector import BaseDetector

# ロガーの設定
logger = logging.getLogger(__name__)


@register_detector(
    name="KROMARDetector",
    description="高精度な多重F0検出と音符認識のためのKROMAR（Kalman-Ridge-Optimized Moment Analysis Resolution）アルゴリズム",
    version="1.0",
    params={
        "sample_rate": 48000,
        "hop_size": 512,
        "morlet_xi0": 5.33,
        "hermite_n": 3,
        "alpha_vr_radius": 0.05,  # 0.12→0.05に変更（仕様通りのVR半径）
        "group_avg_t_window": 0.05,  # 0.03→0.05 (仕様通りのτ窓幅)
        "group_avg_freq_range": 25,  # 15→25セント（仕様通りのν範囲）
        "life_ratio_min": 0.3,  # 0.005→0.3 (仕様通りのリッジ寿命比閾値)
        "lll_delta": 0.99,
        "kalman_q": 0.01,
        "kalman_r": 0.001,
    },
)
class KROMARDetector(BaseDetector):
    """
    KROMAR検出器クラス

    高度な信号処理技術を使用して音楽ノートを検出します。

    仕様書に基づく実装:
    1. 入力オーディオ信号（PCM）を解析信号化 (Hilbert変換等)
    2. 時間–周波数多様体上での複素連続ウェーブレット変換 (Morlet + Hermite fCWT) および Sliding DFT
    3. シンプレクティック散乱を用いた特徴量抽出と群平均化による時間・周波数平滑
    4. Persistent Homology を用いて位相リッジを抽出
    5. Diophantine格子上での最適化（分枝限定法 + QP緩和）によるピッチ（音高）同定
    6. 整数カルマン-KZ フィルタで時系列を平滑化
    7. ノートイベント出力（オンセット・オフセット判定）
    """

    def __init__(self, **kwargs):
        """
        KROMARDetectorの初期化

        Parameters
        ----------
        **kwargs
            検出器のパラメータ。以下のキーをサポート:
            - sample_rate: サンプルレート (Hz) [default: 48000]
            - hop_size: ホップサイズ (サンプル) [default: 512]
            - morlet_xi0: Morletウェーブレットの中心パラメータ [default: 5.33]
                シンプレクティック構造 Ω が最適になる設定
            - hermite_n: Hermite関数の次数 (奇数のみ使用) [default: 3]
            - alpha_vr_radius: Vietoris-Rips複体の半径 [default: 0.05]
                正規化座標系での値
            - group_avg_t_window: 群平均化の時間窓 (秒) [default: 0.05]
                τ窓幅 = 50 ms
            - group_avg_freq_range: 群平均化の周波数範囲 (cent) [default: 25]
                ν範囲 = ±25 cent
            - life_ratio_min: リッジ寿命比閾値 [default: 0.3]
                d-b >= life_ratio_min * b
            - lll_delta: LLLアルゴリズムのδパラメータ [default: 0.99]
                Lovász条件
            - kalman_q: カルマンフィルタのプロセスノイズ共分散 [default: 0.01]
            - kalman_r: カルマンフィルタの観測ノイズ共分散 [default: 0.001]
        """
        super().__init__(**kwargs)

        # 仕様書に基づくデフォルトパラメータ
        self.sample_rate = kwargs.get("sample_rate", 48000)
        self.hop_size = kwargs.get("hop_size", 512)
        self.morlet_xi0 = kwargs.get("morlet_xi0", 5.33)  # シンプレクティック構造最適値
        self.hermite_n = kwargs.get("hermite_n", 3)  # 奇次のみ使用
        self.alpha_vr_radius = kwargs.get(
            "alpha_vr_radius", 0.05
        )  # 0.12→0.05（仕様通りの値）
        self.group_avg_t_window = kwargs.get(
            "group_avg_t_window", 0.05
        )  # 0.03→0.05（仕様通りの値）
        self.group_avg_freq_range = kwargs.get(
            "group_avg_freq_range", 25
        )  # 15→25（仕様通りの値）
        self.life_ratio_min = kwargs.get(
            "life_ratio_min", 0.3
        )  # 0.005→0.3（仕様通りの値）
        self.lll_delta = kwargs.get("lll_delta", 0.99)  # Lovász条件
        self.kalman_q = kwargs.get("kalman_q", 0.01)  # プロセスノイズ
        self.kalman_r = kwargs.get("kalman_r", 0.001)  # 観測ノイズ

        # パラメータのコピーを保存
        self.params = {
            "sample_rate": self.sample_rate,
            "hop_size": self.hop_size,
            "morlet_xi0": self.morlet_xi0,
            "hermite_n": self.hermite_n,
            "alpha_vr_radius": self.alpha_vr_radius,
            "group_avg_t_window": self.group_avg_t_window,
            "group_avg_freq_range": self.group_avg_freq_range,
            "life_ratio_min": self.life_ratio_min,
            "lll_delta": self.lll_delta,
            "kalman_q": self.kalman_q,
            "kalman_r": self.kalman_r,
        }

        # 内部状態の初期化
        self._initialize_internal_state()

        # 検出結果用のバッファ
        self.completed_intervals = []
        self.completed_pitches = []

        logger.debug(f"KROMARDetector initialized with parameters: {self.params}")

    def _initialize_internal_state(self):
        """内部状態を初期化"""
        # 12平均律の基底周波数 (MIDI note 0 = 8.1758 Hz)
        self.base_freq = 8.1758

        # 12平均律の音高配列 (88鍵ピアノの範囲: MIDI notes 21-108)
        self.midi_notes = np.arange(21, 109)
        self.pitch_freqs = self.base_freq * (2 ** (self.midi_notes / 12))

        # カルマンフィルタの状態
        self.kalman_state = None
        self.kalman_cov = None

        # 検出バッファ
        self.note_buffer = []

        # 遅延バッファ（オンセット・オフセット確定用）
        self.delay_buffer_size = 3  # 約30ms (仕様書では128サンプル)
        self.state_buffer = []  # ピッチベクトルの履歴

        # 五度円距離に基づく状態遷移行列（A）の初期化
        self.transition_matrix = self._compute_fifth_circle_transition_matrix()

        # ピッチクラス（0-11、C=0, C#=1, ..., B=11）
        pitch_classes = np.mod(np.arange(21, 109), 12)

        # 隣接リストを事前計算
        self.pc_neighbors = {}
        for i in range(88):
            # 半音または全音の関係にあるノート（五度円上で隣接）
            self.pc_neighbors[i] = np.where(
                np.isin((pitch_classes - pitch_classes[i]) % 12, [1, 2, 10, 11])
            )[0]

        # ノート持続判定用状態
        self.active_notes = {}  # {midi_note: onset_time}
        self.min_duration = 0.03  # 30 ms 最低持続時間

        # 検出結果用バッファ
        self.completed_intervals = []  # [start_time, end_time]のリスト
        self.completed_pitches = []  # 周波数のリスト

    def _compute_fifth_circle_transition_matrix(self) -> np.ndarray:
        """
        五度円距離に基づく状態遷移行列を計算

        仕様書に基づく実装:
        A_{ij} = e^{-d_{ij}/2}
        d_{ij}は五度円上の距離

        Returns
        -------
        np.ndarray
            状態遷移行列 A (88x88)
        """
        n_notes = 88
        A = np.eye(n_notes)

        # MIDIノート番号に対応するピッチクラス（0-11、C=0, C#=1, ..., B=11）
        pitch_classes = np.mod(np.arange(21, 109), 12)

        # 遷移行列の計算
        for i in range(n_notes):
            for j in range(n_notes):
                if i == j:
                    # 対角成分は1.0
                    continue

                # ピッチクラス同士の五度円距離を計算
                pc_i = pitch_classes[i]
                pc_j = pitch_classes[j]

                # 五度円上での距離（0-6の範囲）
                # （ピッチクラスの五度円上の距離は最大で6）
                fifth_circle_dist = min((pc_i - pc_j) % 12, (pc_j - pc_i) % 12)

                # オクターブ差分
                octave_dist = abs(i - j) // 12

                # 合計距離（五度円+オクターブ）
                total_dist = fifth_circle_dist + 2 * octave_dist

                # 仕様書の式に従って: A_{ij} = e^{-d_{ij}/2}
                A[i, j] = np.exp(-total_dist / 2)

        return A

    def detect(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """
        オーディオデータから音符情報を検出

        Parameters
        ----------
        audio_data : np.ndarray
            入力オーディオデータ。形状は (サンプル数,) または (チャンネル数, サンプル数)。
        sample_rate : int
            オーディオデータのサンプルレート (Hz)

        Returns
        -------
        Dict[str, np.ndarray]
            以下のキーを持つ辞書:
            - 'onset_times': オンセット時間の配列
            - 'offset_times': オフセット時間の配列
            - 'pitches': 音高の配列
        """
        # サンプルレートが一致しない場合はリサンプリング
        if sample_rate != self.sample_rate:
            warnings.warn(
                f"サンプルレートが一致しません。{sample_rate} Hz を {self.sample_rate} Hz にリサンプリングします。"
            )
            audio_data = signal.resample_poly(audio_data, self.sample_rate, sample_rate)

        try:
            # 多チャンネルの場合は平均化
            if audio_data.ndim > 1 and audio_data.shape[0] > 1:
                audio_data = np.mean(audio_data, axis=0)
            elif audio_data.ndim > 1:
                audio_data = audio_data[
                    0
                ]  # 単一チャンネルの場合は最初のチャンネルを使用

            # 解析信号化 (Hilbert 変換)
            analytic_signal = self._hilbert_transform(audio_data)

            # 時間–周波数解析 (仕様に合わせた高度な実装に変更)
            tf_representation = self._advanced_time_frequency_analysis(analytic_signal)

            # シンプレクティック散乱 & 群平均化 (仕様に合わせた高度な実装に変更)
            scattered_tf = self._apply_symplectic_scatter_and_averaging(
                tf_representation
            )

            try:
                # Persistent Homology による位相リッジ抽出
                ridges = self._extract_persistent_homology_ridges(scattered_tf)

                # リッジから周波数を抽出
                observed_freqs = self._extract_frequencies_from_ridges(ridges)
                observed_mags = (
                    [ridge["magnitude"] for ridge in ridges] if ridges else []
                )

                # Diophantine Pitch Solver
                try:
                    pitch_vector = self._advanced_lattice_pitch_solver(
                        observed_freqs, observed_mags
                    )
                except RuntimeError as e:
                    logger.error(f"ピッチ解決に失敗: {e}")
                    # 失敗時は空のピッチベクトルを返す（フォールバックなし）
                    return {
                        "onset_times": np.array([]),
                        "offset_times": np.array([]),
                        "pitches": np.array([]),
                    }

                # 整数カルマン-KZ スムージング
                smoothed_pitch_vector = self._apply_kalman_kz_smoothing(pitch_vector)

                # ノートデータへの変換
                note_data = self._convert_to_note_data(smoothed_pitch_vector)

                return note_data

            except RuntimeError as e:
                logger.error(f"リッジ抽出に失敗: {e}")
                # リッジ抽出失敗時は空の結果を返す（フォールバックなし）
                return {
                    "onset_times": np.array([]),
                    "offset_times": np.array([]),
                    "pitches": np.array([]),
                }

        except Exception as e:
            # 予期せぬエラーが発生した場合
            logger.error(f"KROMARアルゴリズムの実行中にエラーが発生しました: {e}")
            # 空の結果を返す
            return {
                "onset_times": np.array([]),
                "offset_times": np.array([]),
                "pitches": np.array([]),
            }

    def _hilbert_transform(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Hilbert変換による解析信号生成

        Parameters
        ----------
        audio_data : np.ndarray
            入力オーディオデータ

        Returns
        -------
        np.ndarray
            解析信号 (複素数)
        """
        # FFTベースのHilbert変換
        analytic_signal = signal.hilbert(audio_data)

        # 過度な振幅潰しを避けるため、正規化を削除
        # 位相情報の取得が主目的なので、振幅スケールは保持

        return analytic_signal

    def _advanced_time_frequency_analysis(
        self, analytic_signal: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        KROMARアルゴリズムによる時間-周波数解析の高度な実装

        Parameters
        ----------
        analytic_signal : np.ndarray
            解析信号

        Returns
        -------
        Dict[str, np.ndarray]
            時間-周波数表現を含む辞書
        """
        # 信号を低周波・高周波領域に分割
        cutoff_freq = 1000  # Hz

        # --- Morlet & Hermite のバンド数を分割 ---
        # 低周波は合計48バンド（Morlet 24 + Hermite 24）、高周波は40バンドで合計88バンド
        n_scales_morlet = 24
        n_scales_hermite = 24
        min_freq = 20  # Hz
        max_freq = cutoff_freq

        # --- Morlet 用のスケールを作る ---
        scales_morlet = np.exp(
            np.linspace(
                np.log(self.sample_rate / max_freq),
                np.log(self.sample_rate / min_freq),
                n_scales_morlet,
            )
        )

        # --- Hermite 用のスケールを作る ---
        scales_hermite = np.exp(
            np.linspace(
                np.log(self.sample_rate / max_freq),
                np.log(self.sample_rate / min_freq),
                n_scales_hermite,
            )
        )

        # Morlet CWTを計算
        morlet_coefs = self._compute_morlet_cwt(analytic_signal, scales_morlet)

        # Hermite CWTを計算
        hermite_coefs = self._compute_hermite_cwt(analytic_signal, scales_hermite)

        # 高周波領域の周波数（40バンド）
        high_freqs = np.linspace(cutoff_freq, 8000, 40)

        # Sliding DFTを計算
        frame_size = 1024
        # 高周波領域のインデックスを取得
        freq_indices = np.zeros(40, dtype=int)
        fft_freqs = fft.fftfreq(frame_size, d=1.0 / self.sample_rate)
        fft_freqs = fft_freqs[: frame_size // 2 + 1]  # 正の周波数のみ
        for i, target_freq in enumerate(high_freqs):
            # 最も近い周波数のインデックスを探す
            freq_indices[i] = np.argmin(np.abs(fft_freqs - target_freq))

        sdft_coefs = self._sliding_dft(
            analytic_signal, frame_size, self.hop_size, freq_bins=freq_indices
        )

        # 時間軸
        signal_len = len(analytic_signal)
        hop = self.hop_size
        n_frames = 1 + (signal_len - hop) // hop
        times = np.arange(n_frames) * hop / self.sample_rate

        # SDFT結果の時間軸と他の変換結果の時間軸が一致するか確認
        sdft_times = sdft_coefs["times"]
        if len(sdft_times) != len(times) or not np.allclose(sdft_times, times):
            # 時間軸が異なる場合、SDFTの結果をリサンプリング
            sdft_magnitude = sdft_coefs["magnitude"]
            sdft_phase = sdft_coefs["phase"]
            resampled_sdft_magnitude = np.zeros((sdft_magnitude.shape[0], len(times)))
            resampled_sdft_phase = np.zeros((sdft_phase.shape[0], len(times)))

            # 各周波数バンドごとにリサンプリング
            for i in range(sdft_magnitude.shape[0]):
                # 振幅の補間
                mag_interp = interp1d(
                    sdft_times,
                    sdft_magnitude[i],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0,
                )
                resampled_sdft_magnitude[i] = mag_interp(times)

                # 位相の補間
                phase_interp = interp1d(
                    sdft_times,
                    sdft_phase[i],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0,
                )
                resampled_sdft_phase[i] = phase_interp(times)

            # リサンプリングした結果で置き換え
            sdft_coefs["magnitude"] = resampled_sdft_magnitude
            sdft_coefs["phase"] = resampled_sdft_phase
            sdft_coefs["times"] = times

        # Morlet と Hermite の中心周波数を計算
        low_freq_centers_morlet = self.sample_rate / scales_morlet  # shape (24,)
        low_freq_centers_hermite = self.sample_rate / scales_hermite  # shape (24,)

        # 周波数を結合：低周波（Morlet + Hermite = 48）＋高周波（40）= 88
        all_low_freqs = np.concatenate(
            [low_freq_centers_morlet, low_freq_centers_hermite]
        )
        all_freqs = np.concatenate([all_low_freqs, high_freqs])  # shape (88,)

        # CWT係数とSDFT係数を時間方向に補間して結合
        morlet_magnitudes = np.abs(morlet_coefs)  # shape (24, T_morlet)
        hermite_magnitudes = np.abs(hermite_coefs)  # shape (24, T_hermite)
        sdft_magnitudes = np.abs(sdft_coefs["magnitude"])  # shape (40, n_frames)

        # 低周波CWT係数を時間でリサンプリング
        morlet_times = np.linspace(
            0, signal_len / self.sample_rate, morlet_coefs.shape[1]
        )
        hermite_times = np.linspace(
            0, signal_len / self.sample_rate, hermite_coefs.shape[1]
        )

        # リサンプリング結果格納用
        resampled_morlet = np.zeros((n_scales_morlet, len(times)), dtype=complex)
        resampled_hermite = np.zeros((n_scales_hermite, len(times)), dtype=complex)

        # Morletの各スケールを補間
        for i in range(n_scales_morlet):
            # 振幅の補間
            morlet_mag_interp = interp1d(
                morlet_times,
                morlet_magnitudes[i],
                kind="cubic",
                bounds_error=False,
                fill_value=0,
            )

            # 位相の補間
            morlet_phase = np.angle(morlet_coefs[i])
            morlet_phase_interp = interp1d(
                morlet_times,
                morlet_phase,
                kind="linear",
                bounds_error=False,
                fill_value=0,
            )

            # 補間した振幅と位相から複素係数を再構成
            resampled_morlet[i] = morlet_mag_interp(times) * np.exp(
                1j * morlet_phase_interp(times)
            )

        # Hermiteの各スケールを補間
        for i in range(n_scales_hermite):
            # 振幅の補間
            hermite_mag_interp = interp1d(
                hermite_times,
                hermite_magnitudes[i],
                kind="cubic",
                bounds_error=False,
                fill_value=0,
            )

            # 位相の補間
            hermite_phase = np.angle(hermite_coefs[i])
            hermite_phase_interp = interp1d(
                hermite_times,
                hermite_phase,
                kind="linear",
                bounds_error=False,
                fill_value=0,
            )

            # 補間した振幅と位相から複素係数を再構成
            resampled_hermite[i] = hermite_mag_interp(times) * np.exp(
                1j * hermite_phase_interp(times)
            )

        # 低周波（Morlet 24 + Hermite 24 = 48）と高周波（40）を結合して合計88バンド
        all_magnitudes = np.vstack(
            [
                np.abs(resampled_morlet),  # shape (24, n_frames)
                np.abs(resampled_hermite),  # shape (24, n_frames)
                sdft_magnitudes,  # shape (40, n_frames)
            ]
        )  # 最終 shape: (88, n_frames)

        all_phases = np.vstack(
            [
                np.angle(resampled_morlet),  # shape (24, n_frames)
                np.angle(resampled_hermite),  # shape (24, n_frames)
                sdft_coefs["phase"],  # shape (40, n_frames)
            ]
        )  # 最終 shape: (88, n_frames)

        # 確認: all_freqs.shape[0] == all_magnitudes.shape[0] == 88
        assert (
            all_freqs.shape[0] == all_magnitudes.shape[0]
        ), f"周波数軸とマグニチュード軸のサイズが不一致: {all_freqs.shape[0]} != {all_magnitudes.shape[0]}"

        return {
            "magnitude": all_magnitudes,  # shape (88, n_frames)
            "phase": all_phases,  # shape (88, n_frames)
            "frequencies": all_freqs,  # shape (88,)
            "times": times,
            "morlet_coefs": resampled_morlet,
            "hermite_coefs": resampled_hermite,
            "sdft_coefs": sdft_coefs,
        }

    def _compute_scattering_coefficients(
        self, tf_representation: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        散乱係数を計算する

        仕様書に基づく実装:
        S¹ = |W|  (1階層散乱係数)
        S² = S¹(S¹) (2階層散乱係数)

        Parameters
        ----------
        tf_representation : Dict[str, np.ndarray]
            時間-周波数表現

        Returns
        -------
        Dict[str, np.ndarray]
            散乱係数S¹とS²を含む辞書
        """
        # 第1階の散乱係数 S¹: 振幅スペクトログラム
        S1 = tf_representation["magnitude"]

        # 時間-周波数情報を取得
        n_freqs, n_times = S1.shape
        times = tf_representation["times"]
        frequencies = tf_representation["frequencies"]

        # 第2階の散乱係数 S² を計算するためのパラメータ
        # 短時間フーリエ変換の窓長 (約50ms)
        win_length = max(8, int(0.05 * self.sample_rate / self.hop_size))

        # 第2階の散乱係数 S² を初期化
        S2 = np.zeros((n_freqs, n_times))

        # 各周波数バンドに対して第2階層の散乱変換を計算
        for i in range(n_freqs):
            # 現在の周波数バンドの時間変動 (S¹の行)
            band_envelope = S1[i, :]

            if len(band_envelope) > win_length:
                # バンドエンベロープのSTFT (短時間フーリエ変換)
                # S²はS¹に対するウェーブレット変換として定義される
                f, t, stft = signal.stft(
                    band_envelope,
                    fs=(
                        1.0 / (times[1] - times[0]) if len(times) > 1 else 1.0
                    ),  # 時間軸のサンプリングレート
                    window="hann",
                    nperseg=win_length,
                    noverlap=win_length // 2,
                    nfft=2 * win_length,
                )

                # STFT係数の絶対値を取得 (S² = |S¹(S¹)|)
                env_spec = np.abs(stft)

                # 低変調周波数成分のみ保持 (主要な構造を捉える)
                # 変調周波数の下位20%までを保持
                max_mod_idx = max(1, int(0.2 * len(f)))
                low_mod_components = env_spec[:max_mod_idx, :]

                # 変調スペクトル成分の重み付き平均
                weights = np.exp(
                    -np.arange(max_mod_idx) / (max_mod_idx / 3)
                )  # 指数減衰重み
                weights = weights / np.sum(weights)

                weighted_avg = np.tensordot(
                    weights, low_mod_components, axes=([0], [0])
                )

                # 時間軸のリサンプリング (S²をS¹と同じ時間軸に合わせる)
                if len(t) > 1 and len(times) > 1:
                    # 時間軸の線形補間
                    interp_func = interp1d(
                        t * (times[-1] - times[0]) + times[0],
                        weighted_avg,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    S2[i, :] = interp_func(times)
                else:
                    # 補間できない場合は単純にコピー
                    if len(weighted_avg) > 0:
                        S2[i, :] = np.tile(weighted_avg.mean(), n_times)
                    else:
                        S2[i, :] = np.zeros(n_times)
            else:
                # 短すぎる場合は平均値で代用
                S2[i, :] = np.full(n_times, np.mean(band_envelope))

        return {"S1": S1, "S2": S2, "frequencies": frequencies, "times": times}

    def _apply_symplectic_scatter_and_averaging(
        self, tf_representation: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        シンプレクティック散乱と群平均化の高度な実装

        仕様書に基づく実装:
        群平均化カーネル:
        Sバー(σ,κ,b) = (1/|Φ|)∑_{(τ,ν)∈Φ} S²(a=e^{σ/12}, b+τ)⋅1_{|ν|≤25 cent}

        τ窓幅: 50 ms
        ν範囲: ±25 cent

        Parameters
        ----------
        tf_representation : Dict[str, np.ndarray]
            時間-周波数表現

        Returns
        -------
        Dict[str, np.ndarray]
            散乱・平均化後の時間-周波数表現
        """
        # 散乱係数を計算
        scatter_coeffs = self._compute_scattering_coefficients(tf_representation)
        S1 = scatter_coeffs["S1"]
        S2 = scatter_coeffs["S2"]

        # 時間と周波数の軸を取得
        times = scatter_coeffs["times"]
        frequencies = scatter_coeffs["frequencies"]

        # 群平均化のパラメータ (仕様書に基づく)
        # 時間窓: 50 ms
        t_window_seconds = self.group_avg_t_window  # 50ms
        # 周波数範囲: ±25 cent
        freq_range_cents = self.group_avg_freq_range  # 25 cent

        # 時間窓をサンプル数に変換
        t_window_samples = max(
            1, int(t_window_seconds / (times[1] - times[0])) if len(times) > 1 else 1
        )

        # 周波数範囲をセント単位からレシオに変換 (1セント = 2^(1/1200))
        cent_ratio = 2 ** (freq_range_cents / 1200)

        # 準備: 群平均化されたS2 (出力)
        n_freqs, n_times = S2.shape
        averaged_S2 = np.zeros_like(S2)

        # 各時間-周波数点について群平均化を実行
        for i in range(n_freqs):
            # 現在の周波数
            current_freq = frequencies[i]

            # 周波数の範囲 (±25セント内)
            lower_freq = current_freq / cent_ratio
            upper_freq = current_freq * cent_ratio

            # 範囲内の周波数インデックスを見つける
            freq_indices = np.where(
                (frequencies >= lower_freq) & (frequencies <= upper_freq)
            )[0]

            # 時間方向の平均化
            for t in range(n_times):
                # 時間窓の範囲
                t_start = max(0, t - t_window_samples // 2)
                t_end = min(n_times, t + t_window_samples // 2 + 1)

                # 周波数と時間の両方で平均化
                if len(freq_indices) > 0:
                    # Φ内の点の散乱係数の平均
                    group_values = S2[np.ix_(freq_indices, range(t_start, t_end))]
                    if group_values.size > 0:
                        averaged_S2[i, t] = np.mean(group_values)
                    else:
                        averaged_S2[i, t] = S2[i, t]  # 該当点がない場合は元の値
                else:
                    averaged_S2[i, t] = S2[i, t]  # 周波数範囲内に点がない場合は元の値

        # シンプレクティック位相補正 (Ω = dt ∧ dω)
        phase = tf_representation["phase"]
        hop_time = self.hop_size / self.sample_rate  # ホップサイズ（秒）

        # 時間方向の位相勾配 (dφ/dt)
        dt_phase = np.diff(phase, axis=1, prepend=phase[:, :1])
        # [-π, π]の範囲に折り返し
        dt_phase = ((dt_phase + np.pi) % (2 * np.pi) - np.pi) / hop_time

        # 周波数方向の位相勾配 (dφ/dω)
        # 周波数差分を計算
        dfreq = np.diff(frequencies, prepend=frequencies[0])
        # ゼロ除算回避
        dfreq[dfreq == 0] = 1e-6

        # 周波数方向の位相差分
        dw_phase = np.diff(phase, axis=0, prepend=phase[:1, :])
        # [-π, π]の範囲に折り返し
        dw_phase = (dw_phase + np.pi) % (2 * np.pi) - np.pi
        # 周波数差で正規化
        dw_phase = dw_phase / dfreq[:, np.newaxis]
        # 無効値をクリーンアップ
        dw_phase[np.isnan(dw_phase) | np.isinf(dw_phase)] = 0

        # シンプレクティック形式 Ω = dt∧dω
        omega = dt_phase - dw_phase

        # 周波数の精緻化（シンプレクティック補正）
        epsilon = 0.2  # 閾値
        correction_mask = np.abs(omega) > epsilon

        # 補正: ω' = ω + Ω*hop_time
        refined_freqs_2d = np.tile(frequencies[:, np.newaxis], (1, n_times))
        refined_freqs_2d[correction_mask] += omega[correction_mask] * hop_time

        # 結果を返す
        result = tf_representation.copy()
        result["magnitude"] = averaged_S2  # 群平均化されたS2
        result["S1"] = S1  # 1階散乱係数
        result["S2"] = S2  # 2階散乱係数 (生)
        result["dt_phase"] = dt_phase
        result["dw_phase"] = dw_phase
        result["omega"] = omega
        result["refined_freqs_2d"] = refined_freqs_2d

        return result

    def _extract_persistent_homology_ridges(
        self, tf_representation: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        Persistent Homologyを用いた位相リッジ抽出の実装

        仕様書に基づく実装:
        - Vietoris–Rips半径 ε = 0.06（正規化座標）を用いてα-complexを生成
        - 永続的ホモロジーの寿命(d-b)が0.005b以上となる1次クラスター（1次ホモロジー成分）を
          位相的なリッジと判定

        Parameters
        ----------
        tf_representation : Dict[str, np.ndarray]
            時間-周波数表現

        Returns
        -------
        List[Dict]
            抽出されたリッジ情報のリスト

        Raises
        ------
        RuntimeError
            gudhiライブラリが利用できない場合、または有意なバーコードが見つからない場合
        """
        # gudhiライブラリが利用可能かチェック
        if not HAS_GUDHI:
            # エラーを発生させて処理を終了
            raise RuntimeError(
                "gudhiライブラリが利用できないためKROMARディテクターを実行できません。gudhiをインストールしてください。"
            )

        try:
            # 振幅を正規化
            magnitude = tf_representation["magnitude"]
            max_magnitude = np.max(magnitude)
            if max_magnitude <= 0:
                raise RuntimeError("振幅データが無効です（最大値 ≤ 0）")

            # 簡易Wienerゲインを適用してデノイズ
            spec = magnitude**2
            noise_psd = np.minimum.accumulate(
                spec, axis=1
            )  # 初期フレームを雑音として更新
            gain = np.maximum(0.05, 1 - noise_psd / (spec + 1e-9))
            magnitude *= gain**0.5

            normalized_magnitude = magnitude / np.max(magnitude)

            # 時間と周波数の軸
            times = tf_representation["times"]
            frequencies = tf_representation["frequencies"]

            # 振幅がしきい値を超えるピークを検出
            thresholds = self._calculate_dynamic_thresholds(
                normalized_magnitude, times, frequencies
            )

            # 時間-周波数空間での点群を生成
            points = []
            values = []

            # 重要なピークだけを選択
            for t_idx, t in enumerate(times):
                if t_idx >= len(thresholds):
                    continue

                # 現在のフレームのしきい値
                threshold = thresholds[t_idx]

                frame = normalized_magnitude[:, t_idx]

                # 帯域幅に適応した距離パラメータを設定
                band_bw = max(1, int(len(frequencies) / 400))  # 約2-3bin @8kHz

                # ローカルピークの検出
                peaks, properties = signal.find_peaks(
                    frame, height=threshold, distance=band_bw
                )

                if len(peaks) == 0:
                    continue  # このフレームにはピークがない

                for peak_idx in peaks:
                    # 時間の正規化（全体の長さで割る）
                    normalized_time = t / times[-1] if times[-1] > 0 else 0

                    # 周波数の正規化（対数スケール、A4=440Hz基準）
                    freq = frequencies[peak_idx]
                    reference_freq = self.base_freq * (
                        2 ** ((69 - 21) / 12)
                    )  # A4 = 440Hz
                    # 周波数の正規化を圧縮（0.5倍スケーリング）
                    normalized_freq = (
                        np.log2(freq / reference_freq) * 0.5 if freq > 0 else 0
                    )

                    # 点の情報を追加
                    points.append([normalized_time, normalized_freq])
                    values.append(
                        {
                            "time": t,
                            "time_idx": t_idx,
                            "freq_idx": peak_idx,
                            "frequency": freq,
                            "magnitude": frame[peak_idx],
                        }
                    )

                    # ピークの振幅に応じてレプリカを作成（SNR補償）
                    # レプリカ数を制限（セグメンテーションフォルト防止）
                    replicas = max(0, min(4, int(np.ceil(frame[peak_idx] * 8) - 1)))
                    for r in range(replicas):
                        # 小さくランダムにずらしたレプリカを追加
                        jitter_t = np.random.normal(0, 0.001)
                        jitter_f = np.random.normal(0, 0.001)
                        points.append(
                            [normalized_time + jitter_t, normalized_freq + jitter_f]
                        )
                        values.append(values[-1].copy())  # 同じ値情報を複製

            # 点群データが少ない場合はエラーを発生
            if len(points) < 3:
                logger.warning("十分な点群データがありません（3点未満）")
                raise RuntimeError(
                    "Persistent Homologyの実行に必要な点群データが不足しています（3点以上必要）"
                )

            # NumPy配列に変換
            points_array = np.array(points)

            # 数値安定性のチェック - NaNや無限大の値を処理
            if np.any(np.isnan(points_array)) or np.any(np.isinf(points_array)):
                logger.warning(
                    "無効な点群データ（NaNまたは無限大）を検出しました - 処理します"
                )
                points_array = np.nan_to_num(
                    points_array, nan=0.0, posinf=1.0, neginf=-1.0
                )

            # 重複点の除去（数値的安定性のため）
            original_count = len(points_array)
            points_array = np.unique(np.round(points_array, 6), axis=0)
            if len(points_array) < original_count:
                logger.debug(f"重複点を除去: {original_count} → {len(points_array)}点")
                # 対応する値情報も更新が必要だが、この実装では複雑になるため省略

            # 点群のサイズ制限（大規模データセットでのセグメンテーションフォルト防止）
            MAX_POINTS = 1000  # 安全な上限
            if len(points_array) > MAX_POINTS:
                logger.warning(
                    f"点群数が多すぎます（{len(points_array)}点）- {MAX_POINTS}点にサブサンプリングします"
                )
                indices = np.random.choice(len(points_array), MAX_POINTS, replace=False)
                points_array = points_array[indices]
                # 対応する値情報も更新
                values = [values[i] for i in indices]

            # デバッグ出力：点群データの詳細情報
            logger.debug(
                f"Persistent Homology点群データ: {len(points_array)}点, 周波数範囲: {np.min(frequencies):.1f}Hz-{np.max(frequencies):.1f}Hz"
            )

            # α-complex を作成
            # 時間長に比例するα半径に調整
            time_scale_factor = max(1.0, times[-1]) if len(times) > 0 else 1.0
            try:
                alpha_complex = gudhi.AlphaComplex(points=points_array)
                # α半径を縮小（セグメンテーションフォルト防止）
                alpha_square_max = (
                    self.alpha_vr_radius * 0.5
                ) ** 2  # 1/4のサイズに縮小
                simplex_tree = alpha_complex.create_simplex_tree(
                    max_alpha_square=alpha_square_max
                )

                # デバッグ出力：点群数とsimplex数を記録
                logger.debug(
                    f"Persistent Homology stats: points={len(points_array)}, simplices={simplex_tree.num_simplices()}, alpha_square_max={alpha_square_max:.6f}"
                )
            except Exception as e:
                logger.error(f"Alpha Complex作成に失敗: {e}")
                logger.warning("単純なピーク検出にフォールバックします")
                return self._extract_simple_peaks(tf_representation)

            # 1次元ホモロジーの計算（H_1）
            # 最小永続性（minimum persistence）を小さく設定して、
            # 後で寿命比率(d-b)/b >= life_ratio_minの条件でフィルタリング
            persistence = simplex_tree.persistence(
                homology_coeff_field=2, min_persistence=0.001
            )

            # 1次元(H1)の永続バーコードを取得（ループ構造）
            h1_bars_np = simplex_tree.persistence_intervals_in_dimension(1)

            # NumPy配列 → list(tuple) に変換して真偽値評価可能にする
            h1_bars = [tuple(bar) for bar in h1_bars_np]

            # デバッグ出力：H1バーコード数
            logger.debug(f"Persistent Homology H1 bars: count={len(h1_bars)}")

            # ライフ比率のヒストグラムを計算
            if len(h1_bars) > 0:
                ratios = [
                    (death - birth) / max(birth, 1e-9) for birth, death in h1_bars
                ]
                if len(ratios) > 0:
                    # 詳細なライフレシオの情報を記録
                    logger.debug(
                        f"ライフレシオ最大値: {max(ratios):.4f}, 最小値: {min(ratios):.4f}, 平均値: {np.mean(ratios):.4f}"
                    )
                    if len(ratios) >= 10:
                        top_ratios = sorted(ratios, reverse=True)[:10]
                        logger.debug(
                            f"上位10個のライフレシオ: {[f'{r:.4f}' for r in top_ratios]}"
                        )

                    # ヒストグラムのビンを作成
                    bins = [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, float("inf")]
                    hist, _ = np.histogram(ratios, bins=bins)
                    logger.debug(
                        f"Life ratio histogram: {list(zip(['0-0.002', '0.002-0.005', '0.005-0.01', '0.01-0.02', '0.02-0.05', '0.05-0.1', '0.1-0.2', '0.2+'], hist))}"
                    )

            # 寿命比率(d-b)/bで評価し、しきい値以上のバーを選択
            significant_bars = []
            for birth, death in h1_bars:
                life_ratio = (death - birth) / max(birth, 1e-9)  # 0除算防止
                if life_ratio >= self.life_ratio_min:
                    significant_bars.append((1, (birth, death)))

            # デバッグ出力：有意なバーの数
            logger.debug(
                f"Significant H1 bars (life_ratio >= {self.life_ratio_min}): count={len(significant_bars)}"
            )

            # 有意なバーがない場合はエラー発生
            if not significant_bars:
                logger.warning("有意な持続バーコードが見つかりません。")
                raise RuntimeError(
                    "ridge extraction failed: 有意なPersistent Homologyバーを検出できませんでした"
                )

            # ジェネレータを取得して位相リッジを構築
            ridges = []

            # バー区間キーでジェネレータを辞書化（GUDHI APIに合わせた修正）
            # GUDHI 3.4+互換性対応
            try:
                if hasattr(simplex_tree, "flag_persistence_generators"):
                    # 新しいAPI (GUDHI 3.4+)
                    simplex_tree.flag_persistence_generators()
                    gens_all = simplex_tree.persistence_generators()

                    # より安全な次元チェック（GUDHI 3.9+以降向け）
                    if hasattr(simplex_tree, "dimension"):
                        # dimension APIを使用して正確に1次元サイクルを抽出
                        gens_dim1 = []
                        for iv, gen in gens_all:
                            if len(gen) == 0:  # 空の生成器をスキップ
                                continue
                            try:
                                if simplex_tree.dimension(simplex=gen[0]) == 1:
                                    gens_dim1.append((iv, gen))
                            except Exception as e:
                                logger.warning(
                                    f"次元チェック中にエラー: {e} - スキップします"
                                )
                                continue
                    else:
                        # 従来の方法で安全に次元チェック
                        gens_dim1 = []
                        for iv, gen in gens_all:
                            if len(gen) == 0:  # 空の生成器をスキップ
                                continue
                            try:
                                if len(gen[0]) == 2:  # 二頂点シンプレックス = 1次元
                                    gens_dim1.append((iv, gen))
                            except Exception as e:
                                logger.warning(
                                    f"次元チェック中にエラー: {e} - スキップします"
                                )
                                continue
                else:
                    # 古いAPI (GUDHI 3.3以前)
                    gens_dim1 = simplex_tree.persistence_generators_in_dimension(1)

                # 生成器を辞書化
                gen_dict = {}
                for iv, gen in gens_dim1:
                    if len(gen) > 0:  # 空でない生成器のみ登録
                        gen_dict[tuple(iv)] = gen

                logger.debug(f"有効なジェネレータ数: {len(gen_dict)}")
            except Exception as e:
                logger.error(f"ジェネレータ取得に失敗: {e}")
                logger.warning("単純なピーク検出にフォールバックします")
                return self._extract_simple_peaks(tf_representation)

            # 各有意なバーに対応するジェネレータを取得
            for bar_idx, bar in enumerate(significant_bars):
                try:
                    # 対応する1次元ジェネレータを取得
                    bar_dim, bar_interval = bar
                    birth, death = bar_interval

                    # バー区間でジェネレータを安全に取得
                    gen_simplices = gen_dict.get((birth, death), [])

                    # 生成サイクルから頂点インデックスを抽出
                    vertices = set()
                    for simplex in gen_simplices:
                        vertices.update(simplex)

                    # 頂点に対応する点の情報を取得
                    ridge_points = []
                    for idx in vertices:
                        if 0 <= idx < len(values):
                            point_info = values[idx].copy()
                            point_info["ridge_id"] = bar_idx
                            point_info["birth"] = birth
                            point_info["death"] = death
                            point_info["life_ratio"] = (
                                (death - birth) / birth if birth > 0 else 0
                            )
                            ridge_points.append(point_info)

                    # 時間でソート
                    ridge_points.sort(key=lambda x: x["time"])

                    # リッジが3点以上あれば追加
                    if len(ridge_points) >= 3:
                        ridges.extend(ridge_points)

                except Exception as e:
                    logger.error(f"ジェネレータの取得に失敗: {e}")
                    continue

            # リッジがない場合はエラー
            if not ridges:
                logger.warning("有効なリッジを抽出できませんでした。")
                raise RuntimeError(
                    "ridge extraction failed: 有効なリッジを検出できませんでした"
                )

            return ridges

        except Exception as e:
            # セグメンテーションフォルト防止のため、単純なピーク検出にフォールバック
            logger.error(f"Persistent Homologyによるリッジ抽出に失敗: {e}")
            logger.error(traceback.format_exc())

            # シンプルなピーク検出アルゴリズムを使用
            logger.warning("単純なピーク検出にフォールバックします")
            return self._extract_simple_peaks(tf_representation)

    def _calculate_dynamic_thresholds(
        self, magnitude: np.ndarray, times: np.ndarray, freqs: np.ndarray
    ) -> np.ndarray:
        """
        モーメント写像に基づく動的しきい値を計算
        仕様書の数式：
        μ_t = Σ_{b,a} t |W(a,b)|^2 / Σ |W|^2
        μ_ω = Σ_{b,a} ω |W(a,b)|^2 / Σ |W|^2

        Parameters
        ----------
        magnitude : np.ndarray
            正規化された振幅スペクトログラム
        times : np.ndarray
            時間軸
        freqs : np.ndarray
            周波数軸

        Returns
        -------
        np.ndarray
            各時間フレームに対するしきい値
        """
        n_freqs, n_times = magnitude.shape

        # --- 雑音適応型閾値へ変更 ---
        # ノイズレール推定（周波数方向の中央値）
        noise_floor = np.median(magnitude, axis=0)

        # MAD（Median Absolute Deviation）からノイズσを推定
        sigma = 1.4826 * np.median(
            np.abs(magnitude - noise_floor[np.newaxis, :]), axis=0
        )

        # SNR 10dB程度を想定した係数
        k = 4.0

        # 基本閾値: k·σ_noise （従来の1.4*meanを置換）
        base_threshold = k * sigma

        if np.all(base_threshold <= 0):
            # 推定に失敗した場合は安全な閾値を設定
            base_threshold = 0.1 * np.ones(n_times)

        # --- 以下、モーメント写像に基づく動的補正 ---
        # エネルギー（振幅の二乗）
        energy = magnitude**2
        total_energy = np.sum(energy)

        if total_energy <= 0:
            return base_threshold

        # 時間モーメント μ_t = Σ_{b,a} t |W(a,b)|^2 / Σ |W|^2
        t_grid, f_grid = np.meshgrid(times, freqs)
        mu_t = np.sum(t_grid * energy) / total_energy

        # 周波数モーメント μ_ω = Σ_{b,a} ω |W(a,b)|^2 / Σ |W|^2
        mu_omega = np.sum(f_grid * energy) / total_energy

        # 時間モーメントからの偏差に基づく動的係数
        time_deviation = times - mu_t
        alpha = 5.0  # 感度パラメータ
        time_factor = 1.0 + 0.5 * np.tanh(alpha * time_deviation)

        # 周波数モーメントを考慮した係数
        # 各フレームのスペクトルセントロイド
        centroids = np.zeros(n_times)
        for t in range(n_times):
            if np.sum(magnitude[:, t]) > 0:
                centroids[t] = np.sum(freqs * magnitude[:, t]) / np.sum(magnitude[:, t])
            else:
                centroids[t] = mu_omega

        freq_deviation = (
            (centroids - mu_omega) / mu_omega if mu_omega > 0 else np.zeros(n_times)
        )
        beta = 3.0
        freq_factor = 1.0 + 0.3 * np.tanh(beta * freq_deviation)

        # フレームごとのしきい値（時間・周波数モーメントを考慮）
        # 過補正を防ぐためframe_energy_ratioは使用しない
        thresholds = base_threshold * time_factor * freq_factor

        # しきい値の下限を設定
        min_threshold = 0.3 * np.mean(base_threshold)
        thresholds = np.maximum(thresholds, min_threshold)

        return thresholds

    def _extract_simple_peaks(
        self, tf_representation: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        シンプルなピーク検出に基づくリッジ抽出（フォールバック用）

        Parameters
        ----------
        tf_representation : Dict[str, np.ndarray]
            時間-周波数表現

        Returns
        -------
        List[Dict]
            抽出されたリッジ情報のリスト
        """
        magnitude = tf_representation["magnitude"]
        frequencies = tf_representation["frequencies"]
        times = tf_representation["times"]

        # しきい値（平均振幅の1.5倍）
        threshold = 1.5 * np.mean(magnitude)

        ridges = []
        ridge_id = 0

        # 各時間フレームでピークを検出
        for t_idx, t in enumerate(times):
            frame = magnitude[:, t_idx]

            # ローカルピークの検出
            peaks, _ = signal.find_peaks(frame, height=threshold, distance=3)

            if len(peaks) > 0:
                for peak_idx in peaks:
                    ridges.append(
                        {
                            "time": t,
                            "time_idx": t_idx,
                            "freq_idx": peak_idx,
                            "frequency": frequencies[peak_idx],
                            "magnitude": frame[peak_idx],
                            "ridge_id": ridge_id,
                        }
                    )
                    ridge_id += 1

        return ridges

    def _advanced_lattice_pitch_solver(
        self, observed_freqs: List[float], observed_magnitudes: List[float] = None
    ) -> np.ndarray:
        """
        高度なDiophantine Pitch Solver

        仕様書に基づく実装:
        1. 観測周波数集合をリッジから抽出
        2. 12平均律格子に対してLLL（Lovász条件付き基底還元）で基底を還元
        3. 分枝限定法(B&B)を用いて最適化:
           min_{n∈{0,1}^|F|} ||Pn - ω^||₂²

        Parameters
        ----------
        observed_freqs : List[float]
            観測された周波数リスト
        observed_magnitudes : List[float], optional
            各周波数の振幅

        Returns
        -------
        np.ndarray
            ピッチベクトル (88次元バイナリベクトル)
        """
        # fpylllライブラリが無い場合は直接QPソルバーを使用
        if not HAS_FPYLLL:
            logger.warning("fpylllが利用できないため、QP-onlyパスを使用します")
            # 入力チェック
            if not observed_freqs or len(observed_freqs) == 0:
                return np.zeros(88, dtype=np.int32)

            # 振幅情報がない場合は1.0で初期化
            if observed_magnitudes is None:
                observed_magnitudes = np.ones(len(observed_freqs))

            # QP-onlyソルバーを呼び出し
            valid_freqs = np.array(observed_freqs)
            valid_magnitudes = np.array(observed_magnitudes)
            note_freqs = self.pitch_freqs

            # 参照周波数
            ref_freq = 440.0  # A4

            # 測定行列P（観測周波数と音符周波数の割合）
            P = np.zeros((len(valid_freqs), len(note_freqs)))
            for i, freq in enumerate(valid_freqs):
                P[i, :] = note_freqs / freq

            # 重み（振幅から計算）
            weights = np.array(valid_magnitudes) / np.sum(valid_magnitudes)

            # QP-onlyソルバーを使用
            return self._solve_qp_relaxation(
                valid_freqs, valid_magnitudes, note_freqs, P, ref_freq, weights
            )

        try:
            if not observed_freqs:
                raise RuntimeError("観測周波数が空です")

            # 振幅が指定されていない場合は1.0とする
            if observed_magnitudes is None:
                observed_magnitudes = [1.0] * len(observed_freqs)

            # 観測周波数を整理（有効な周波数のみ）
            valid_indices = [i for i, f in enumerate(observed_freqs) if f > 0]
            if not valid_indices:
                raise RuntimeError("有効な周波数がありません")

            valid_freqs = np.array([observed_freqs[i] for i in valid_indices])
            valid_magnitudes = np.array([observed_magnitudes[i] for i in valid_indices])

            # 高度なピッチ解決アルゴリズム
            if HAS_FPYLLL and len(valid_freqs) > 0:
                # ===== 1. 参照ノート周波数の計算 =====
                # MIDI note 21 (A0, 27.5 Hz) ～ MIDI note 108 (C8, 4186.0 Hz)
                note_freqs = self.base_freq * (2 ** (np.arange(21, 109) / 12))

                # 参照周波数（A4 = 440Hz付近のMIDIノート69）
                ref_freq = note_freqs[69 - 21]

                # ===== 2. 測定行列Pの構築 =====
                # 測定行列 P の導入 (n_freqs x 88)
                # 各行: 観測周波数に対する振幅応答
                n_freqs = len(valid_freqs)
                P = np.zeros((n_freqs, 88))

                for i, freq in enumerate(valid_freqs):
                    # 各周波数の倍音構造を考慮
                    # 基本周波数からの偏差によるガウシアン応答
                    # 半音±50セント内で応答するフィルタ
                    for j, note_freq in enumerate(note_freqs):
                        cents_diff = 1200 * np.log2(freq / note_freq)
                        P[i, j] = np.exp(-((cents_diff / 50) ** 2))

                # 正規化された周波数ベクトル
                norm_freqs = valid_freqs / ref_freq

                # 重み付け
                weights = (
                    valid_magnitudes / np.sum(valid_magnitudes)
                    if np.sum(valid_magnitudes) > 0
                    else np.ones(n_freqs) / n_freqs
                )

                # ===== 3. LLL基底還元 =====
                # 仕様書に従ってLLL基底還元を実装
                try:
                    # fpylllを用いたLLL還元の準備
                    # 12平均律格子（12-ET格子）をLLL還元

                    # 格子基底行列の構築
                    # 12平均律のユニタリー行列
                    # 12列の行列で、各列は音程関係を表す（C, C#, ..., B）
                    B = np.zeros((12, 12), dtype=np.float64)
                    for i in range(12):
                        for j in range(12):
                            # 12平均律の行列要素 e^{2πi*j*k/12}
                            B[i, j] = np.cos(2 * np.pi * i * j / 12)

                    # fpylllを使用してLLL基底還元
                    # スケーリングして整数行列に変換
                    scale_factor = 1000
                    B_scaled = np.round(B * scale_factor).astype(np.int64)

                    try:
                        # 整数行列をfpylll形式に変換
                        from fpylll import LLL, IntegerMatrix

                        B_fpylll = IntegerMatrix.from_matrix(B_scaled.tolist())

                        # LLLを安全に実行するラッパー関数
                        def _lll_safe(matrix, delta=0.99):
                            try:
                                LLL.reduction(matrix, delta=delta)
                                return np.array(matrix.to_matrix(), dtype=np.float64)
                            except (RuntimeError, SystemError) as e:
                                logger.warning(
                                    f"LLL還元に失敗: {e} - QP-onlyパスにフォールバックします"
                                )
                                raise

                        # 安全なLLL還元を実行
                        B_reduced_matrix = _lll_safe(B_fpylll, self.lll_delta)
                        B_reduced = B_reduced_matrix / scale_factor

                        # 転置して、周波数空間でのプロジェクション行列として使用
                        P_reduced = P @ B_reduced.T

                        # LLL還元結果を活用した二次計画法緩和
                        continuous_sol = self._solve_qp_relaxation_lll(
                            valid_freqs,
                            valid_magnitudes,
                            note_freqs,
                            P,
                            P_reduced,
                            ref_freq,
                            weights,
                        )

                    except Exception as e:
                        # LLL還元に失敗した場合は通常のQPにフォールバック
                        logger.warning(
                            f"LLL還元処理に失敗したためQP-onlyパスにフォールバックします: {e}"
                        )
                        return self._solve_qp_relaxation(
                            valid_freqs,
                            valid_magnitudes,
                            note_freqs,
                            P,
                            ref_freq,
                            weights,
                        )

                    # ===== 4. 分枝限定法(B&B)による最適化 =====
                    # 還元空間を考慮した分枝限定法
                    pitch_vector = self._branch_and_bound_optimize(
                        continuous_sol,
                        valid_freqs,
                        valid_magnitudes,
                        P,
                        ref_freq,
                        weights,
                    )

                    return pitch_vector

                except Exception as e:
                    # LLL還元失敗時はQP-onlyパスにフォールバック
                    logger.warning(
                        f"高度なLLL処理に失敗しました: {e} - 通常のQPパスにフォールバックします"
                    )
                    return self._solve_qp_relaxation(
                        valid_freqs, valid_magnitudes, note_freqs, P, ref_freq, weights
                    )

            else:
                # fpylllがない場合やvalidFreqsが空の場合は通常のQPにフォールバック
                note_freqs = self.pitch_freqs
                ref_freq = 440.0  # A4

                # 測定行列
                P = np.zeros((len(valid_freqs), len(note_freqs)))
                for i, freq in enumerate(valid_freqs):
                    P[i, :] = note_freqs / freq

                # 重み
                weights = (
                    valid_magnitudes / np.sum(valid_magnitudes)
                    if np.sum(valid_magnitudes) > 0
                    else np.ones(len(valid_freqs)) / len(valid_freqs)
                )

                # QP-onlyパス
                return self._solve_qp_relaxation(
                    valid_freqs, valid_magnitudes, note_freqs, P, ref_freq, weights
                )

        except Exception as e:
            logger.error(f"高度なピッチ解決に失敗しました: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(
                f"高度なピッチ解決処理中に予期せぬエラーが発生しました: {str(e)}"
            )

    def _solve_qp_relaxation_lll(
        self,
        freqs: np.ndarray,
        magnitudes: np.ndarray,
        note_freqs: np.ndarray,
        P: np.ndarray,
        P_reduced: np.ndarray,
        ref_freq: float,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        LLL還元された空間での二次計画法による緩和問題

        Parameters
        ----------
        freqs : np.ndarray
            観測された周波数
        magnitudes : np.ndarray
            各周波数の振幅
        note_freqs : np.ndarray
            MIDI音符の周波数
        P : np.ndarray
            元の測定行列 (n_freqs x 88)
        P_reduced : np.ndarray
            LLL還元された測定行列
        ref_freq : float
            参照周波数
        weights : np.ndarray
            周波数の重み

        Returns
        -------
        np.ndarray
            緩和問題の解（連続値）
        """
        try:
            # 観測周波数のベクトル化
            n_freqs = len(freqs)
            if n_freqs == 0:
                return np.zeros(88, dtype=np.float64)

            # 正規化された周波数ベクトル
            norm_freqs = freqs / ref_freq

            # 重み付き周波数ベクトル
            omega = norm_freqs * weights.reshape(-1, 1)  # 列ベクトル化

            # ----- 還元空間でのQP問題 -----
            # 還元測定行列のサイズ
            n_reduced = P_reduced.shape[1]

            # Hessian行列: 2*P_reduced^T*W*P_reduced (W: 重み対角行列)
            W_diag = sparse.diags(weights)
            H_reduced = 2 * sparse.csc_matrix(P_reduced.T @ W_diag @ P_reduced)

            # 線形項: -2*P_reduced^T*W*omega
            g_reduced = -2 * P_reduced.T @ (W_diag @ omega).flatten()

            # 制約条件: 0 <= x_reduced <= 1
            A_reduced = sparse.eye(n_reduced).tocsc()
            l_reduced = np.zeros(n_reduced)
            u_reduced = np.ones(n_reduced)

            # OSQPソルバーの設定（還元空間）
            solver_reduced = osqp.OSQP()
            solver_reduced.setup(
                H_reduced,
                g_reduced,
                A_reduced,
                l_reduced,
                u_reduced,
                verbose=False,
                eps_abs=1e-4,
                eps_rel=1e-4,
            )

            # 問題を解く（還元空間）
            results_reduced = solver_reduced.solve()

            if results_reduced.info.status != "solved":
                raise ValueError(f"還元QP問題の解決失敗: {results_reduced.info.status}")

            # 還元空間での連続解
            continuous_sol_reduced = results_reduced.x

            # ----- 元の空間に戻す -----
            # LLL基底を使用して元の空間に戻す
            continuous_sol = P_reduced @ continuous_sol_reduced

            # 範囲を[0,1]に制限
            continuous_sol = np.clip(continuous_sol, 0, 1)

            return continuous_sol

        except Exception as e:
            logger.error(f"LLL-QP緩和問題の解決に失敗: {e}")
            # 通常のQP緩和にフォールバック
            return self._solve_qp_relaxation(
                freqs, magnitudes, note_freqs, P, ref_freq, weights
            )

    def _solve_qp_relaxation(
        self,
        freqs: np.ndarray,
        magnitudes: np.ndarray,
        note_freqs: np.ndarray,
        P: np.ndarray,
        ref_freq: float,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        二次計画法による緩和問題を解く

        Parameters
        ----------
        freqs : np.ndarray
            観測された周波数
        magnitudes : np.ndarray
            各周波数の振幅
        note_freqs : np.ndarray
            MIDI音符の周波数
        P : np.ndarray
            測定行列 (n_freqs x 88)
        ref_freq : float
            参照周波数
        weights : np.ndarray
            周波数の重み

        Returns
        -------
        np.ndarray
            ピッチベクトル (88要素の連続値)
        """
        try:
            # 観測周波数のベクトル化
            n_freqs = len(freqs)
            if n_freqs == 0:
                return np.zeros(88, dtype=np.float64)

            # 正規化された周波数ベクトル
            norm_freqs = freqs / ref_freq

            # 重み付き周波数ベクトル
            omega = norm_freqs * weights

            # 問題の定式化: min ||Px - omega||^2 subject to 0 <= x <= 1
            # Hessian行列: 2*P^T*W*P (W: 重み対角行列)
            W_diag = sparse.diags(weights)
            H = 2 * sparse.csc_matrix(P.T @ W_diag @ P)

            # 線形項: -2*P^T*W*omega
            g = -2 * P.T @ (W_diag @ omega)

            # 制約条件: 0 <= x <= 1
            A = sparse.eye(88).tocsc()
            l = np.zeros(88)
            u = np.ones(88)

            # OSQPソルバーの設定
            solver = osqp.OSQP()
            solver.setup(H, g, A, l, u, verbose=False, eps_abs=1e-4, eps_rel=1e-4)

            # 問題を解く
            results = solver.solve()

            if results.info.status != "solved":
                logger.warning(f"QP問題が解けませんでした: {results.info.status}")
                # 初期値として平均律に近い解を返す
                continuous_solution = np.zeros(88)
                for freq in freqs:
                    if freq <= 0:
                        continue
                    # 周波数から最も近いノートを見つける
                    cents = 1200 * np.log2(freq / self.base_freq)
                    midi_note = int(round(cents / 100))
                    if 21 <= midi_note <= 108:
                        continuous_solution[midi_note - 21] = 0.8
                return continuous_solution

            # 連続解（0～1の間の値）
            continuous_solution = results.x

            return continuous_solution

        except Exception as e:
            logger.error(f"QP緩和問題の解決に失敗: {e}")
            # 例外時にはシンプルな初期解を返す
            continuous_solution = np.zeros(88)
            for freq in freqs:
                if freq <= 0:
                    continue
                # 周波数から最も近いノートを見つける
                cents = 1200 * np.log2(freq / self.base_freq)
                midi_note = int(round(cents / 100))
                if 21 <= midi_note <= 108:
                    continuous_solution[midi_note - 21] = 0.8
            return continuous_solution

    def _branch_and_bound_optimize(
        self,
        continuous_sol: np.ndarray,
        freqs: np.ndarray,
        magnitudes: np.ndarray,
        P: np.ndarray,
        ref_freq: float,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        分枝限定法による整数解最適化

        仕様書に基づく実装:
        高速な分枝限定法を用いて整数解を探索

        Parameters
        ----------
        continuous_sol : np.ndarray
            QP緩和問題の連続解
        freqs : np.ndarray
            観測された周波数
        magnitudes : np.ndarray
            各周波数の振幅
        P : np.ndarray
            測定行列 (n_freqs x 88)
        ref_freq : float
            参照周波数
        weights : np.ndarray
            周波数の重み

        Returns
        -------
        np.ndarray
            最適化されたピッチベクトル (88要素の0/1ベクトル)
        """
        try:
            # コスト（0.5からの距離）を計算
            costs = np.abs(continuous_sol - 0.5)

            # 上位K個の変数を選定（不確実性の高い変数）
            # メモリ安全性とパフォーマンスのためにKを制限
            K = min(16, len(costs))

            # インデックスとコストのペアを作成
            index_cost_pairs = [(i, costs[i]) for i in range(len(costs))]

            # 0.5に最も近い変数（コストの小さい順）をK個選択
            selected_indices = sorted(index_cost_pairs, key=lambda x: x[1])[:K]
            selected_indices = [idx for idx, _ in selected_indices]

            # 初期解（連続解を0.5で閾値処理）
            initial_sol = (continuous_sol > 0.5).astype(np.int32)

            # 初期コスト計算
            norm_freqs = freqs / ref_freq
            best_sol = initial_sol.copy()
            best_cost = self._calculate_bnb_cost(best_sol, P, norm_freqs, weights)

            # 早期終了のためのカウンタ
            iter_count = 0
            max_iter_without_improvement = min(
                1000, 2**K
            )  # Kが大きいと組合せ爆発を防ぐ

            # itertools.productを使用して全ビットパターンを効率的に探索
            for bit_pattern in itertools.product(
                [0, 1], repeat=min(K, len(selected_indices))
            ):
                iter_count += 1

                # 現在の解を作成
                current_sol = initial_sol.copy()
                for j, idx in enumerate(selected_indices[: len(bit_pattern)]):
                    current_sol[idx] = bit_pattern[j]

                # コスト計算
                current_cost = self._calculate_bnb_cost(
                    current_sol, P, norm_freqs, weights
                )

                # より良い解が見つかれば更新
                if current_cost < best_cost:
                    best_sol = current_sol.copy()
                    best_cost = current_cost
                    iter_count = 0  # リセット
                elif iter_count > max_iter_without_improvement:
                    # 一定回数改善がなければ早期終了
                    break

            return best_sol

        except Exception as e:
            logger.error(f"分枝限定法最適化に失敗: {e}")
            # エラー時は閾値0.5で2値化
            binary_solution = (continuous_sol > 0.5).astype(np.int32)
            return binary_solution

    def _calculate_bnb_cost(
        self,
        solution: np.ndarray,
        P: np.ndarray,
        norm_freqs: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        分枝限定法のコスト関数

        Parameters
        ----------
        solution : np.ndarray
            ピッチベクトル (88要素の0/1ベクトル)
        P : np.ndarray
            測定行列
        norm_freqs : np.ndarray
            正規化された周波数
        weights : np.ndarray
            周波数の重み

        Returns
        -------
        float
            コスト値（二乗誤差）
        """
        # 推定周波数: P*x
        estimated = P @ solution

        # 重み付き二乗誤差: ||P*x - omega||_W^2
        # Wは重み対角行列
        weighted_error = weights * ((estimated - norm_freqs) ** 2)
        error = float(np.sum(weighted_error))

        # 和音の複雑さにペナルティを加える（アクティブノートが多すぎる場合）
        active_notes = np.sum(solution)
        complexity_penalty = 0.0

        # 5音以上の和音にはペナルティ
        if active_notes > 4:
            complexity_penalty = 0.05 * (active_notes - 4)

        # 音程の不協和度に基づくペナルティ
        # 活性化されたノート間の音程関係を評価
        dissonance_penalty = 0.0
        active_indices = np.where(solution > 0)[0]

        if len(active_indices) > 1:
            # 活性化されたノートのペア間で音程関係を評価
            for i, j in itertools.combinations(active_indices, 2):
                # 半音差（0～11）
                semitone_diff = (i - j) % 12
                if semitone_diff > 6:
                    semitone_diff = 12 - semitone_diff

                # 不協和度の高い音程（短2度、増4度など）にペナルティ
                if semitone_diff == 1 or semitone_diff == 6:
                    dissonance_penalty += 0.02

        # 総コスト
        total_cost = error + complexity_penalty + dissonance_penalty

        return total_cost

    def _simple_pitch_solver(
        self, freqs: np.ndarray, magnitudes: np.ndarray
    ) -> np.ndarray:
        """
        シンプルなピッチ解決アルゴリズム（フォールバック用）

        Parameters
        ----------
        freqs : np.ndarray
            観測された周波数
        magnitudes : np.ndarray
            各周波数の振幅

        Returns
        -------
        np.ndarray
            ピッチベクトル (88要素の0/1ベクトル)
        """
        # 各観測周波数に最も近いMIDIノートを見つける
        pitch_vector = np.zeros(88, dtype=np.int32)

        # MIDI音符の周波数（21-108）
        note_freqs = self.base_freq * (2 ** (np.arange(21, 109) / 12))

        for freq, magnitude in zip(freqs, magnitudes):
            if freq <= 0:
                continue

            # 各MIDIノートの周波数との差を計算
            freq_diffs = np.abs(note_freqs - freq)

            # セント単位の差に変換
            cents_diffs = 1200 * np.log2(note_freqs / freq)

            # 差が40セント以内の候補を検討
            candidates = np.where(np.abs(cents_diffs) < 40)[0]

            if len(candidates) > 0:
                # 振幅でスケールした二乗誤差を計算
                weighted_errors = magnitude * (cents_diffs[candidates] ** 2)

                # 最小誤差の候補を選択
                best_candidate = candidates[np.argmin(weighted_errors)]

                # ピッチベクトルを更新
                pitch_vector[best_candidate] = 1

        return np.asarray(pitch_vector, dtype=np.int32)

    # 以下のレガシー関数は実装は残しておきますが、現在は_advanced_*関数に置き換えられています

    def _extract_frequencies_from_ridges(
        self, ridges: List[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        リッジから周波数を抽出

        Parameters
        ----------
        ridges : List[Dict[str, np.ndarray]]
            リッジ情報のリスト

        Returns
        -------
        np.ndarray
            抽出された周波数
        """
        if not ridges:
            return np.array([])

        # リッジを時間でソート
        ridges = sorted(ridges, key=lambda x: x["time"])

        # シンプレクティック構造からの精緻化された周波数があれば使用
        frequencies = []
        for ridge in ridges:
            if "refined_frequency" in ridge:
                frequencies.append(ridge["refined_frequency"])
            else:
                frequencies.append(ridge["frequency"])

        return np.array(frequencies)

    def _solve_pitch_lattice(self, observed_freqs: np.ndarray) -> np.ndarray:
        """
        Diophantine Pitch Solver

        Parameters
        ----------
        observed_freqs : np.ndarray
            観測された周波数

        Returns
        -------
        np.ndarray
            ピッチベクトル (88要素の0/1ベクトル)
        """
        if len(observed_freqs) == 0:
            return np.zeros(88, dtype=np.int32)

        # 各観測周波数に最も近いMIDIノートを見つける
        pitch_vector = np.zeros(88, dtype=np.int32)

        for freq in observed_freqs:
            if freq <= 0:
                continue

            # 周波数からMIDIノート番号を計算
            midi_note = int(round(12 * np.log2(freq / self.base_freq)))

            # 範囲内のノートのみ考慮 (21-108)
            if 21 <= midi_note <= 108:
                pitch_vector[midi_note - 21] = 1

        return pitch_vector

    def _compute_morlet_cwt(self, signal: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        Morletウェーブレット変換を計算する

        Parameters
        ----------
        signal : np.ndarray
            入力信号（時間軸）
        scales : np.ndarray
            Morletウェーブレットのスケールのリスト

        Returns
        -------
        np.ndarray
            shape = (len(scales), len(signal)) の複素ウェーブレット係数
        """
        n = len(signal)
        # 1. 信号のFFT
        signal_fft = fft.fft(signal)
        freq = fft.fftfreq(n, d=1.0 / self.sample_rate)

        # 出力用
        coefs = np.zeros((len(scales), n), dtype=complex)

        # モデルパラメータ（中心周波数 xi0 = self.morlet_xi0）
        omega0 = self.morlet_xi0

        for i, scale in enumerate(scales):
            # Morletウェーブレットの周波数ドメイン表現
            # scale*freq - ω0 の2乗に基づいてガウス窓を生成
            wavelet = np.exp(-0.5 * ((scale * freq - omega0) ** 2))
            # 正規化（Morletの場合、π^{-1/4} * sqrt(scale) など）
            wavelet *= (np.pi ** (-0.25)) * np.sqrt(scale)

            # 畳み込み（周波数領域で乗算）
            convolved = signal_fft * wavelet

            # 逆FFTして時間領域に戻す
            coefs[i] = fft.ifft(convolved)

        return coefs

    def _compute_hermite_cwt(
        self, signal: np.ndarray, scales: np.ndarray
    ) -> np.ndarray:
        """
        Hermite関数を使った複素連続ウェーブレット変換

        Parameters
        ----------
        signal : np.ndarray
            入力信号（時間軸）
        scales : np.ndarray
            Hermiteウェーブレットのスケールのリスト

        Returns
        -------
        np.ndarray
            shape = (len(scales), len(signal)) の複素ウェーブレット係数
        """
        n = len(signal)
        # 1. 信号のFFT
        signal_fft = fft.fft(signal)
        freq = fft.fftfreq(n, d=1.0 / self.sample_rate)

        coefs = np.zeros((len(scales), n), dtype=complex)

        # 使用するHermite関数の次数
        hermite_order = self.hermite_n

        for i, scale in enumerate(scales):
            # ガウス包絡
            gaussian = np.exp(-0.5 * ((scale * freq) ** 2))

            # Hermite多項式による変調
            if hermite_order == 3:
                # H3(x) = 8x^3 - 12x
                hermite_poly = 8 * (scale * freq) ** 3 - 12 * (scale * freq)
            else:
                # 他の次数を扱う場合は適宜実装
                # 例: H1(x) = 2x, H5(x) = ...
                hermite_poly = 2 * (scale * freq)

            # 複素位相（i^order）や正規化係数など
            wavelet = (1j**hermite_order) * hermite_poly * gaussian
            norm_factor = np.sqrt(scale) * (np.pi ** (-0.25))
            wavelet *= norm_factor

            # 周波数領域で乗算 → 逆FFT
            convolved = signal_fft * wavelet
            coefs[i] = fft.ifft(convolved)

        return coefs

    def _sliding_dft(
        self,
        signal: np.ndarray,
        frame_size: int,
        hop_size: int,
        freq_bins: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        スライディングDFT（離散フーリエ変換）を計算する

        Parameters
        ----------
        signal : np.ndarray
            入力信号
        frame_size : int
            フレームサイズ
        hop_size : int
            ホップサイズ
        freq_bins : Optional[np.ndarray]
            分析する周波数ビンのインデックス（Noneの場合は全周波数を使用）

        Returns
        -------
        Dict[str, np.ndarray]
            DFT結果を含む辞書:
                - 'magnitude': 振幅スペクトログラム
                - 'phase': 位相スペクトログラム
                - 'frequencies': 周波数軸
                - 'times': 時間軸
        """
        # 信号の長さとフレーム数の計算
        signal_length = len(signal)
        n_frames = 1 + (signal_length - frame_size) // hop_size

        # 周波数軸の設定
        frequencies = fft.fftfreq(frame_size, d=1.0 / self.sample_rate)

        # 正の周波数のみを使用
        pos_freq_mask = frequencies >= 0
        n_freqs = np.sum(pos_freq_mask)
        frequencies = frequencies[pos_freq_mask]

        # 特定の周波数ビンのみ分析する場合
        if freq_bins is not None:
            mask = np.zeros_like(frequencies, dtype=bool)
            mask[freq_bins] = True
            frequencies = frequencies[mask]
            n_freqs = len(frequencies)

        # 結果を格納する配列
        magnitude = np.zeros((n_freqs, n_frames))
        phase = np.zeros((n_freqs, n_frames))

        # 各フレームでDFTを計算
        for i in range(n_frames):
            # フレーム範囲
            start = i * hop_size
            end = start + frame_size

            # フレームの切り出し（必要に応じてパディング）
            if end <= signal_length:
                frame = signal[start:end]
            else:
                frame = np.zeros(frame_size)
                frame[: signal_length - start] = signal[start:]

            # 窓関数適用（ハニング窓）
            window = np.hanning(frame_size)
            windowed_frame = frame * window

            # DFT計算
            spectrum = fft.fft(windowed_frame)

            # 正の周波数部分のみ取得
            spectrum = spectrum[pos_freq_mask]

            # 特定の周波数ビンのみ使用する場合
            if freq_bins is not None:
                spectrum = spectrum[freq_bins]

            # 振幅と位相を計算
            magnitude[:, i] = np.abs(spectrum)
            phase[:, i] = np.angle(spectrum)

        # 時間軸の計算
        times = np.arange(n_frames) * hop_size / self.sample_rate

        return {
            "magnitude": magnitude,
            "phase": phase,
            "frequencies": frequencies,
            "times": times,
        }
