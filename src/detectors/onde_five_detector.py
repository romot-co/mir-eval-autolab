"""
統合型動的アンチパターンピッチ検出アルゴリズム (ONDE-FIVE)

時間領域の周期検出と周波数領域の倍音構造解析を、多次元ガウス分布による
動的背景モデルに統合し、音声・音楽信号からのピッチ検出、オンセット検出、
オフセット検出を単一の理論的枠組みで実現します。

特徴:
- 単一メタパラメータ θ による感度・安定性の制御
- 時間-周波数の相互補完によるピッチ検出精度の向上
- リアルタイム処理に最適化された計算効率
- 統計的に堅牢な動的アンチパターン検出
"""

import numpy as np
import librosa
import logging
import time
import os
from typing import Dict, Any, Optional, Tuple, List, Union
from src.detectors.base_detector import BaseDetector
from src.detectors.modules.filterbank import FilterBank, CQTFilterBank, STFTFilterBank


class GaussianBackgroundModel:
    """単変量ガウス背景モデル（倍音成分ごとの背景追跡に使用）"""
    
    def __init__(self, alpha_base=0.95, initial_mean=0.1, initial_var=0.01):
        self.alpha_base = alpha_base
        self.alpha_event = 1.0 - (1.0 - alpha_base) / 10.0
        self.mean = initial_mean
        self.var = initial_var
        self.history = []
    
    def compute_score(self, value):
        """観測値の背景からの逸脱度（Z-score）を計算"""
        if self.var <= 1e-10:  # より安全な小さい値のチェック
            return 0.0
        
        z_score = abs(value - self.mean) / np.sqrt(self.var)
        return z_score
    
    def update(self, value, in_event=False):
        """背景モデルを更新"""
        alpha = self.alpha_event if in_event else self.alpha_base
        
        # 平均更新
        self.mean = alpha * self.mean + (1 - alpha) * value
        
        # 分散更新
        delta = value - self.mean
        self.var = alpha * self.var + (1 - alpha) * delta * delta
        
        # 下限を設定
        self.var = max(self.var, 1e-4)
    
    def set_alpha_base(self, new_alpha):
        """更新速度を変更"""
        self.alpha_base = new_alpha
        self.alpha_event = 1.0 - (1.0 - new_alpha) / 10.0


class TimeDomainPitchDetector:
    """
    時間領域ピッチ検出器
    
    自己相関分析を用いて時間領域でのピッチと周期性を検出します。
    """
    
    def __init__(self, min_f0=50.0, max_f0=1000.0, sr=44100):
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.sr = sr
        self.min_period = int(sr / max_f0)
        self.max_period = int(sr / min_f0)
    
    def analyze(self, audio_buffer):
        """
        オーディオバッファから基本周波数と周期性を検出
        
        Parameters
        ----------
        audio_buffer : np.ndarray
            オーディオバッファ
            
        Returns
        -------
        tuple
            (基本周波数f0, 周期確信度 [0-1])
        """
        # バッファがゼロの場合は早期リターン
        if np.all(np.abs(audio_buffer) < 1e-6):
            return 0.0, 0.0
        
        # 正規化
        audio = audio_buffer.copy()
        if np.std(audio) > 0:
            audio = (audio - np.mean(audio)) / np.std(audio)
        
        # 自己相関関数を計算
        corr = self._compute_autocorrelation(audio)
        
        # ピークを探す
        peak_period, confidence = self._find_peak(corr)
        
        # 周波数に変換
        if peak_period > 0:
            f0 = self.sr / peak_period
        else:
            f0 = 0.0
        
        return f0, confidence
    
    def _compute_autocorrelation(self, audio):
        """高速自己相関計算"""
        N = len(audio)
        # FFTを使用した高速自己相関の計算
        fft = np.fft.rfft(np.pad(audio, (0, N)))
        power_spec = np.abs(fft) ** 2
        corr = np.fft.irfft(power_spec)[:N]
        
        # 正規化
        if corr[0] > 0:
            corr = corr / corr[0]
        
        return corr
    
    def _find_peak(self, corr):
        """自己相関のピークを探す"""
        # 検索範囲を制限
        search_start = max(1, self.min_period)
        search_end = min(len(corr) - 1, self.max_period)
        
        if search_end <= search_start:
            return 0, 0.0
        
        # 探索範囲の自己相関
        search_corr = corr[search_start:search_end+1]
        
        # 最大ピークを探す
        peak_idx = np.argmax(search_corr)
        peak_val = search_corr[peak_idx]
        peak_period = search_start + peak_idx
        
        # ピークがない場合
        if peak_val <= 0 or peak_val < 0.3:
            return 0, 0.0
        
        # 周期確信度を計算: 近似的な周期性の指標
        # ピーク値が1に近いほど周期的
        confidence = peak_val
        
        return peak_period, confidence


class ONDEFIVEDetector(BaseDetector):
    """
    統合型動的アンチパターンピッチ検出アルゴリズム（ONDE-FIVE）
    
    時間領域と周波数領域を統合した多次元ガウス分布による背景モデルを用いて、
    ピッチ検出とイベント検出を単一の枠組みで実現します。
    
    Attributes:
        theta (float):              単一メタパラメータθ（感度・時間スケールを制御）
        sr (int):                   サンプリングレート
        hop_length (int):           ホップ長
        buffer_size (int):          処理バッファサイズ
        fft_size (int):             FFTサイズ
        feature_dim (int):          特徴ベクトル次元
        min_f0 (float):             最小周波数 (Hz)
        max_f0 (float):             最大周波数 (Hz)
        c_T (float):                閾値係数
        c_alpha (float):            背景モデル更新係数
        gamma_slow (float):         イベント中抑制係数
        
        # 内部変数
        mu (np.ndarray):            背景モデルの平均ベクトル
        sigma (np.ndarray):         背景モデルの共分散行列
        sigma_inv (np.ndarray):     共分散行列の逆行列
        
        in_event (bool):            イベント中かどうか
        event_frames (int):         現在のイベントの継続フレーム数
        prev_pitch (float):         前フレームのピッチ
        
        # 検出器
        time_domain_detector (TimeDomainPitchDetector): 時間領域ピッチ検出器
        filterbank (FilterBank):    時間-周波数変換器
    """
    
    def __init__(self, 
                 theta: float = 1.0,           # 単一メタパラメータ
                 c_T: float = 1.5,             # 閾値係数
                 c_alpha: float = 0.5,         # 更新速度係数
                 gamma_slow: float = 5.0,     # イベント中抑制係数
                 feature_dim: int = 3,         # 特徴ベクトル次元
                 min_f0: float = 50.0,         # 最小周波数 (Hz)
                 max_f0: float = 4000.0,       # 最大周波数 (Hz)
                 sr: int = 44100,              # サンプリングレート
                 buffer_size: int = 1024,      # 処理バッファサイズ
                 fft_size: int = 2048,         # FFTサイズ
                 hop_length: int = 512,        # ホップ長
                 filterbank_type: str = "cqt", # フィルターバンクタイプ
                 filterbank_params: Optional[Dict[str, Any]] = None,
                 max_harmonics: int = 10,      # 最大倍音数
                 adaptive_harmonics: bool = True, # 適応的倍音モデリング
                 max_harmonic_models: int = 1000, # 最大ハーモニクスモデル数
                 polyphonic: bool = False,     # ポリフォニック検出を有効にするかどうか
                 max_polyphony: int = 3,       # 同時に検出する最大音数
                 enable_debug: bool = False,    # デバッグモードを有効にするかどうか
                 harmonic_decay_threshold: float = 0.1,  # 倍音減衰閾値 (何割の倍音が背景レベルに近づいたらオフセット扱いか)
                 min_harmonics_for_offset: int = 3,      # オフセット判定に必要な最小倍音数
                 harmonic_decay_frames: int = 1,         # 倍音減衰の連続フレーム数要件
                 harmonic_decay_alpha_factor: float = 1.2, # 減衰中の背景モデル更新速度係数,
                 **kwargs):
        """
        ONDE-FIVE検出器の初期化
        
        Parameters
        ----------
        theta : float
            単一メタパラメータ (デフォルト=1.0)
        c_T : float
            閾値係数 (デフォルト=1.5)
        c_alpha : float
            更新速度係数 (デフォルト=1.0)
        gamma_slow : float
            イベント中抑制係数 (デフォルト=10.0)
        feature_dim : int
            特徴ベクトル次元 (デフォルト=3)
        min_f0 : float
            最小周波数 (Hz) (デフォルト=50.0)
        max_f0 : float
            最大周波数 (Hz) (デフォルト=1000.0)
        sr : int
            サンプリングレート (デフォルト=44100)
        buffer_size : int
            処理バッファサイズ (デフォルト=1024)
        fft_size : int
            FFTサイズ (デフォルト=2048)
        hop_length : int
            ホップ長 (デフォルト=512)
        filterbank_type : str
            フィルターバンクタイプ (デフォルト="cqt")
        filterbank_params : Dict[str, Any], optional
            フィルターバンクのパラメータ辞書
        max_harmonics : int
            最大倍音数 (デフォルト=10)
        adaptive_harmonics : bool
            適応的倍音モデリングを使用するか (デフォルト=True)
        max_harmonic_models : int
            保持する最大ハーモニクスモデル数 (デフォルト=1000)
        polyphonic : bool
            ポリフォニック検出を有効にするかどうか (デフォルト=False)
        max_polyphony : int
            同時に検出する最大音数 (デフォルト=3)
        enable_debug : bool
            デバッグモードを有効にするかどうか (デフォルト=False)
        harmonic_decay_threshold : float
            倍音減衰閾値 (何割の倍音が背景レベルに近づいたらオフセット扱いか)
        min_harmonics_for_offset : int
            オフセット判定に必要な最小倍音数
        harmonic_decay_frames : int
            連続フレーム数要件
        harmonic_decay_alpha_factor : float
            減衰中の背景モデル更新速度係数
        """
        # BaseDetectorの初期化
        super().__init__()
        
        # デバッグモードの設定
        self.enable_debug = enable_debug
        if self.enable_debug:
            # デバッグモードが有効な場合、ロギングレベルをDEBUGに設定
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            logging.debug("ONDEFIVEDetector: デバッグモードが有効です")
        
        # 基本パラメータ
        self.theta = theta
        self.c_T = c_T
        self.c_alpha = c_alpha
        self.gamma_slow = gamma_slow
        self.feature_dim = feature_dim
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.sr = sr
        self.buffer_size = buffer_size
        self.fft_size = fft_size
        self.hop_length = hop_length
        
        # ポリフォニック検出パラメータ
        self.polyphonic = polyphonic
        self.max_polyphony = max_polyphony
        
        # 倍音モデリングパラメータ
        self.max_harmonics = max_harmonics
        self.adaptive_harmonics = adaptive_harmonics
        self.used_harmonics = 5  # 初期値
        self.max_harmonic_models = max_harmonic_models
        
        # 閾値と更新速度の計算
        self.threshold = c_T * theta
        self.alpha_base = np.exp(-c_alpha / theta)
        self.alpha_event = 1.0 - (1.0 - self.alpha_base) / gamma_slow
        
        # フィルターバンクタイプとパラメータ
        self.filterbank_type = filterbank_type
        self.filterbank_params = filterbank_params or {}
        
        # 背景ガウスモデル初期化
        self.mu = np.zeros(feature_dim)
        self.sigma = np.eye(feature_dim) * 0.1  # 初期共分散行列
        try:
            self.sigma_inv = np.linalg.inv(self.sigma)
        except np.linalg.LinAlgError:
            # 特異行列の場合は正則化
            self.sigma += np.eye(feature_dim) * 1e-3
            self.sigma_inv = np.linalg.inv(self.sigma)
        
        # 時間領域ピッチ検出器初期化
        self.time_domain_detector = TimeDomainPitchDetector(
            min_f0=min_f0, 
            max_f0=max_f0,
            sr=sr
        )
        
        # 状態追跡
        self.in_event = False
        self.event_frames = 0
        self.prev_z_score = 0
        self.prev_pitch = 0
        self.prev_feature = np.zeros(feature_dim)
        
        # 複数のピッチを追跡するための状態変数
        self.multi_pitch_history = []  # 複数ピッチの履歴
        
        # ハーモニクス背景モデル
        self.harmonic_models = {}
        self.z_scores_history = []
        
        # バッファとフレーム
        self._audio_buffer = np.zeros(buffer_size)
        self._prev_frames = []  # 過去フレーム保存用
        self._prev_spec_energy = np.array([0.1])  # 初期エネルギー値
        
        # 現在のスペクトルフレームを保存する変数
        self._current_spec_frame = None
        self._current_freq_bins = None
        
        # フィルターバンクの初期化
        self.filterbank = self.initialize_filterbank()
        
        # 小さい値を防ぐための定数
        self.eps = 1e-8
        
        # 最後のハーモニクスモデル整理時間
        self.last_prune_time = time.time()
        
        # 周波数領域背景モデル - 新規追加
        self.freq_bg_models = {}  # {bin_idx: (mean, var)}の辞書
        
        # イベント検出のためのロジスティック変換パラメータ
        # 設定ファイルから指定があれば、それを使用
        self.logistic_a = kwargs.get('logistic_a', 2.0)  # ロジスティック変換の傾き
        self.logistic_b = kwargs.get('logistic_b', 1.0)  # ロジスティック変換の中心
        
        # ステートマシンのヒステリシスパラメータ
        # 設定ファイルから指定があれば、それを使用
        # デフォルトはthetaに基づく自動計算
        if 'onset_prob' in kwargs and 'offset_prob' in kwargs:
            self.onset_prob = kwargs.get('onset_prob')
            self.offset_prob = kwargs.get('offset_prob')
            self._onset_prob_fixed = True
            self._offset_prob_fixed = True
        else:
            # set_thetaと同じロジックに合わせる
            self.onset_prob = 0.5 + 0.1 * (1.0 / theta)
            self.offset_prob = 0.2 - 0.1 * (1.0 / theta)
        
        # スペクトル逸脱スコアの履歴
        self.deviation_scores_history = []
        
        # オフセット検出の改善のための変数
        if 'offset_wait_frames' in kwargs:
            self.offset_wait_frames = int(kwargs['offset_wait_frames'])
            self._offset_wait_frames_fixed = True
        else:
            self.offset_wait_frames = max(5, int(2.0 + 1.0 * theta))  # set_thetaと同じ計算式
        self.low_event_count = 0  # p_event < offset_prob が続いたカウンタ
        self.recent_frame_data = []  # 直近フレームのデータを保存（オフセット微調整用）
        self.max_recent_frames = 10  # 保存する最大フレーム数
        
        # ノート後処理のパラメータ
        self.min_note_duration = 0.05  # ノートの最小長さ（秒）
        self.min_gap_duration = max(0.08, 0.03 * theta)   # ノート間の最小ギャップ（秒）
        self.max_pitch_ratio = 1.06  # 3%以内の変動は同一ピッチと見なす
        
        # ハーモニクス減衰検出のパラメータ
        self.harmonic_decay_threshold = harmonic_decay_threshold  # 倍音減衰閾値 (0-1)
        self.min_harmonics_for_offset = min_harmonics_for_offset  # オフセット判定に必要な最小倍音数
        if 'harmonic_decay_frames' in kwargs:
            self.harmonic_decay_frames = int(kwargs['harmonic_decay_frames'])
            self._harmonic_decay_frames_fixed = True
        else:
            self.harmonic_decay_frames = max(2, int(1.0 + 0.2 * theta))  # set_thetaと同じ計算式
        self.harmonic_decay_alpha_factor = harmonic_decay_alpha_factor  # 減衰中の背景モデル更新速度係数
        
        # ハーモニクス減衰の状態変数
        self._harmonic_decay_metric = 0.0  # 現在の倍音減衰度 (0-1)
        self._harmonic_decay_count = 0     # 連続減衰フレーム数
        self._active_harmonics = []        # 現在アクティブな倍音のリスト (h_key)
        self._decaying_harmonics = []      # 減衰中の倍音のリスト (h_key)
        
        if self.enable_debug:
            logging.debug(f"ONDEFIVEDetector: 初期化完了 theta={theta}, feature_dim={feature_dim}")

    def initialize_filterbank(self) -> FilterBank:
        """
        選択されたフィルターバンクを初期化
        
        Returns
        -------
        FilterBank
            初期化されたフィルターバンクのインスタンス
        """
        if self.filterbank_type == "cqt":
            # CQTフィルターバンク
            cqt_params = {
                'n_bins': 84,  # 7オクターブ相当
                'bins_per_octave': 12,
                'f_min': self.min_f0,
                'f_max': self.max_f0,
                'hop_length': self.hop_length,
                **self.filterbank_params
            }
            
            if self.enable_debug:
                logging.debug(f"CQTフィルターバンクを初期化します: {cqt_params}")
                
            return CQTFilterBank(**cqt_params)
            
        elif self.filterbank_type == "stft":
            # STFTフィルターバンク
            stft_params = {
                'n_fft': self.fft_size,
                'hop_length': self.hop_length,
                'f_min': self.min_f0,
                'f_max': self.max_f0,
                **self.filterbank_params
            }
            
            if self.enable_debug:
                logging.debug(f"STFTフィルターバンクを初期化します: {stft_params}")
                
            return STFTFilterBank(**stft_params)
        else:
            # 未知のフィルターバンクタイプ
            raise ValueError(f"未知のフィルターバンクタイプ: {self.filterbank_type}")
    
    def process_frame(self, audio_buffer, spec_frame=None, freq_bins=None):
        """
        単一フレームを処理し、ピッチとイベント情報を抽出
        
        Parameters
        ----------
        audio_buffer : np.ndarray
            オーディオバッファ (時間領域サンプル)
        spec_frame : np.ndarray, optional
            スペクトログラムフレーム (周波数領域)
        freq_bins : np.ndarray, optional
            周波数ビン
            
        Returns
        -------
        dict
            ピッチ、イベント確率、イベント情報を含む辞書
        """
        # バッファを更新
        self._audio_buffer = audio_buffer
        
        # スペクトログラムが提供されていない場合は計算
        if spec_frame is None or freq_bins is None:
            spec_frame, freq_bins = self._compute_spectrogram(audio_buffer)
        
        # 1. 周波数領域ベースの逸脱量計算 (新しいアプローチ)
        spectrum_deviation = self._compute_spectrum_deviation(spec_frame, freq_bins)
        
        # 2. ロジスティック変換でイベント確率に変換
        p_event = self._logistic_transform(spectrum_deviation)
        
        # 3. ステートマシンによるイベント検出
        is_onset, is_offset = self._detect_events(p_event)
        
        # 4. 特徴ベクトル抽出 (従来機能との互換性維持)
        r = self._extract_feature_vector(audio_buffer, spec_frame, freq_bins)
        
        # 5. マハラノビス距離に基づくZ-score計算 (従来機能との互換性維持)
        delta = r - self.mu
        z_score = np.sqrt(delta.T @ self.sigma_inv @ delta)
        
        # Z-score履歴の更新
        self.z_scores_history.append(z_score)
        if len(self.z_scores_history) > 200:  # 最大200個保持
            self.z_scores_history = self.z_scores_history[-200:]
        
        # 6. 背景モデル更新
        alpha = self._calculate_alpha(p_event)  # イベント確率に基づく更新係数
        self._update_background_model(r, alpha)
        
        # 7. ピッチ決定（単音または多音）
        if self.polyphonic:
            # ポリフォニック検出
            pitches = self._determine_multiple_pitches(spec_frame, freq_bins, z_score)
            final_pitch = pitches[0] if pitches else 0.0  # 主要なピッチ
        else:
            # モノフォニック検出
            final_pitch = self._determine_final_pitch(r, z_score)
            pitches = [final_pitch] if final_pitch > 0 else []
        
        # 状態更新
        self.prev_z_score = z_score
        self.prev_feature = r.copy()
        
        # 最近のフレームデータを保存（オフセット微調整用）
        frame_data = {
            'pitch': final_pitch,
            'p_event': p_event,
            'energy': np.sum(spec_frame),
            'spec_frame': spec_frame.copy() if self.enable_debug else None,  # デバッグモード時のみスペクトルも保存
            'z_score': z_score,
            'is_onset': is_onset,
            'is_offset': is_offset
        }
        self.recent_frame_data.append(frame_data)
        if len(self.recent_frame_data) > self.max_recent_frames:
            self.recent_frame_data.pop(0)  # 古いデータを削除
        
        # 結果を返す
        result = {
            'pitch': final_pitch,
            'z_score': z_score,
            'p_event': p_event,  # 新しいイベント確率
            'spectrum_deviation': spectrum_deviation,  # 新しいスペクトル逸脱スコア
            'is_onset': is_onset,
            'is_offset': is_offset,
            'feature_vector': r,
            'in_event': self.in_event,
            'used_harmonics': self.used_harmonics
        }
        
        # ポリフォニックモードの場合、全ピッチを追加
        if self.polyphonic:
            result['pitches'] = pitches
        
        return result
    
    def _extract_feature_vector(self, audio_buffer, spec_frame, freq_bins):
        """
        特徴ベクトルの抽出
        
        Parameters
        ----------
        audio_buffer : np.ndarray
            オーディオバッファ
        spec_frame : np.ndarray
            スペクトログラムフレーム
        freq_bins : np.ndarray
            周波数ビン
            
        Returns
        -------
        np.ndarray
            特徴ベクトル
        """
        # 1. 時間領域特徴抽出
        f_time, p_time = self.time_domain_detector.analyze(audio_buffer)
        
        # 2. 周波数領域特徴抽出
        harmonic_score = self._calculate_harmonic_score(spec_frame, freq_bins, f_time)
        
        # 3. 特徴ベクトル構築 (3次元または4次元)
        if self.feature_dim == 3:
            r = np.array([
                np.log1p(max(f_time, 1e-3)),  # 時間領域ピッチ (対数スケール)
                p_time,                        # 周期確信度
                harmonic_score                 # 倍音整合度
            ])
        elif self.feature_dim == 4:
            # フレームエネルギー比率を追加
            energy_ratio = np.sum(spec_frame) / (np.mean(self._prev_spec_energy) + 1e-6)
            energy_ratio = np.clip(energy_ratio, 0.01, 100.0)  # 範囲制限
            
            r = np.array([
                np.log1p(max(f_time, 1e-3)),  # 時間領域ピッチ (対数スケール)
                p_time,                        # 周期確信度
                harmonic_score,               # 倍音整合度
                np.log(energy_ratio)          # エネルギー比率 (対数スケール)
            ])
        else:
            raise ValueError(f"Feature dimension {self.feature_dim} not supported")
        
        # 異常値を制限
        r = np.clip(r, -10, 10)
        
        return r
    
    def _compute_spectrogram(self, audio_buffer):
        """
        オーディオバッファからスペクトログラムを計算
        
        Parameters
        ----------
        audio_buffer : np.ndarray
            オーディオバッファ
            
        Returns
        -------
        tuple
            (magnitude_spec, freq_bins)
        """
        # STFTを計算
        X = np.fft.rfft(audio_buffer * np.hanning(len(audio_buffer)), n=self.fft_size)
        magnitude_spec = np.abs(X)
        
        # 振幅スペクトラム
        magnitude_spec = magnitude_spec[:self.fft_size//2 + 1]
        
        # 周波数ビンを計算
        freq_bins = np.fft.rfftfreq(self.fft_size, 1.0/self.sr)
        
        # 状態保存
        self._current_spec_frame = magnitude_spec
        self._current_freq_bins = freq_bins
        
        # 前フレームエネルギー履歴の更新
        if not hasattr(self, '_prev_spec_energy'):
            self._prev_spec_energy = np.array([np.sum(magnitude_spec)])
        else:
            self._prev_spec_energy = np.append(self._prev_spec_energy, np.sum(magnitude_spec))
            if len(self._prev_spec_energy) > 10:
                self._prev_spec_energy = self._prev_spec_energy[-10:]
        
        return magnitude_spec, freq_bins 

    def _calculate_harmonic_score(self, spec_frame, freq_bins, f_time):
        """
        倍音構造の整合度を計算
        
        Parameters
        ----------
        spec_frame : np.ndarray
            スペクトログラムフレーム
        freq_bins : np.ndarray
            周波数ビン
        f_time : float
            時間領域で推定されたピッチ (Hz)
            
        Returns
        -------
        float
            倍音整合度スコア
        """
        
        if f_time < self.min_f0 or f_time > self.max_f0:
            return 0.0
        
        # 適応的倍音数の決定
        if self.adaptive_harmonics:
            # スペクトル特性に基づいて使用する倍音数を動的に決定
            if np.sum(spec_frame) > self.eps:
                spectral_centroid = np.sum(spec_frame * freq_bins) / np.sum(spec_frame)
                self.used_harmonics = min(self.max_harmonics, 
                                      max(3, int(spectral_centroid / f_time / 2)))
            else:
                self.used_harmonics = 5
        else:
            self.used_harmonics = 5
        
        total_energy = 0
        total_deviation = 0
        
        # 倍音の情報を保存するリストをリセット
        self._active_harmonics = []
        self._decaying_harmonics = []
        harmonic_energies = []
        harmonic_deviations = []
        harmonic_bg_means = []
        harmonic_bg_stds = []
        
        for h in range(1, self.used_harmonics + 1):
            h_freq = f_time * h
            
            # 周波数範囲内かチェック
            if h_freq > freq_bins[-1]:
                continue
            
            # 最も近い周波数ビンを探す
            bin_idx = np.argmin(np.abs(freq_bins - h_freq))
            
            if bin_idx < len(spec_frame):
                # 倍音エネルギーを取得
                h_energy = spec_frame[bin_idx]
                
                # この倍音に対する背景モデルを取得または作成
                h_key = f"{f_time:.1f}_{h}"
                if h_key not in self.harmonic_models:
                    self.harmonic_models[h_key] = GaussianBackgroundModel(
                        alpha_base=self.alpha_base,
                        initial_mean=0.1,
                        initial_var=0.01
                    )
                
                # 背景からの逸脱を計算
                deviation = self.harmonic_models[h_key].compute_score(h_energy)
                
                # 倍音情報を保存
                harmonic_energies.append(h_energy)
                harmonic_deviations.append(deviation)
                harmonic_bg_means.append(self.harmonic_models[h_key].mean)
                harmonic_bg_stds.append(np.sqrt(self.harmonic_models[h_key].var))
                self._active_harmonics.append(h_key)
                
                # 減衰中の倍音を特定
                # 倍音エネルギーが背景平均に近いかどうかを判断
                if abs(h_energy - self.harmonic_models[h_key].mean) <= 1.0 * np.sqrt(self.harmonic_models[h_key].var):
                    # 減衰中なので、アクティブリストから除外し、減衰リストに追加
                    if h_key in self._active_harmonics:
                        self._active_harmonics.remove(h_key)
                    self._decaying_harmonics.append(h_key)
                else:
                    # 減衰していないのでアクティブリストに追加
                    self._active_harmonics.append(h_key)
                
                # エネルギーと逸脱スコアを倍音の次数に応じて重み付け
                weight = 1.0 / h  # 高次倍音ほど重みを下げる
                
                total_energy += h_energy * weight
                total_deviation += max(0, deviation) * weight
                
                # 背景モデルを更新（減衰中の倍音は更新速度を調整）
                in_event = deviation > 2.0  # イベント判定
                
                # 減衰中の倍音はより速く背景に近づくように更新速度を調整
                if h_key in self._decaying_harmonics and in_event:
                    # 減衰中はより速く背景に近づける（alpha_baseに近づける）
                    alpha_adjusted = self.alpha_base / self.harmonic_decay_alpha_factor
                    # あまりにも小さくしすぎないように制限
                    alpha_adjusted = max(alpha_adjusted, 0.5)
                    self.harmonic_models[h_key].update(h_energy, in_event=False)
                else:
                    # 通常の更新
                    self.harmonic_models[h_key].update(h_energy, in_event=in_event)
                
                # アクセス時間を更新
                if not hasattr(self.harmonic_models[h_key], 'last_access_time'):
                    self.harmonic_models[h_key].last_access_time = time.time()
                else:
                    self.harmonic_models[h_key].last_access_time = time.time()
        
        # 倍音減衰度を計算
        self._calculate_harmonic_decay_metric(harmonic_energies, harmonic_bg_means, harmonic_bg_stds)
        
        # 倍音構造のスコアを計算
        harmonic_score = total_energy * np.sqrt(1 + total_deviation)  # エネルギーと逸脱の関数
        
        # スコアの正規化
        harmonic_score = np.clip(harmonic_score / (self.used_harmonics * 0.5), 0, 10)
        
        # 定期的にハーモニクスモデルを整理
        current_time = time.time()
        if current_time - self.last_prune_time > 30.0:  # 30秒ごとに整理
            self._prune_harmonic_models()
            self.last_prune_time = current_time
        
        return harmonic_score
    
    def _prune_harmonic_models(self):
        """
        使用されていないハーモニクスモデルを整理してメモリを節約
        """
        current_time = time.time()
        expired_keys = []
        
        # アクセスされていない期間が長いモデルを特定
        for key, model in self.harmonic_models.items():
            if not hasattr(model, 'last_access_time'):
                model.last_access_time = current_time
                continue
                
            # 60秒以上アクセスされていないモデルを期限切れとする
            if current_time - model.last_access_time > 60.0:
                expired_keys.append(key)
        
        # 期限切れのモデルを削除
        for key in expired_keys:
            del self.harmonic_models[key]
        
        # モデル数が上限を超えた場合は最も古いものから削除
        if len(self.harmonic_models) > self.max_harmonic_models:
            # アクセス時間でソート
            sorted_models = sorted(
                self.harmonic_models.items(),
                key=lambda x: getattr(x[1], 'last_access_time', 0)
            )
            
            # 削除する数を計算
            excess = len(self.harmonic_models) - self.max_harmonic_models
            # 最も古いモデルを削除
            for i in range(excess):
                if i < len(sorted_models):
                    del self.harmonic_models[sorted_models[i][0]]
        
        if self.enable_debug:
            logging.debug(f"ハーモニクスモデル整理: {len(expired_keys)}個削除, 残り{len(self.harmonic_models)}個")
    
    def _calculate_harmonic_decay_metric(self, energies, bg_means, bg_stds):
        """
        倍音の減衰度合いを計算
        
        Parameters
        ----------
        energies : list
            倍音エネルギーのリスト
        bg_means : list
            各倍音の背景モデル平均値
        bg_stds : list
            各倍音の背景モデル標準偏差
            
        Returns
        -------
        float
            倍音減衰度 (0-1)
        """
        if not energies or len(energies) < self.min_harmonics_for_offset:
            # 十分な倍音がない場合は減衰していないとみなす
            self._harmonic_decay_metric = 0.0
            self._harmonic_decay_count = 0
            return self._harmonic_decay_metric
        
        # 各倍音について、背景レベルにどれだけ近いかを計算
        decay_scores = []
        for energy, mean, std in zip(energies, bg_means, bg_stds):
            # 背景からの相対的な距離（標準偏差単位）
            rel_distance = abs(energy - mean) / (std + self.eps)
            # 1.0 以下なら背景レベルに近いと判断
            decay_score = 1.0 if rel_distance <= 1.0 else max(0.0, 2.0 - rel_distance) / 1.0
            decay_scores.append(decay_score)
        
        # 倍音減衰度 = 背景レベルに近い倍音の割合
        decay_metric = np.mean(decay_scores)
        
        # 連続フレームカウントの更新
        if decay_metric >= self.harmonic_decay_threshold:
            self._harmonic_decay_count += 1
        else:
            self._harmonic_decay_count = 0
        
        # 減衰メトリックを保存
        self._harmonic_decay_metric = decay_metric
        
        if self.enable_debug and decay_metric >= self.harmonic_decay_threshold:
            logging.debug(f"倍音減衰検出: {decay_metric:.2f} (閾値: {self.harmonic_decay_threshold:.2f}), "
                         f"連続フレーム: {self._harmonic_decay_count}/{self.harmonic_decay_frames}")
        
        return self._harmonic_decay_metric
    
    def _check_pitch_decay_event(self, pitch_freq):
        """
        ピッチ周辺エネルギーの減衰を検出（オフセット検出補助）
        
        Parameters
        ----------
        pitch_freq : float
            検出されたピッチ周波数
            
        Returns
        -------
        tuple
            (decay_detected, decay_score)
            decay_detected: ピッチエネルギーが十分に減衰していればTrue
            decay_score: 減衰度合い (0-1)
        """
        # 基本的なバリデーション
        if (pitch_freq < self.min_f0 or 
            self._current_freq_bins is None or 
            self._current_spec_frame is None or
            len(self._current_freq_bins) == 0 or
            len(self._current_spec_frame) == 0):
            return False, 0.0
            
        # ピッチ周辺のビンを取得（基本周波数とその倍音をチェック）
        main_bin_idx = np.argmin(np.abs(self._current_freq_bins - pitch_freq))
        
        # インデックス範囲の追加チェック
        if main_bin_idx >= len(self._current_spec_frame):
            return False, 0.0
            
        energy = self._current_spec_frame[main_bin_idx]
        
        # 最初の倍音も確認（より安定した検出のため）
        harmonic_energy = 0
        h2_freq = pitch_freq * 2
        if h2_freq <= self._current_freq_bins[-1]:
            h2_bin_idx = np.argmin(np.abs(self._current_freq_bins - h2_freq))
            if h2_bin_idx < len(self._current_spec_frame):
                harmonic_energy = self._current_spec_frame[h2_bin_idx]
        
        # 総エネルギー（基音 + 倍音）
        total_energy = energy + harmonic_energy * 0.5  # 倍音は重みを半分に
        
        # 周波数ビンの背景モデルを取得
        main_bin_key = f"bin_{main_bin_idx}"
        if main_bin_key not in self.freq_bg_models:
            return False, 0.0
            
        bg_mean = self.freq_bg_models[main_bin_key]['mean']
        bg_std = np.sqrt(self.freq_bg_models[main_bin_key]['var'])
        
        # 背景レベルへの接近度を計算
        # よりエネルギーが背景に近づいたかを判定（閾値を調整）
        decay_threshold = bg_mean + 0.7 * bg_std
        
        # ピッチ周辺の複数のビンをチェック（より安定した検出のため）
        window_size = 2  # ピッチの前後2ビンまで確認
        window_start = max(0, main_bin_idx - window_size)
        window_end = min(len(self._current_spec_frame) - 1, main_bin_idx + window_size)
        
        # 周辺ビンの平均エネルギーも計算
        surrounding_decay = False
        surrounding_energy_decay_score = 0.0
        
        if window_end > window_start:
            surrounding_energy = np.mean(self._current_spec_frame[window_start:window_end+1])
            
            # 周辺エネルギーも閾値以下なら減衰と判断
            surrounding_decay = surrounding_energy <= (bg_mean + 1.0 * bg_std)
            
            # 周辺エネルギーの減衰スコア（0-1）
            if surrounding_energy <= bg_mean:
                surrounding_energy_decay_score = 1.0
            else:
                rel_level = (surrounding_energy - bg_mean) / (bg_std + self.eps)
                surrounding_energy_decay_score = max(0.0, 1.0 - rel_level / 2.0)
        
        # スペクトル特性による減衰判定
        spectral_decay = total_energy <= decay_threshold
        
        # メインビンの減衰スコア（0-1）
        if total_energy <= bg_mean:
            main_decay_score = 1.0
        else:
            rel_level = (total_energy - bg_mean) / (bg_std + self.eps)
            main_decay_score = max(0.0, 1.0 - rel_level / 2.0)
        
        # 倍音構造による減衰判定（ハーモニクス減衰度）
        harmonic_decay = self._harmonic_decay_count >= self.harmonic_decay_frames
        
        # 総合的な減衰判定
        # 1. スペクトル特性 OR 周辺ビン減衰
        # 2. かつ倍音減衰度が閾値以上
        basic_decay = spectral_decay or surrounding_decay
        
        # 総合的な減衰スコア（0-1）- 各要素の重み付け平均
        decay_score = 0.3 * main_decay_score + 0.3 * surrounding_energy_decay_score + 0.4 * self._harmonic_decay_metric
        
        # 最終的な減衰判定
        # 1. 基本的な減衰条件が満たされている
        # 2. かつ倍音減衰度が十分高い
        decay_detected = basic_decay and (harmonic_decay or self._harmonic_decay_metric >= self.harmonic_decay_threshold)
        
        if self.enable_debug and decay_detected:
            logging.debug(f"ピッチ減衰検出: スコア={decay_score:.2f}, "
                        f"基本減衰={basic_decay}, 倍音減衰={harmonic_decay}, "
                        f"倍音減衰度={self._harmonic_decay_metric:.2f}")
        
        return decay_detected, decay_score

    def _detect_events(self, p_event):
        """
        イベント確率に基づいてオンセットとオフセットを検出するステートマシン
        
        Parameters
        ----------
        p_event : float
            イベント確率 [0-1]
            
        Returns
        -------
        tuple
            (is_onset, is_offset)
        """
        
        is_onset = False
        is_offset = False
        
        # ステートマシンによるイベント検出（ヒステリシス付き）
        if not self.in_event:
            # 非イベント状態 → イベント状態への遷移
            if p_event > self.onset_prob:
                self.in_event = True
                self.event_frames = 1
                self.low_event_count = 0  # オフセットカウンタをリセット
                self._harmonic_decay_count = 0  # 倍音減衰カウンタもリセット
                is_onset = True
                
                if self.enable_debug:
                    logging.debug(f"オンセット検出: p_event={p_event:.2f}, onset_prob={self.onset_prob:.2f}")
        else:
            # イベント状態での処理
            self.event_frames += 1
            
            # イベント状態 → 非イベント状態への遷移
            # p_eventが低いかつピッチエネルギーが減衰している場合にカウント (AND条件)
            current_pitch = 0.0
            if len(self.recent_frame_data) > 0:
                current_pitch = self.recent_frame_data[-1]['pitch']
                
            # 拡張されたピッチ減衰検出を使用
            pitch_decayed, decay_score = self._check_pitch_decay_event(current_pitch)
            
            # 複合条件でオフセット遷移を評価
            # 1. イベント確率が低い
            # 2. かつピッチ減衰が検出された
            # 3. かつハーモニクス減衰度が閾値以上
            harmonic_decay_condition = self._harmonic_decay_count >= self.harmonic_decay_frames
            
            # より厳しいAND条件（すべての条件を満たす必要がある）
            # 条件を意味のある変数に分解
            low_event_probability = p_event < self.offset_prob
            sufficient_harmonic_decay = (harmonic_decay_condition or 
                                        self._harmonic_decay_metric >= self.harmonic_decay_threshold)
            offset_condition_met = low_event_probability and pitch_decayed and sufficient_harmonic_decay
            
            if offset_condition_met:
                self.low_event_count += 1  # 条件を満たすフレームをカウント
                
                if self.enable_debug and self.low_event_count == 1:
                    logging.debug(f"オフセット条件検出開始: p_event={p_event:.2f}, "
                                f"減衰スコア={decay_score:.2f}, "
                                f"倍音減衰度={self._harmonic_decay_metric:.2f}/{self.harmonic_decay_threshold:.2f}, "
                                f"倍音減衰フレーム={self._harmonic_decay_count}/{self.harmonic_decay_frames}")
            else:
                self.low_event_count = 0  # カウンタをリセット
            
            # 一定フレーム連続して条件を満たした場合にオフセット確定
            if self.low_event_count >= self.offset_wait_frames:
                # 最小イベント長のチェック（最低5フレーム確保）
                min_event_frames = max(5, int(self.theta * 2.5))
                
                if self.event_frames >= min_event_frames:
                    self.in_event = False
                    self.event_frames = 0
                    self.low_event_count = 0
                    self._harmonic_decay_count = 0  # 倍音減衰カウンタもリセット
                    is_offset = True
                    
                    # 倍音減衰情報をオフセット時間の最適化のために保存
                    self._offset_decay_info = {
                        'decay_score': decay_score,
                        'harmonic_decay_metric': self._harmonic_decay_metric,
                        'decaying_harmonics_count': len(self._decaying_harmonics)
                    }
                    
                    if self.enable_debug:
                        logging.debug(f"オフセット検出確定: p_event={p_event:.2f}, "
                                    f"連続低確率フレーム={self.offset_wait_frames}, "
                                    f"倍音減衰度={self._harmonic_decay_metric:.2f}")
            
            # 長すぎるイベントも終了させる（最大長のチェック）
            max_event_frames = int(self.theta * 30)
            if self.event_frames > max_event_frames and p_event < self.onset_prob:
                self.in_event = False
                self.event_frames = 0
                self.low_event_count = 0
                self._harmonic_decay_count = 0  # 倍音減衰カウンタもリセット
                is_offset = True
                
                if self.enable_debug:
                    logging.debug(f"長期イベント終了: p_event={p_event:.2f}, event_frames={self.event_frames} > max={max_event_frames}")
        
        return is_onset, is_offset
    
    def _calculate_alpha(self, p_event: float) -> float:
        """
        更新係数の計算 (イベント確率に基づいて更新速度を調整)
        
        Parameters
        ----------
        p_event : float
            イベント確率 [0-1]
            
        Returns
        -------
        float
            更新係数
        """
        # イベント確率が高いまたはイベント中の場合は更新を抑制
        if p_event > self.onset_prob or self.in_event:
            return self.alpha_event
        return self.alpha_base

    def set_theta(self, new_theta):
        """
        メタパラメータθを変更し、関連する全パラメータを自動調整
        
        Parameters
        ----------
        new_theta : float
            新しいθ値
        """
        self.theta = new_theta
        
        # 背景更新速度 (既存)
        self.alpha_base = np.exp(-self.c_alpha / new_theta)
        self.alpha_event = 1.0 - (1.0 - self.alpha_base) / self.gamma_slow
        
        # ロジスティックの中心/スケール
        # 中心は過去のスペクトル逸脱スコアの中央値から設定
        if len(self.deviation_scores_history) > 20:  # 50→20に変更して早期適応
            self.logistic_b = np.median(self.deviation_scores_history)
        else:
            # 履歴が少ない場合はθに基づいて調整
            self.logistic_b = 2.0 * (0.8 + 0.2 * new_theta)  # 5.0→2.0に変更
            
        # 傾きはθの逆数に比例（θが小さいほど傾きを大きく）
        self.logistic_a = 2.0 / (0.5 + 0.2 * new_theta)
        
        # ステートマシンのヒステリシス閾値調整（外部指定がない場合のみ）
        if not hasattr(self, '_onset_prob_fixed'):
            self.onset_prob = 0.5 + 0.1 * (1.0 / new_theta)
        if not hasattr(self, '_offset_prob_fixed'):
            self.offset_prob = 0.5 - 0.1 * (1.0 / new_theta)
        
        # 最小イベント長調整（最低5フレーム確保）
        self.min_event_frames = max(5, int(2.5 * new_theta))
        
        # オフセット検出の連続フレーム数も調整（外部指定がない場合のみ）
        if not hasattr(self, '_offset_wait_frames_fixed'):
            self.offset_wait_frames = max(5, int(4.0 + 1.0 * new_theta))  # 連続フレーム数を増加
        
        # ハーモニクス減衰関連のパラメータも調整（外部指定がない場合のみ）
        # 連続減衰フレーム数要件
        if not hasattr(self, '_harmonic_decay_frames_fixed'):
            self.harmonic_decay_frames = max(2, int(1.0 + 0.2 * new_theta))
        
        # 倍音減衰閾値 - thetaが大きいほど厳しめに
        base_decay_threshold = 0.6
        self.harmonic_decay_threshold = base_decay_threshold + 0.05 * (new_theta - 1.0)
        self.harmonic_decay_threshold = max(0.4, min(0.8, self.harmonic_decay_threshold))
        
        # 背景モデル更新速度係数 - thetaが大きいほど抑制的に
        self.harmonic_decay_alpha_factor = 1.2 + 0.05 * (new_theta - 1.0)
        self.harmonic_decay_alpha_factor = max(1.0, min(1.5, self.harmonic_decay_alpha_factor))
        
        # 既存の閾値も更新（従来機能との互換性維持）
        self.threshold = self.c_T * new_theta
        
        # ハーモニックモデルのパラメータも更新
        for model in self.harmonic_models.values():
            model.set_alpha_base(self.alpha_base)
        
        # ノート後処理のパラメータも更新
        # ノートの最小長さ（秒）- thetaに応じて調整
        self.min_note_duration = max(0.05, 0.02 * new_theta)
        
        # ノート間の最小ギャップ（秒）- これより短いギャップは統合対象
        self.min_gap_duration = max(0.08, 0.03 * new_theta)
        
        if self.enable_debug:
            logging.debug(f"メタパラメータを更新: theta={new_theta:.2f}, alpha_base={self.alpha_base:.4f}, " +
                         f"logistic_a={self.logistic_a:.2f}, logistic_b={self.logistic_b:.2f}, " +
                         f"onset_prob={self.onset_prob:.2f}, offset_prob={self.offset_prob:.2f}, " +
                         f"offset_wait_frames={self.offset_wait_frames}, " +
                         f"harmonic_decay_threshold={self.harmonic_decay_threshold:.2f}, " +
                         f"harmonic_decay_frames={self.harmonic_decay_frames}")

    def _compute_spectrum_deviation(self, spec_frame, freq_bins):
        """
        周波数領域ベースの逸脱量を計算
        
        Parameters
        ----------
        spec_frame : np.ndarray
            スペクトログラムフレーム
        freq_bins : np.ndarray
            周波数ビン
            
        Returns
        -------
        float
            スペクトル逸脱スコア S(t)
        """
        # 周波数範囲を制限（最小周波数から最大周波数まで）
        valid_indices = (freq_bins >= self.min_f0) & (freq_bins <= self.max_f0)
        
        if not np.any(valid_indices):
            return 0.0
        
        # 各ビンごとの逸脱量を計算するベクトル化処理
        total_deviation = 0.0
        
        # グローバルなインデックスを使用
        for bin_idx in np.where(valid_indices)[0]:
            freq = freq_bins[bin_idx]
            energy = spec_frame[bin_idx]
            
            # このビンの背景モデルを取得または作成
            # グローバルなインデックスをキーとして使用
            bin_key = f"bin_{bin_idx}"
            
            if bin_key not in self.freq_bg_models:
                # 新しいビンの場合、背景モデルを初期化
                self.freq_bg_models[bin_key] = {
                    'mean': energy,  # 初期平均
                    'var': max(energy * 0.1, 1e-4)  # 初期分散
                }
            
            # 現在のモデル値を取得
            bg_mean = self.freq_bg_models[bin_key]['mean']
            bg_var = self.freq_bg_models[bin_key]['var']
            
            # ビンごとの逸脱量を計算: Δ_k(t) = max(0, (|X_k(t)| - μ_k(t)) / (σ_k(t) + ε))
            delta_k = max(0, (energy - bg_mean) / (np.sqrt(bg_var) + self.eps))
            
            # 総逸脱量に加算
            total_deviation += delta_k
            
            # 背景モデルを更新（イベント中は更新速度を遅くする）
            alpha = self.alpha_event if self.in_event else self.alpha_base
            
            # 平均の更新
            self.freq_bg_models[bin_key]['mean'] = alpha * bg_mean + (1 - alpha) * energy
            
            # 分散の更新
            delta = energy - bg_mean
            self.freq_bg_models[bin_key]['var'] = max(
                alpha * bg_var + (1 - alpha) * delta * delta,
                1e-4  # 数値安定性のための最小値
            )
        
        # スペクトル逸脱スコア履歴の更新
        self.deviation_scores_history.append(total_deviation)
        if len(self.deviation_scores_history) > 200:
            self.deviation_scores_history = self.deviation_scores_history[-200:]
        
        # 正規化（オプション）- 周波数ビン数で正規化して音源間の違いを吸収
        normalized_deviation = total_deviation / np.sum(valid_indices) if np.sum(valid_indices) > 0 else 0.0
        
        return normalized_deviation
        
    def _determine_multiple_pitches(self, spec_frame, freq_bins, z_score):
        """
        複数のピッチを検出（ポリフォニック検出モード用）
        
        Parameters
        ----------
        spec_frame : np.ndarray
            スペクトログラムフレーム
        freq_bins : np.ndarray
            周波数ビン
        z_score : float
            マハラノビス距離に基づくZ-score
            
        Returns
        -------
        list
            検出されたピッチのリスト
        """
        # イベント中でない場合や逸脱度が低い場合は空リストを返す
        if not self.in_event or z_score < self.threshold:
            return []
        
        # ピッチ候補を見つける
        pitch_candidates = []
        
        # 周波数ビンの振幅ピークを探す
        peak_indices = []
        
        # 周波数範囲を制限
        valid_indices = (freq_bins >= self.min_f0) & (freq_bins <= self.max_f0)
        
        if not np.any(valid_indices):
            return []
        
        # より頑健なピーク検出
        for i in np.where(valid_indices)[0]:
            # 端のケースを考慮
            if i <= 0 or i >= len(spec_frame) - 1:
                continue
                
            # ローカルピークの検出
            if spec_frame[i] > spec_frame[i-1] and spec_frame[i] > spec_frame[i+1]:
                # より頑健な閾値計算
                # 1. 周辺のノイズフロア推定
                window_size = min(10, len(spec_frame) // 3)  # 窓サイズ（最大10または全長の1/3）
                start_idx = max(0, i - window_size)
                end_idx = min(len(spec_frame), i + window_size + 1)
                local_spec = spec_frame[start_idx:end_idx]
                
                # 下位25%の振幅をノイズフロアとして使用
                noise_floor = np.percentile(local_spec, 25)
                
                # 2. 局所的な信号対雑音比を計算
                local_snr = spec_frame[i] / (noise_floor + self.eps)
                
                # SNRが十分高い場合のみピークとして認識
                if local_snr > 3.0 or spec_frame[i] > np.mean(spec_frame) * 1.2:
                    peak_indices.append(i)
        
        # 各ピークを評価
        for idx in peak_indices:
            freq = freq_bins[idx]
            amplitude = spec_frame[idx]
            
            # 基本周波数候補を追加
            pitch_candidates.append((freq, amplitude))
            
            # 倍音関係を確認（簡易実装）
            for j in peak_indices:
                if j == idx:
                    continue
                
                ratio = freq_bins[j] / freq
                # 倍音関係にあるか
                if abs(round(ratio) - ratio) < 0.05 and round(ratio) > 1:
                    # 倍音スコアを加算
                    pitch_candidates[-1] = (freq, amplitude + spec_frame[j] / round(ratio))
        
        # 振幅でソートして上位を選択
        pitch_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 最大ポリフォニー数まで選択
        pitches = [p[0] for p in pitch_candidates[:self.max_polyphony]]
        
        return pitches

    def _logistic_transform(self, score):
        """
        スペクトル逸脱スコアをロジスティック変換してイベント確率に変換
        
        Parameters
        ----------
        score : float
            スペクトル逸脱スコア
            
        Returns
        -------
        float
            イベント確率 p_event [0-1]
        """
        # p_event(t) = σ(a(S(t) - b))
        p_event = 1.0 / (1.0 + np.exp(-self.logistic_a * (score - self.logistic_b)))
        return p_event

    def _update_background_model(self, feature_vector, alpha):
        """
        背景ガウス分布モデルを更新
        
        Parameters
        ----------
        feature_vector : np.ndarray
            新しい特徴ベクトル
        alpha : float
            指数平滑化係数 (高いほど過去の値を重視)
        """
        # 平均ベクトルを更新
        self.mu = alpha * self.mu + (1 - alpha) * feature_vector
        
        # 共分散行列を更新 (シンプルな指数平滑化)
        delta = feature_vector - self.mu
        delta_outer = np.outer(delta, delta)
        
        self.sigma = alpha * self.sigma + (1 - alpha) * delta_outer
        
        # 数値安定性のために対角成分に小さな値を加える
        self.sigma += np.eye(self.feature_dim) * 1e-5
        
        # 共分散行列の逆行列を再計算
        try:
            self.sigma_inv = np.linalg.inv(self.sigma)
        except np.linalg.LinAlgError:
            # 特異行列の場合は対角成分をさらに強化
            self.sigma += np.eye(self.feature_dim) * 1e-3
            self.sigma_inv = np.linalg.inv(self.sigma)

    def _determine_final_pitch(self, feature_vector, z_score):
        """
        最終的なピッチを決定
        
        Parameters
        ----------
        feature_vector : np.ndarray
            特徴ベクトル
        z_score : float
            マハラノビス距離に基づくZ-score
            
        Returns
        -------
        float
            最終的なピッチ (Hz)
        """
        # 時間領域ピッチ（特徴ベクトルの第0成分、対数スケール）
        log_pitch = feature_vector[0]
        pitch = np.exp(log_pitch) - 1.0  # 逆変換

        if pitch < self.min_f0 or pitch > self.max_f0:
            return 0.0
        
        # 周期確信度（特徴ベクトルの第1成分）
        periodicity = feature_vector[1]

        # z_scoreを活用して信頼性の低いピッチを除外
        # 背景からの逸脱が小さすぎる場合は棄却
        #if z_score < self.threshold * 0.7:
        #    return 0.0

        # ピッチ扱いにする閾値
        periodicity_threshold = 0.1
        
        # 閾値の決定
        if periodicity < periodicity_threshold:
            return 0.0
    
        return pitch
    
    def _optimize_offset_time(self, frame_time, buffer_size, sr, max_lookback=5):
        """
        オフセット時間を最適化（過去フレームの情報を使用）
        
        Parameters
        ----------
        frame_time : float
            現在のフレーム時間
        buffer_size : int
            バッファサイズ
        sr : int
            サンプリングレート
        max_lookback : int
            最大遡り検索フレーム数
            
        Returns
        -------
        float
            最適化されたオフセット時間
        """
        # デフォルト: 現在フレームの中央
        default_offset_time = frame_time + (buffer_size / (2 * sr))
        
        # 過去のフレームデータがない場合はデフォルト値を返す
        if len(self.recent_frame_data) <= 1:
            return default_offset_time
            
        # 参照する過去フレーム数を決定
        lookback = min(max_lookback, len(self.recent_frame_data) - 1)
        if lookback <= 0:
            return default_offset_time
        
        # ハーモニクス減衰情報を活用
        # 倍音減衰が十分に進行していた場合、より早期のオフセットを検討
        harmonic_decay_adjusted = False
        if hasattr(self, '_offset_decay_info'):
            decay_info = self._offset_decay_info
            
            # 十分な倍音が減衰していた場合は、より早いフレームを探す
            if decay_info['harmonic_decay_metric'] > 0.7 and decay_info['decaying_harmonics_count'] >= self.min_harmonics_for_offset:
                # より早期のオフセットを優先
                max_lookback = min(max_lookback + 2, len(self.recent_frame_data) - 1)
                lookback = max_lookback
                harmonic_decay_adjusted = True
                
                if self.enable_debug:
                    logging.debug(f"倍音減衰情報に基づくオフセット時間最適化: "
                                f"倍音減衰度={decay_info['harmonic_decay_metric']:.2f}, "
                                f"減衰倍音数={decay_info['decaying_harmonics_count']}")
        
        # 最適なオフセットフレームを探索
        best_idx = -1  # 現在のフレーム
        min_energy_ratio = 1.0
        baseline_energy = self.recent_frame_data[-1]['energy']
        
        # エネルギーが急激に減少するフレームを探す
        for i in range(1, lookback + 1):
            if len(self.recent_frame_data) > i:
                idx = -1 - i  # 現在から i フレーム前
                prev_energy = self.recent_frame_data[idx]['energy']
                
                # エネルギー比率（現在のフレームに対する比率）
                if baseline_energy > 0:
                    energy_ratio = prev_energy / baseline_energy
                else:
                    energy_ratio = 1.0
                
                # より低いエネルギー比率のフレームを見つけた場合、それを選択
                # 倍音減衰が顕著な場合は、より早いエネルギー減少ポイントを重視
                if harmonic_decay_adjusted:
                    energy_threshold = 0.6  # より厳しい閾値で早期のエネルギー減少を検出
                    if energy_ratio < energy_threshold and energy_ratio < min_energy_ratio:
                        min_energy_ratio = energy_ratio
                        best_idx = idx
                else:
                    # 通常の処理
                    if energy_ratio < min_energy_ratio:
                        min_energy_ratio = energy_ratio
                        best_idx = idx
        
        # 最適なフレームが見つからなかった場合はデフォルト
        if best_idx == -1 or abs(best_idx) >= len(self.recent_frame_data):
            # 倍音減衰が顕著な場合は、少し早めにオフセットを設定
            if harmonic_decay_adjusted:
                # 早めに設定（1-2フレーム前）
                hop_time = self.hop_length / sr
                return frame_time - (hop_time * 1.5) + (buffer_size / (2 * sr))
            return default_offset_time
            
        # フレーム間のホップ数を計算
        hop_length = self.hop_length
        hop_time = hop_length / sr
        
        # 最適なオフセット時間を計算（ホップ時間単位で調整）
        offset_frames_back = abs(best_idx + 1)  # +1は-1が現在のフレームだから
        # 負の時間にならないよう制限
        max_frames_back = (frame_time * sr - buffer_size/2) / hop_length
        offset_frames_back = min(offset_frames_back, max_frames_back)
        optimized_offset_time = max(0.0, frame_time - (offset_frames_back * hop_time) + (buffer_size / (2 * sr)))
        
        # 倍音減衰に基づく微調整
        if harmonic_decay_adjusted and min_energy_ratio > 0.4:
            # エネルギー減少が明確でない場合、少し早めに
            optimized_offset_time = max(0.0, optimized_offset_time - hop_time * 0.5)
        
        if self.enable_debug:
            frames_adjusted = offset_frames_back
            logging.debug(f"オフセット時間最適化: 現在時間={frame_time:.3f}s → 最適時間={optimized_offset_time:.3f}s "
                        f"({frames_adjusted}フレーム調整, エネルギー比={min_energy_ratio:.2f})")
        
        # 一時情報をクリア
        if hasattr(self, '_offset_decay_info'):
            delattr(self, '_offset_decay_info')
        
        return optimized_offset_time

    def detect(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        オーディオデータからピッチとノートを検出
        
        Parameters
        ----------
        audio_data : np.ndarray
            オーディオ信号データ
        sr : int
            サンプリングレート
            
        Returns
        -------
        Dict[str, Any]
            検出結果を含む辞書:
            - onsets: 検出されたオンセット時間（秒）
            - offsets: 検出されたオフセット時間（秒）
            - pitches: 各ノートのピッチ（Hz）
            - confidences: 各ノートの信頼度
        """
        # 入力データの基本的な検証を追加
        if audio_data is None or len(audio_data) == 0:
            logging.warning("空のオーディオデータが渡されました。")
            return {
                'intervals': np.array([]),
                'note_pitches': np.array([]),
                'frame_times': np.array([]),
                'frame_frequencies': np.array([]),
                'additional_data': {
                    'onsets': np.array([]),
                    'offsets': np.array([]),
                    'pitches': np.array([]),
                    'confidences': np.array([]),
                    'frame_energies': np.array([]),
                    'frame_p_events': np.array([]),
                    'frame_harmonic_metrics': np.array([])
                }
            }
        
        # 異常値の処理（NaNや無限大）
        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
            logging.warning("オーディオデータにNaNまたは無限大の値が含まれています。これらの値を0に置き換えます。")
            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # サンプリングレートの検証
        if sr <= 0:
            logging.error(f"無効なサンプリングレート: {sr}")
            sr = self.sr  # デフォルト値を使用
            logging.warning(f"デフォルトのサンプリングレート {sr} を使用します。")
            
        # サンプリングレートが違う場合はリサンプリング
        if sr != self.sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sr)
            sr = self.sr
        
        # バッファサイズとホップ長を設定
        buffer_size = self.buffer_size
        hop_length = self.hop_length
        
        # 結果格納用の変数
        onsets = []
        offsets = []
        pitches = []
        confidences = []
        
        # フレームごとのピッチとタイム情報を保存
        frame_times = []
        frame_frequencies = []
        frame_energies = []
        frame_p_events = []
        frame_harmonic_metrics = []  # 各フレームの倍音減衰度を保存
        
        # デバッグ情報用に検出したオフセットとその調整量を記録
        detected_offsets = []  # [frame_time, adjusted_time, adjustment]
        
        # 現在の検出状態
        current_onset = None
        current_pitch_sum = 0
        current_pitch_count = 0
        note_frames = []  # ノートに含まれるフレームデータ
        
        # プログレス表示（長いファイル処理時）
        total_frames = (len(audio_data) - buffer_size) // hop_length
        progress_step = max(1, total_frames // 20)  # 5%ごとに表示
        
        # 長時間音声データの場合はメモリ使用量を最適化
        if total_frames > 10000:  # 約3分以上の音声
            import gc  # ガベージコレクション用
            logging.info(f"長時間音声データを検出: {total_frames}フレーム、メモリ最適化を適用")
            # 定期的にメモリを解放するための処理
            gc.collect()  # ガベージコレクション強制実行
        
        if self.enable_debug:
            logging.debug(f"検出開始: 音声長={len(audio_data)/sr:.2f}秒, 総フレーム数={total_frames}")
        
        # フレームごとに処理
        for i in range(0, len(audio_data) - buffer_size, hop_length):
            frame_idx = i // hop_length
            
            # プログレス表示
            if self.enable_debug and frame_idx % progress_step == 0:
                progress = frame_idx / total_frames * 100
                logging.debug(f"検出進捗: {progress:.1f}% ({frame_idx}/{total_frames}フレーム)")
            
            # 現在のフレーム時間を計算
            frame_time = i / sr
            frame_times.append(frame_time)
            
            # バッファを取得
            buffer = audio_data[i:i+buffer_size]
            
            # バッファが短い場合はパディング
            if len(buffer) < buffer_size:
                buffer = np.pad(buffer, (0, buffer_size - len(buffer)), 'constant')
            
            # フレームを処理
            frame_result = self.process_frame(buffer)
            
            # 現在のフレームのピッチとエネルギーを保存
            current_frame_pitch = frame_result['pitch']
            frame_frequencies.append(current_frame_pitch)
            frame_p_events.append(frame_result['p_event'])
            frame_harmonic_metrics.append(self._harmonic_decay_metric)
            
            # エネルギー計算（スペクトラムの合計）
            if hasattr(self, '_current_spec_frame') and self._current_spec_frame is not None:
                current_energy = np.sum(self._current_spec_frame)
            else:
                current_energy = 0.0
            frame_energies.append(current_energy)
            
            # オンセットの検出
            if frame_result['is_onset']:
                # 新しいノートの開始
                current_onset = frame_time
                current_pitch_sum = 0
                current_pitch_count = 0
                note_frames = []  # ノートフレームをリセット
                
                if current_frame_pitch > 0:
                    current_pitch_sum += current_frame_pitch
                    current_pitch_count += 1
                
                # このフレームをノートに追加
                note_frames.append({
                    'time': frame_time,
                    'pitch': current_frame_pitch,
                    'energy': current_energy,
                    'p_event': frame_result['p_event'],
                    'harmonic_metric': self._harmonic_decay_metric
                })
            
            # ノート中のピッチ追跡
            elif current_onset is not None:
                # このフレームをノートに追加
                note_frames.append({
                    'time': frame_time,
                    'pitch': current_frame_pitch,
                    'energy': current_energy,
                    'p_event': frame_result['p_event'],
                    'harmonic_metric': self._harmonic_decay_metric
                })
                
                if current_frame_pitch > 0:
                    current_pitch_sum += current_frame_pitch
                    current_pitch_count += 1
            
            # オフセットの検出
            if frame_result['is_offset'] and current_onset is not None:
                # オフセット時間を最適化 (実際の音が消えたタイミングに近づける)
                offset_time = self._optimize_offset_time(frame_time, buffer_size, sr, max_lookback=5)
                
                # デバッグ情報を保存
                if self.enable_debug:
                    time_adjustment = offset_time - (frame_time + (buffer_size / (2 * sr)))
                    detected_offsets.append((frame_time, offset_time, time_adjustment))
                
                # 平均ピッチを計算
                avg_pitch = 0
                if current_pitch_count > 0:
                    avg_pitch = current_pitch_sum / current_pitch_count
                
                # 有効なピッチの場合のみ記録
                if avg_pitch >= self.min_f0 and avg_pitch <= self.max_f0:
                    onsets.append(current_onset)
                    offsets.append(offset_time)
                    pitches.append(avg_pitch)
                    confidences.append(frame_result['p_event'])
                    
                    if self.enable_debug:
                        note_duration = offset_time - current_onset
                        logging.debug(f"ノート検出: {current_onset:.3f}s → {offset_time:.3f}s (長さ: {note_duration:.3f}s, ピッチ: {avg_pitch:.1f}Hz)")
                
                # 状態をリセット
                current_onset = None
                current_pitch_sum = 0
                current_pitch_count = 0
                note_frames = []
        
        # 最後のノートが終了していない場合
        if current_onset is not None:
            # 最後のフレームの中央を使用
            offset_time = (len(audio_data) - buffer_size/2) / sr
            
            # 平均ピッチを計算
            avg_pitch = 0
            if current_pitch_count > 0:
                avg_pitch = current_pitch_sum / current_pitch_count
            
            # 有効なピッチの場合のみ記録
            if avg_pitch >= self.min_f0 and avg_pitch <= self.max_f0:
                onsets.append(current_onset)
                offsets.append(offset_time)
                pitches.append(avg_pitch)
                confidences.append(0.8)  # デフォルト信頼度
                
                if self.enable_debug:
                    note_duration = offset_time - current_onset
                    logging.debug(f"最終ノート: {current_onset:.3f}s → {offset_time:.3f}s (長さ: {note_duration:.3f}s, ピッチ: {avg_pitch:.1f}Hz)")
        
        # フレーム情報を後処理用に保存
        self.detect_frames_info = {
            'frame_times': np.array(frame_times),
            'frame_frequencies': np.array(frame_frequencies),
            'frame_energies': np.array(frame_energies),
            'frame_p_events': np.array(frame_p_events),
            'frame_harmonic_metrics': np.array(frame_harmonic_metrics),
            'detected_offsets': detected_offsets
        }
        
        # 短すぎるノートのフィルタリングと近接ノートの統合
        if len(onsets) > 0:
            if self.enable_debug:
                logging.debug(f"後処理前ノート数: {len(onsets)}")
                
            filtered_onsets, filtered_offsets, filtered_pitches, filtered_confidences = self._post_process_notes(
                onsets, offsets, pitches, confidences, sr)
            
            onsets = filtered_onsets
            offsets = filtered_offsets
            pitches = filtered_pitches
            confidences = filtered_confidences
            
            if self.enable_debug:
                logging.debug(f"後処理後ノート数: {len(onsets)}")
        
        # 検出結果を標準フォーマットに変換
        intervals = []
        note_pitches = []
        
        for onset, offset, pitch in zip(onsets, offsets, pitches):
            intervals.append([onset, offset])
            note_pitches.append(pitch)
        
        # 結果を辞書形式で返す
        result = {
            'intervals': np.array(intervals),
            'note_pitches': np.array(note_pitches),
            'frame_times': np.array(frame_times),
            'frame_frequencies': np.array(frame_frequencies),
            'additional_data': {
                'onsets': np.array(onsets),
                'offsets': np.array(offsets),
                'pitches': np.array(pitches),
                'confidences': np.array(confidences),
                'frame_energies': np.array(frame_energies),
                'frame_p_events': np.array(frame_p_events),
                'frame_harmonic_metrics': np.array(frame_harmonic_metrics)
            }
        }
        
        return result
        
    def _post_process_notes(self, onsets, offsets, pitches, confidences, sr):
        """
        検出したノートの後処理（短いノートの除去、近接ノートの統合、音楽的ルールの適用）
        
        Parameters
        ----------
        onsets : list
            オンセット時間のリスト
        offsets : list
            オフセット時間のリスト
        pitches : list
            ピッチのリスト
        confidences : list
            信頼度のリスト
        sr : int
            サンプリングレート
            
        Returns
        -------
        tuple
            (新しいonsets, 新しいoffsets, 新しいpitches, 新しいconfidences)
        """
        if len(onsets) == 0:
            return [], [], [], []
        
        # detectメソッドで計算されたフレームタイム・エネルギー情報を取得
        frame_times = None
        frame_energies = None
        if hasattr(self, 'detect_frames_info'):
            frame_times = self.detect_frames_info.get('frame_times')
            frame_energies = self.detect_frames_info.get('frame_energies')
        
        # 処理済みノートのリスト
        new_onsets = []
        new_offsets = []
        new_pitches = []
        new_confidences = []
        
        # 最初のノートを処理済みリストに追加
        prev_onset = onsets[0]
        prev_offset = offsets[0]
        prev_pitch = pitches[0]
        prev_confidence = confidences[0]
        
        # 統合理由を記録（デバッグ用）
        merge_reasons = []
        
        for i in range(1, len(onsets)):
            curr_onset = onsets[i]
            curr_offset = offsets[i]
            curr_pitch = pitches[i]
            curr_confidence = confidences[i]
            
            # 現在のノートの長さと前のノートとの間隔
            curr_duration = curr_offset - curr_onset
            gap_duration = curr_onset - prev_offset
            
            # 最も高いピッチ / 最も低いピッチの比率を計算
            if prev_pitch > 0 and curr_pitch > 0:
                pitch_ratio = max(prev_pitch, curr_pitch) / min(prev_pitch, curr_pitch)
                is_musical, interval_type = self._is_musical_interval(pitch_ratio)
            else:
                pitch_ratio = float('inf')
                is_musical = False
                interval_type = "unknown"
            
            # エネルギー遷移の分析（フレーム情報がある場合のみ）
            energy_analysis = {"is_continuous": False}
            if frame_times is not None and frame_energies is not None:
                energy_analysis = self._analyze_energy_transition(
                    curr_onset, prev_offset, frame_times, frame_energies
                )
            
            # ノート統合の条件チェック
            should_merge = False
            merge_reason = ""
            
            # 1. 短いノートの場合は統合を検討
            if curr_duration < self.min_note_duration:
                should_merge = True
                merge_reason = f"短いノート（{curr_duration:.3f}秒）"
            
            # 2. ギャップが小さい場合は統合を検討
            if gap_duration < self.min_gap_duration:
                # ピッチが近い場合は統合
                if pitch_ratio < self.max_pitch_ratio:
                    should_merge = True
                    merge_reason = f"短いギャップ（{gap_duration:.3f}秒）+ 近いピッチ（比率:{pitch_ratio:.3f}）"
                
                # レガート判定：エネルギーが連続している場合
                elif energy_analysis["is_continuous"]:
                    # 音楽的に意味のあるピッチ変化の場合はレガートとして扱う
                    if is_musical and interval_type in ["semitone", "wholetone", "third"]:
                        should_merge = True
                        merge_reason = f"レガート奏法（{interval_type}, エネルギー連続性:{energy_analysis['energy_drop_ratio']:.2f}）"
            
            # 3. オクターブジャンプの特別処理
            if is_musical and interval_type == "octave":
                # オクターブジャンプで、エネルギーが大きく変化していない場合は
                # 倍音の誤検出の可能性が高いため統合
                if energy_analysis["energy_drop_ratio"] > 0.5:
                    should_merge = True
                    merge_reason = f"オクターブ跳躍の誤検出の可能性（エネルギー連続性:{energy_analysis['energy_drop_ratio']:.2f}）"
            
            # ノートを統合するか、確定するか判断
            if should_merge:
                # ノートを統合
                prev_offset = curr_offset
                
                # ピッチは重み付け平均（長さで重み付け）
                prev_duration = prev_offset - prev_onset
                if prev_duration + curr_duration > 0:
                    # レガートの場合は加重平均ではなく2つのノートを区別して残す
                    if "レガート" in merge_reason:
                        # ピッチ変化の遷移を考慮（単純に最後のピッチにはしない）
                        # 前半と後半でピッチを分け、滑らかな変化を表現
                        prev_pitch = prev_pitch  # 変更なし、以降のロジックで必要に応じて変更
                    else:
                        # 通常のノート統合（加重平均）
                        prev_pitch = (prev_pitch * prev_duration + curr_pitch * curr_duration) / (prev_duration + curr_duration)
                
                # 信頼度は最大値を採用
                prev_confidence = max(prev_confidence, curr_confidence)
                
                # 統合理由を記録
                merge_reasons.append(merge_reason)
            else:
                # 前のノートを確定
                if prev_offset - prev_onset >= self.min_note_duration:
                    new_onsets.append(prev_onset)
                    new_offsets.append(prev_offset)
                    new_pitches.append(prev_pitch)
                    new_confidences.append(prev_confidence)
                    
                    # デバッグ情報表示
                    if self.enable_debug and merge_reasons:
                        logging.debug(f"ノート統合: {prev_onset:.3f}-{prev_offset:.3f}秒 ({prev_pitch:.1f}Hz), 理由: {', '.join(merge_reasons)}")
                        merge_reasons = []
                
                # 現在のノートを次の比較対象に
                prev_onset = curr_onset
                prev_offset = curr_offset
                prev_pitch = curr_pitch
                prev_confidence = curr_confidence
        
        # 最後のノートの処理
        if prev_offset - prev_onset >= self.min_note_duration:
            new_onsets.append(prev_onset)
            new_offsets.append(prev_offset)
            new_pitches.append(prev_pitch)
            new_confidences.append(prev_confidence)
            
            # デバッグ情報表示
            if self.enable_debug and merge_reasons:
                logging.debug(f"ノート統合: {prev_onset:.3f}-{prev_offset:.3f}秒 ({prev_pitch:.1f}Hz), 理由: {', '.join(merge_reasons)}")
        
        if self.enable_debug and len(new_onsets) < len(onsets):
            logging.debug(f"ノート後処理: {len(onsets)}個 → {len(new_onsets)}個 ({len(onsets) - len(new_onsets)}個削減)")
        
        return new_onsets, new_offsets, new_pitches, new_confidences

    def _is_musical_interval(self, pitch_ratio, tolerance=0.03):
        """
        ピッチ比が音楽的な間隔（半音、全音、オクターブなど）かどうかを判定
        
        Parameters
        ----------
        pitch_ratio : float
            ピッチ比率（高い音/低い音）
        tolerance : float
            許容誤差
            
        Returns
        -------
        tuple(bool, str)
            音楽的間隔かどうかのフラグと、間隔の種類
        """
        # ピッチ比が1に近い場合（同一音）
        if abs(pitch_ratio - 1.0) < tolerance:
            return True, "unison"
            
        # オクターブ関係（比率 ≈ 2.0）
        if abs(pitch_ratio - 2.0) < tolerance * 2:
            return True, "octave"
            
        # 5度（比率 ≈ 1.5）
        if abs(pitch_ratio - 1.5) < tolerance * 1.5:
            return True, "fifth"
            
        # 4度（比率 ≈ 1.33）
        if abs(pitch_ratio - 4/3) < tolerance * 1.5:
            return True, "fourth"
            
        # 長3度・短3度（比率 ≈ 1.25-1.2）
        if abs(pitch_ratio - 5/4) < tolerance * 1.5 or abs(pitch_ratio - 6/5) < tolerance * 1.5:
            return True, "third"
            
        # 半音（比率 ≈ 1.0595）
        semitone_ratio = 2**(1/12)  # 約1.0595
        if abs(pitch_ratio - semitone_ratio) < tolerance:
            return True, "semitone"
            
        # 全音（比率 ≈ 1.1225）
        wholetone_ratio = 2**(2/12)  # 約1.1225
        if abs(pitch_ratio - wholetone_ratio) < tolerance:
            return True, "wholetone"
            
        return False, "not_musical"
        
    def _analyze_energy_transition(self, curr_onset, prev_offset, frame_times, frame_energies):
        """
        ノート間のエネルギー遷移を分析
        
        Parameters
        ----------
        curr_onset : float
            現在のノートのオンセット時間
        prev_offset : float
            前のノートのオフセット時間
        frame_times : np.ndarray
            全フレームの時間情報
        frame_energies : np.ndarray
            全フレームのエネルギー情報
            
        Returns
        -------
        dict
            エネルギー遷移の分析結果
        """
        # 該当時間範囲のフレームインデックスを特定
        gap_start_idx = np.searchsorted(frame_times, prev_offset) - 1
        gap_end_idx = np.searchsorted(frame_times, curr_onset) + 1
        
        # 範囲が不正な場合は空の結果を返す
        if gap_start_idx < 0 or gap_end_idx >= len(frame_times) or gap_end_idx <= gap_start_idx:
            return {
                "energy_continuity": 0.0,
                "energy_drop_ratio": 1.0,
                "is_continuous": False
            }
        
        # ギャップ前後のエネルギー値を取得
        pre_energies = frame_energies[max(0, gap_start_idx-2):gap_start_idx+1]
        gap_energies = frame_energies[gap_start_idx:gap_end_idx+1]
        post_energies = frame_energies[gap_end_idx:min(len(frame_energies), gap_end_idx+3)]
        
        # 前後のエネルギー平均
        pre_energy = np.mean(pre_energies) if len(pre_energies) > 0 else 0
        post_energy = np.mean(post_energies) if len(post_energies) > 0 else 0
        gap_min_energy = np.min(gap_energies) if len(gap_energies) > 0 else 0
        
        # 最小値が両端の平均に対してどれくらい減少しているか
        energy_avg = (pre_energy + post_energy) / 2 if pre_energy > 0 and post_energy > 0 else max(pre_energy, post_energy)
        energy_drop_ratio = gap_min_energy / energy_avg if energy_avg > 0 else 0
        
        # エネルギーの連続性を評価（1に近いほど連続的）
        energy_continuity = 1.0 - (1.0 - energy_drop_ratio) * 2
        energy_continuity = max(0, min(1, energy_continuity))
        
        # エネルギーがある程度連続しているかを判断
        is_continuous = energy_drop_ratio > 0.7  # 70%以上のエネルギーが維持されていれば連続と判断
        
        return {
            "energy_continuity": energy_continuity,
            "energy_drop_ratio": energy_drop_ratio,
            "is_continuous": is_continuous
        }