"""
FAZEDetector: FAZΞ-CN（Frequency Amplitude Z-domain Extraction with Curvature Nonperiodicity）アルゴリズムの実装

このアルゴリズムは以下の特徴を持ちます：
1. 定Q複素共振フィルタバンク：音声信号を複数の帯域に分割し、各帯域ごとに詳細な情報（複素包絡）を抽出
2. 自己微分積（Ξ）：複素包絡の時間的な微小変化を敏感に捉え、オンセットやピッチ変動の検出に活用
3. 曲率エントロピー：音の周期的・非周期的特性を統計的に評価し、打楽器音のような非調性音を識別

詳細な仕様は algos/faze.md を参照してください。
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from scipy import signal
from scipy.stats import entropy
import matplotlib.pyplot as plt
import logging
import time
import os
import librosa
import librosa.display
import librosa.onset
import librosa.effects
import scipy.fftpack
from scipy.ndimage import gaussian_filter1d
from src.detectors.base_detector import BaseDetector
from src.utils.detection_result import DetectionResult

class FAZEDetector(BaseDetector):
    """
    FAZΞ-CN（Frequency Amplitude Z-domain Extraction with Curvature Nonperiodicity）アルゴリズム
    
    音声信号からオンセット、オフセット、ピッチ、音量、非周期性指標を抽出するための検出器です。
    定Q複素共振フィルタバンク、自己微分積（Ξ）、曲率エントロピーを組み合わせて実装しています。
    """
    
    def __init__(self,
                 bands: int = 72,                 # バンド数（36→72に増加）
                 f_min: float = 30.0,             # 最低中心周波数 [Hz]（55→30に引き下げ）
                 beta: int = 12,                  # オクターブあたりのバンド数
                 q_factor: float = 24.0,          # 品質係数（Q値）（16→24に引き上げ）
                 frame_length: int = 1024,        # フレーム長（512→1024に増加）
                 hop_length: int = 512,           # ホップ長（256→512に増加）
                 hist_bins: int = 24,             # ヒストグラムのビン数（12→24に増加）
                 onset_threshold_factor: float = 2.5,  # オンセット検出閾値係数
                 offset_threshold_factor: float = 2.5, # オフセット検出閾値係数
                 nonperiodicity_threshold: float = 0.6, # 非周期音判定閾値
                 min_note_length: float = 0.1,     # 最小ノート長（秒）
                 moving_avg_window: int = 10,      # 動的閾値計算用の移動平均ウィンドウ
                 consecutive_offset_frames: int = 3, # オフセット検出に必要な連続フレーム数
                 debug: bool = False,              # デバッグモード
                 **kwargs):
        """
        初期化
        
        Parameters
        ----------
        bands : int, optional
            バンド数, by default 72 (A1~C8の広域をカバー、従来の36から増加)
        f_min : float, optional
            最低中心周波数 [Hz], by default 30.0 (より低い音域をカバー)
        beta : int, optional
            オクターブあたりのバンド数, by default 12
        q_factor : float, optional
            品質係数（Q値）, by default 24.0 (周波数分解能向上のため増加)
        frame_length : int, optional
            フレーム長, by default 1024 (周波数分解能向上のため増加)
        hop_length : int, optional
            ホップ長, by default 512 (時間分解能と周波数分解能のバランス)
        hist_bins : int, optional
            ヒストグラムのビン数, by default 24 (より細かい曲率分布の評価)
        onset_threshold_factor : float, optional
            オンセット検出閾値係数, by default 2.5
        offset_threshold_factor : float, optional
            オフセット検出閾値係数, by default 2.5
        nonperiodicity_threshold : float, optional
            非周期音判定閾値, by default 0.6
        min_note_length : float, optional
            最小ノート長(秒), by default 0.1
        moving_avg_window : int, optional
            動的閾値計算用の移動平均ウィンドウ, by default 10
        consecutive_offset_frames : int, optional
            オフセット検出に必要な連続フレーム数, by default 3
        debug : bool, optional
            デバッグモード, by default False
        """
        super().__init__(config=kwargs)
        
        # パラメータの保存
        self.bands = bands
        self.f_min = f_min
        self.beta = beta
        self.q_factor = q_factor
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.hist_bins = hist_bins
        self.onset_threshold_factor = onset_threshold_factor
        self.offset_threshold_factor = offset_threshold_factor
        self.nonperiodicity_threshold = nonperiodicity_threshold
        self.min_note_length = min_note_length
        self.moving_avg_window = moving_avg_window
        self.consecutive_offset_frames = consecutive_offset_frames
        self.debug = debug
        
        # フィルタバンク関連のパラメータ
        self.center_freqs = None
        self.filter_coefs = None
        self.sr = None  # サンプリングレートは音声読み込み時に設定
        
        # 検出結果の保存用変数
        self.xi_frames = None  # 自己微分積のフレームごとの合計
        self.np_indices = None  # 非周期性指標
        self.frame_energies = None  # フレームごとのエネルギー
        
        # ロガーの設定
        self.logger = logging.getLogger(__name__)
        
        # ファイルにもログを出力する設定を追加
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        if debug:
            self.logger.setLevel(logging.DEBUG)
            # コンソールにもDEBUGレベルのログを出力
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        else:
            self.logger.setLevel(logging.INFO)
        
        # 初期設定の理論的な中心周波数範囲をログ出力
        f_max_theory = f_min * (2 ** ((bands - 1) / beta))
        self.logger.info(f"理論上の周波数範囲: {f_min} - {f_max_theory} Hz (bands={bands}, beta={beta})")
        
        # 音楽的に重要な周波数（C3～C6）をカバーしているか確認
        important_notes = {
            'C3': 130.81, 'E3': 164.81, 'G3': 196.00, 'C4': 261.63, 
            'E4': 329.63, 'G4': 392.00, 'C5': 523.25, 'A4': 440.00,
            'E5': 659.25, 'G5': 783.99, 'C6': 1046.50
        }
        
        covered_notes = []
        uncovered_notes = []
        
        for note, freq in important_notes.items():
            if freq <= f_max_theory:
                covered_notes.append(f"{note} ({freq:.2f} Hz)")
            else:
                uncovered_notes.append(f"{note} ({freq:.2f} Hz)")
                
        self.logger.info(f"カバーする音域: {', '.join(covered_notes)}")
        if uncovered_notes:
            self.logger.warning(f"カバーしない音域: {', '.join(uncovered_notes)}")
        else:
            self.logger.info("すべての重要な音域をカバーしています")
        
    def _design_filterbank(self, sr: int) -> None:
        """
        定Q複素共振フィルタバンクを設計する（直接離散時間極方式）
        
        Parameters
        ----------
        sr : int
            サンプリングレート [Hz]
        """
        # サンプリングレートをインスタンス変数として保存
        self.sr = sr
        
        self.logger.debug(f"フィルタバンク設計用のサンプリングレート: {sr} Hz")
        # サンプリングレートの妥当性確認
        if sr < 8000 or sr > 192000:
            self.logger.warning(f"不適切なサンプリングレート: {sr} Hz (通常は8k-192kHz)")
            
        # サンプリング周期
        T = 1.0 / sr
        
        # 中心周波数の計算
        self.center_freqs = np.zeros(self.bands)
        for b in range(self.bands):
            # 仕様書に合わせて (b-1)/beta を使用
            self.center_freqs[b] = self.f_min * (2 ** ((b-1) / self.beta))
        
        self.logger.debug(f"中心周波数: {self.center_freqs}")
        self.logger.debug(f"最低中心周波数: {self.center_freqs[0]:.2f} Hz (A1付近)")
        self.logger.debug(f"最高中心周波数: {self.center_freqs[-1]:.2f} Hz")
        
        # 音域カバレッジのログ出力
        piano_a1 = 55.0  # A1の周波数
        piano_c8 = 4186.0  # C8の周波数
        coverage_low = self.center_freqs[0] <= piano_a1 * 1.05  # 5%の余裕を見る
        coverage_high = self.center_freqs[-1] >= piano_c8 * 0.95  # 5%の余裕を見る
        self.logger.debug(f"A1(55Hz)カバー: {'OK' if coverage_low else 'NG'}")
        self.logger.debug(f"C8(4186Hz)カバー: {'OK' if coverage_high else 'NG - 最高周波数が足りません'}")
        
        # 音楽的に重要ないくつかの周波数のカバー状況を確認
        important_freqs = [55.0, 110.0, 261.63, 440.0, 880.0, 1760.0, 3520.0, 4186.0]  # A1, A2, C4, A4, A5, A6, A7, C8
        for freq in important_freqs:
            closest_band = np.argmin(np.abs(self.center_freqs - freq))
            closest_freq = self.center_freqs[closest_band]
            error_cents = 1200 * np.log2(freq / closest_freq) if closest_freq > 0 else float('inf')
            self.logger.debug(f"周波数 {freq:.2f} Hz の最近接バンド: {closest_band}, 中心周波数: {closest_freq:.2f} Hz, 誤差: {abs(error_cents):.2f} cents")
        
        # 各バンドのフィルタ係数を計算
        self.filter_coefs = []
        for b in range(self.bands):
            # アナログ角周波数
            omega_0 = 2.0 * np.pi * self.center_freqs[b]
            
            # 減衰率
            alpha = omega_0 / (2.0 * self.q_factor)
            
            # 直接離散時間極を計算
            rho = np.exp(-alpha * T)
            theta = omega_0 * T * np.sqrt(1.0 - 1.0 / (4.0 * self.q_factor**2))
            
            # 数値安定性チェック
            if rho > 0.9999:
                self.logger.warning(f"バンド {b}, 中心周波数 {self.center_freqs[b]:.2f} Hz: rho値が1に非常に近い ({rho:.8f}), 数値不安定の可能性あり")
            
            # IIRフィルタ係数（極の係数）
            a1 = -2.0 * rho * np.cos(theta)
            a2 = rho**2
            
            # 正規化係数
            K = (1.0 - rho**2) / 2.0
            
            # 実部チャネルの零点係数
            b0_cos = K * (1.0 - a1 + a2)
            b1_cos = K * 2.0 * (a2 - 1.0)
            b2_cos = K * (1.0 + a1 + a2)
            
            # 虚部チャネルの零点係数
            b0_sin = K * (1.0 - a2)
            b1_sin = 0.0
            b2_sin = -K * (1.0 - a2)
            
            # フィルタ係数をまとめて保存
            coefs = {
                'a': np.array([1.0, a1, a2]),              # 分母（共通）
                'b_cos': np.array([b0_cos, b1_cos, b2_cos]), # 分子（実部）
                'b_sin': np.array([b0_sin, b1_sin, b2_sin]), # 分子（虚部）
                'rho': rho,
                'theta': theta,
                'center_freq': self.center_freqs[b]
            }
            self.filter_coefs.append(coefs)
            
            if b in [0, self.bands//4, self.bands//2, 3*self.bands//4, self.bands-1] or self.debug:
                self.logger.debug(f"バンド {b}, 中心周波数 {self.center_freqs[b]:.2f} Hz のフィルタ係数:")
                self.logger.debug(f"  減衰率: {alpha:.6f}, rho: {rho:.6f}, theta: {theta:.6f}")
                self.logger.debug(f"  a: {coefs['a']}")
                self.logger.debug(f"  b_cos: {coefs['b_cos']}")
                self.logger.debug(f"  b_sin: {coefs['b_sin']}")
        
        self.logger.debug("フィルタバンクの設計完了")
        
    def _apply_complex_filters(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        複素共振フィルタバンクを適用し、各バンドの複素包絡を計算する
        
        Parameters
        ----------
        audio : np.ndarray
            入力音声信号
        sr : int
            サンプリングレート [Hz]
            
        Returns
        -------
        Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]
            (自己微分積の合計, バンドごとの複素包絡, バンドごとの振幅)
        """
        if self.filter_coefs is None:
            self._design_filterbank(sr)
            
        # サンプル数と時間
        n_samples = len(audio)
        
        # サンプル数のバリデーション
        if n_samples < 3:
            self.logger.warning(f"音声データのサンプル数が少なすぎます: {n_samples} (最低3サンプル必要)")
            # 空の結果を返す（処理を続行できない）
            z_bands = [np.zeros(n_samples, dtype=np.complex128) for _ in range(self.bands)]
            r_bands = [np.zeros(n_samples) for _ in range(self.bands)]
            xi_total = np.zeros(n_samples, dtype=np.complex128)
            return xi_total, z_bands, r_bands
        
        # バンドごとの複素包絡と振幅
        z_bands = []         # 各バンドの複素包絡
        r_bands = []         # 各バンドの振幅
        kappa_bands = []     # 各バンドの曲率
        
        # 全バンド合算の自己微分積
        xi_total = np.zeros(n_samples, dtype=np.complex128)
        
        self.logger.debug(f"複素共振フィルタの適用開始（サンプル数: {n_samples}）")
        
        # 各バンドのフィルタ処理
        for b in range(self.bands):
            try:
                # フィルタ係数の取り出し
                a = self.filter_coefs[b]['a']
                b_cos = self.filter_coefs[b]['b_cos']
                b_sin = self.filter_coefs[b]['b_sin']
                
                # フィルタ適用（scipy.signal.lfilterを使用）
                y_cos = signal.lfilter(b_cos, a, audio)
                y_sin = signal.lfilter(b_sin, a, audio)
                
                # 複素包絡の形成
                z = y_cos + 1j * y_sin
                z_bands.append(z)
                
                # 振幅計算
                r = np.abs(z)
                r_bands.append(r)
                
                # 自己微分積の計算
                z_dot = np.zeros_like(z, dtype=np.complex128)
                # インデックスエラーを防ぐため、範囲を明示的に制限
                if len(z) > 1:  # 配列長が1以上の場合のみ実行
                    z_dot[1:] = z[1:] - z[:-1]  # 離散微分
                
                # 自己微分積の計算
                xi = np.conj(z) * z_dot  # 仕様書に基づく正しい計算方法
                xi_total += xi
                
                # 曲率計算用のバッファ準備
                kappa = np.zeros(n_samples)
                kappa_bands.append(kappa)
                
                if self.debug:
                    # バンド情報の表示
                    center_freq = self.center_freqs[b]
                    mean_amp = np.mean(r)
                    max_amp = np.max(r)
                    self.logger.debug(f"バンド {b}, 中心周波数 {center_freq:.2f} Hz: "
                                     f"平均振幅 {mean_amp:.6f}, 最大振幅 {max_amp:.6f}")
            except Exception as e:
                self.logger.error(f"バンド {b} の処理中にエラーが発生しました: {e}")
                # エラーが発生した場合は0配列を追加して続行
                z = np.zeros(n_samples, dtype=np.complex128)
                r = np.zeros(n_samples)
                kappa = np.zeros(n_samples)
                
                z_bands.append(z)
                r_bands.append(r)
                kappa_bands.append(kappa)
        
        self.logger.debug("複素共振フィルタの適用完了")
        
        return xi_total, z_bands, r_bands
        
    def _calculate_curvature(self, z_bands: List[np.ndarray], r_bands: List[np.ndarray]) -> List[np.ndarray]:
        """
        各バンドの複素包絡から曲率を計算する
        
        Parameters
        ----------
        z_bands : List[np.ndarray]
            各バンドの複素包絡
        r_bands : List[np.ndarray]
            各バンドの振幅
            
        Returns
        -------
        List[np.ndarray]
            各バンドの曲率
        """
        self.logger.debug("曲率の計算開始")
        
        # バリデーション
        if not z_bands or len(z_bands) == 0:
            self.logger.warning("z_bandsが空です。空の結果を返します。")
            return []
        
        n_samples = len(z_bands[0])
        kappa_bands = []
        
        for b in range(min(self.bands, len(z_bands))):
            try:
                z = z_bands[b]
                
                # 2次元ベクトル表現（実部・虚部）
                x_real = np.real(z)
                x_imag = np.imag(z)
                
                # 1階微分（中央差分）
                x_dot_real = np.zeros_like(x_real)
                x_dot_imag = np.zeros_like(x_imag)
                
                # インデックスエラーを防ぐためのチェック
                if len(x_real) > 2:  # 少なくとも3点必要
                    x_dot_real[1:-1] = (x_real[2:] - x_real[:-2]) / 2.0
                    x_dot_imag[1:-1] = (x_imag[2:] - x_imag[:-2]) / 2.0
                    
                    # 端点の処理（前方/後方差分）
                    x_dot_real[0] = x_real[1] - x_real[0]
                    x_dot_imag[0] = x_imag[1] - x_imag[0]
                    x_dot_real[-1] = x_real[-1] - x_real[-2]
                    x_dot_imag[-1] = x_imag[-1] - x_imag[-2]
                    
                    # 2階微分
                    x_ddot_real = np.zeros_like(x_real)
                    x_ddot_imag = np.zeros_like(x_imag)
                    
                    x_ddot_real[1:-1] = x_real[2:] - 2.0 * x_real[1:-1] + x_real[:-2]
                    x_ddot_imag[1:-1] = x_imag[2:] - 2.0 * x_imag[1:-1] + x_imag[:-2]
                    
                    # 端点の処理
                    if len(x_real) > 2:  # インデックス[2]にアクセスするために必要
                        x_ddot_real[0] = x_real[0] - 2.0 * x_real[1] + x_real[2]
                        x_ddot_imag[0] = x_imag[0] - 2.0 * x_imag[1] + x_imag[2]
                        
                        # 末尾の処理には少なくとも3点必要
                        if len(x_real) > 3: 
                            x_ddot_real[-1] = x_real[-3] - 2.0 * x_real[-2] + x_real[-1]
                            x_ddot_imag[-1] = x_imag[-3] - 2.0 * x_imag[-2] + x_imag[-1]
                        else:
                            # 3点しかない場合は別の方法で計算
                            x_ddot_real[-1] = x_ddot_real[0]  # 単純に最初の値をコピー
                            x_ddot_imag[-1] = x_ddot_imag[0]
                    
                    # クロス積計算
                    cross_prod = x_dot_real * x_ddot_imag - x_dot_imag * x_ddot_real
                    
                    # ベクトルのノルムの3乗
                    norm_cubed = (x_dot_real**2 + x_dot_imag**2) ** 1.5
                    
                    # 曲率計算（0除算回避）
                    epsilon = 1e-6
                    kappa = np.zeros_like(cross_prod)
                    mask = norm_cubed > epsilon
                    kappa[mask] = np.abs(cross_prod[mask]) / norm_cubed[mask]
                else:
                    # データ点が不足している場合は0で埋める
                    kappa = np.zeros_like(x_real)
                
                kappa_bands.append(kappa)
                
                if self.debug:
                    mean_kappa = np.mean(kappa)
                    max_kappa = np.max(kappa)
                    self.logger.debug(f"バンド {b}: 平均曲率 {mean_kappa:.6f}, 最大曲率 {max_kappa:.6f}")
                    
            except Exception as e:
                self.logger.error(f"バンド {b} の曲率計算中にエラーが発生しました: {e}")
                # エラーが発生した場合は0配列を追加
                kappa = np.zeros(n_samples)
                kappa_bands.append(kappa)
        
        # バンド数のチェック
        if len(kappa_bands) < self.bands:
            # 足りないバンドを0で埋める
            for _ in range(self.bands - len(kappa_bands)):
                kappa_bands.append(np.zeros(n_samples))
        
        self.logger.debug("曲率の計算完了")
        
        return kappa_bands
        
    def _calculate_entropy(self, kappa_bands: List[np.ndarray], r_bands: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        曲率からエントロピーを計算し、非周期性指標を求める
        
        Parameters
        ----------
        kappa_bands : List[np.ndarray]
            各バンドの曲率
        r_bands : List[np.ndarray]
            各バンドの振幅
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (フレームごとの非周期性指標, フレームごとのエネルギー)
        """
        self.logger.debug("エントロピーと非周期性指標の計算開始")
        
        n_samples = len(kappa_bands[0])
        # フレーム長を使ってフレーム数を計算（hop_lengthではなく）
        n_frames = int(np.ceil(n_samples / self.hop_length))
        
        # フレームごとの非周期性指標
        np_indices = np.zeros(n_frames)
        
        # フレームごとのエネルギー
        frame_energies = np.zeros(n_frames)
        
        # 各フレームの処理
        for m in range(n_frames):
            # フレームの開始・終了サンプル - hop_lengthでフレームを進める
            start = m * self.hop_length
            end = min(start + self.frame_length, n_samples)
            
            # 全バンドのエネルギー合計
            frame_energy = 0.0
            weighted_entropy_sum = 0.0
            
            # 各バンドのエントロピー計算
            for b in range(self.bands):
                r = r_bands[b][start:end]
                kappa = kappa_bands[b][start:end]
                
                # バンドのエネルギー（2乗平均）
                band_energy = np.mean(r**2)
                frame_energy += band_energy
                
                # 曲率のヒストグラム計算（エントロピー用）
                if band_energy > 0 and len(kappa) > 0:
                    # 0～1の範囲内のデータのみを使用
                    valid_kappa = kappa[(kappa >= 0) & (kappa <= 1)]
                    if len(valid_kappa) > 0:
                        # 入力が空の場合にエラーが発生するのを防ぐためにチェック
                        try:
                            hist, _ = np.histogram(valid_kappa, bins=self.hist_bins, range=(0, 1), density=True)
                            # シャノンエントロピー計算
                            epsilon = 1e-10  # 0 log 0 を避けるための小さな値
                            band_entropy = entropy(hist + epsilon)
                            # エネルギーで重み付けしたエントロピー
                            weighted_entropy_sum += band_energy * band_entropy
                        except Exception as e:
                            self.logger.warning(f"ヒストグラム計算中にエラーが発生しました: {e}")
                            # エラー発生時はこのバンドのエントロピーを0とする
                            continue
            
            # フレームエネルギーの保存
            frame_energies[m] = frame_energy
            
            # 非周期性指標の計算（エネルギー加重平均）
            if frame_energy > 0:
                np_indices[m] = weighted_entropy_sum / frame_energy
                # 最大1に正規化
                np_indices[m] = min(1.0, np_indices[m])
            else:
                np_indices[m] = 0.0
        
        self.logger.debug("エントロピーと非周期性指標の計算完了")
        
        return np_indices, frame_energies
        
    def _process_frames(self, xi_total: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        自己微分積をフレーム単位で処理する
        
        Parameters
        ----------
        xi_total : np.ndarray
            全バンド合算の自己微分積
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (実部フレーム合計, 虚部フレーム合計, フレーム時刻)
        """
        self.logger.debug("フレーム処理開始")
        
        # サンプル数とフレーム数
        n_samples = len(xi_total)
        
        # サンプル数が非常に少ない場合は特別処理
        if n_samples < self.hop_length:
            self.logger.warning(f"サンプル数がホップ長より少ないため、フレーム処理を省略します: サンプル数={n_samples}, ホップ長={self.hop_length}")
            # 単一フレームとみなす
            xi_real_frames = np.array([np.sum(np.real(xi_total))])
            xi_imag_frames = np.array([np.sum(np.imag(xi_total))])
            frame_times = np.array([0])
            return xi_real_frames, xi_imag_frames, frame_times
        
        # フレーム数を計算（切り上げ）- hop_lengthで一貫して計算
        n_frames = int(np.ceil(n_samples / self.hop_length))
        
        # 各フレームの自己微分積（実部・虚部）
        xi_real_frames = np.zeros(n_frames)
        xi_imag_frames = np.zeros(n_frames)
        
        # フレーム時刻
        frame_times = np.arange(n_frames) * self.hop_length
        
        # 各フレームの処理
        for m in range(n_frames):
            # フレームの開始・終了サンプル
            start = m * self.hop_length
            end = min(start + self.frame_length, n_samples)
            
            # フレーム内の自己微分積の実部と虚部の合計
            if end > start:  # 有効なサンプルがある場合のみ
                xi_real_frames[m] = np.sum(np.real(xi_total[start:end]))
                xi_imag_frames[m] = np.sum(np.imag(xi_total[start:end]))
            # end <= startの場合（通常発生しないが念のため）は0のまま
        
        self.logger.debug("フレーム処理完了")
        
        return xi_real_frames, xi_imag_frames, frame_times

    def _detect_onsets_offsets(self, xi_real_frames: np.ndarray, frame_times: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        フレーム処理された自己微分積の実部からオンセットとオフセットを検出
        
        Parameters
        ----------
        xi_real_frames : np.ndarray
            フレームごとの自己微分積実部の合計
        frame_times : np.ndarray
            フレーム時刻（サンプルインデックス）
        sr : int
            サンプリングレート
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (オンセット時刻の配列, オフセット時刻の配列)（単位：秒）
        """
        self.logger.debug("オンセット・オフセット検出開始")
        
        # フレーム数
        n_frames = len(xi_real_frames)
        
        # フレーム数が少ない場合は空の結果を返す
        if n_frames < 3:
            self.logger.warning(f"フレーム数が少なすぎます: {n_frames}. 少なくとも3フレーム必要です。空の結果を返します。")
            return np.array([]), np.array([])
        
        # 動的閾値計算のための統計量
        mu = np.zeros(n_frames)
        sigma = np.zeros(n_frames)
        
        # 最初のウィンドウ（冒頭フレーム）の統計量
        window_size = min(self.moving_avg_window, n_frames)
        if window_size > 0:
            mu[0] = np.mean(xi_real_frames[:window_size])
            sigma[0] = np.std(xi_real_frames[:window_size])
        
        # 以降のフレームでの移動統計量
        for m in range(1, n_frames):
            # ウィンドウの範囲を決定
            start = max(0, m - self.moving_avg_window)
            end = m
            
            if end > start:  # 有効なウィンドウがある場合
                mu[m] = np.mean(xi_real_frames[start:end])
                sigma[m] = np.std(xi_real_frames[start:end])
            else:  # ウィンドウが空の場合（通常発生しない）
                mu[m] = mu[m-1]
                sigma[m] = sigma[m-1]
        
        # 動的閾値の計算
        onset_threshold = mu + self.onset_threshold_factor * sigma
        offset_threshold = mu - self.offset_threshold_factor * sigma
        
        # オンセット/オフセット検出
        onsets = []
        offsets = []
        
        # 検出状態（音があるかどうか）
        in_note = False
        onset_frame = -1
        
        # オフセット検出用の連続カウント
        below_threshold_count = 0
        
        for m in range(n_frames):
            # オンセット検出: 閾値超過かつ局所ピーク
            is_local_peak = False
            if m > 0 and m < n_frames - 1:
                # 両隣との比較で厳密な局所ピーク判定（より仕様書に忠実に）
                is_local_peak = (xi_real_frames[m] > xi_real_frames[m-1]) and (xi_real_frames[m] > xi_real_frames[m+1])
            
            if not in_note and xi_real_frames[m] > onset_threshold[m] and is_local_peak:
                # 音の開始
                in_note = True
                onset_frame = m
                onsets.append(frame_times[m] / sr)
                # オフセットカウンタをリセット
                below_threshold_count = 0
            
            # オフセット検出: 閾値以下が一定フレーム数連続
            elif in_note:
                if xi_real_frames[m] < offset_threshold[m]:
                    below_threshold_count += 1
                    # 指定フレーム数連続で閾値以下なら音の終了と判断
                    if below_threshold_count >= self.consecutive_offset_frames:
                        in_note = False
                        # オフセット時刻はカウント開始フレームに設定
                        offset_frame = m - below_threshold_count + 1
                        if offset_frame >= 0 and offset_frame < len(frame_times):
                            offsets.append(frame_times[offset_frame] / sr)
                        else:
                            # 範囲外の場合は現在のフレームを使用
                            offsets.append(frame_times[m] / sr)
                        below_threshold_count = 0
                else:
                    # 閾値を上回ったらカウンタをリセット
                    below_threshold_count = 0
        
        # 最後のノートがオフセット未検出の場合
        if in_note and len(frame_times) > 0:
            offsets.append(frame_times[-1] / sr)
        
        # 配列に変換
        onsets = np.array(onsets)
        offsets = np.array(offsets)
        
        # ノート長が最小長より短いものを除外
        valid_notes = []
        if len(onsets) > 0 and len(offsets) > 0:
            for i in range(min(len(onsets), len(offsets))):
                if offsets[i] - onsets[i] >= self.min_note_length:
                    valid_notes.append(i)
            
            if valid_notes:
                onsets = onsets[valid_notes]
                offsets = offsets[valid_notes]
            else:
                onsets = np.array([])
                offsets = np.array([])
        
        self.logger.debug(f"検出結果: {len(onsets)}個のオンセット, {len(offsets)}個のオフセット")
        
        return onsets, offsets

    def _estimate_pitches(self, xi_imag_frames: np.ndarray, frame_energies: np.ndarray, 
                         np_indices: np.ndarray, frame_times: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        自己微分積の虚部からピッチを推定
        
        Parameters
        ----------
        xi_imag_frames : np.ndarray
            フレームごとの自己微分積虚部の合計
        frame_energies : np.ndarray
            フレームごとのエネルギー
        np_indices : np.ndarray
            フレームごとの非周期性指標
        frame_times : np.ndarray
            フレーム時刻（サンプルインデックス）
        sr : int
            サンプリングレート
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (ピッチ時刻の配列, ピッチ周波数の配列)
        """
        self.logger.debug("ピッチ推定開始")
        
        # フレーム数
        n_frames = len(xi_imag_frames)
        
        # 入力配列のサイズ確認
        min_size = min(len(xi_imag_frames), len(frame_energies), len(np_indices), len(frame_times))
        if min_size < n_frames:
            self.logger.warning(f"入力配列のサイズが不一致です: xi_imag_frames={len(xi_imag_frames)}, frame_energies={len(frame_energies)}, np_indices={len(np_indices)}, frame_times={len(frame_times)}")
            n_frames = min_size
        
        # ピッチ推定結果
        pitch_freqs = np.zeros(n_frames)
        
        # 各フレームのピッチを計算
        for m in range(n_frames):
            # インデックスのバリデーション
            if m >= len(frame_energies) or m >= len(np_indices) or m >= len(xi_imag_frames):
                self.logger.warning(f"インデックス {m} が範囲外です: frame_energies={len(frame_energies)}, np_indices={len(np_indices)}, xi_imag_frames={len(xi_imag_frames)}")
                continue
                
            # エネルギーが十分あり、かつ非周期性が閾値以下のフレームのみピッチを計算
            if frame_energies[m] > 0 and np_indices[m] <= self.nonperiodicity_threshold:
                # 自己微分積の虚部から角周波数を計算し、周波数[Hz]に変換
                angular_freq = xi_imag_frames[m] / frame_energies[m]
                pitch_freqs[m] = (sr / (2.0 * np.pi)) * np.abs(angular_freq)
            else:
                pitch_freqs[m] = 0.0  # 無音/非周期的なフレームは0Hz
        
        # 異常値除去（不自然に高いまたは低いピッチを除外、範囲を拡大）
        pitch_freqs[(pitch_freqs < 15.0) | (pitch_freqs > 8000.0)] = 0.0
        
        # 平滑化（可選、安定したピッチ推定のため）
        if np.sum(pitch_freqs > 0) > 0:
            # 有効なピッチ値のみを平滑化
            valid_mask = pitch_freqs > 0
            if np.sum(valid_mask) > 2:  # 十分なデータポイントがある場合
                valid_indices = np.where(valid_mask)[0]
                valid_freqs = pitch_freqs[valid_indices]
                # メディアンフィルタによる平滑化（窓サイズ拡大）
                window_size = min(7, len(valid_freqs))
                if window_size % 2 == 0:  # 窓サイズは奇数でなければならない
                    window_size += 1
                if window_size >= 3:  # 最小窓サイズ
                    smoothed_freqs = signal.medfilt(valid_freqs, kernel_size=window_size)
                    pitch_freqs[valid_indices] = smoothed_freqs
        
        # 結果の抽出（有効なピッチのみ）
        valid_indices = np.where(pitch_freqs > 0)[0]
        if len(valid_indices) > 0 and len(frame_times) > 0:
            # frame_timesのサイズを確認
            safe_indices = valid_indices[valid_indices < len(frame_times)]
            if len(safe_indices) > 0:
                valid_times = frame_times[safe_indices] / sr
                valid_freqs = pitch_freqs[safe_indices]
            else:
                self.logger.warning("有効なピッチポイントがないか、全てのインデックスがframe_timesの範囲外です")
                valid_times = np.array([])
                valid_freqs = np.array([])
        else:
            valid_times = np.array([])
            valid_freqs = np.array([])
        
        self.logger.debug(f"ピッチ推定完了: {len(valid_times)}個の有効なピッチポイント")
        
        return valid_times, valid_freqs

    def _get_avg_nonperiodicity(self, onset: float, offset: float) -> float:
        """
        指定されたオンセットとオフセット間の平均非周期性指標を取得
        
        Parameters
        ----------
        onset : float
            オンセット時刻（秒）
        offset : float
            オフセット時刻（秒）
            
        Returns
        -------
        float
            平均非周期性指標（0～1の範囲）
        """
        if self.np_indices is None or len(self.np_indices) == 0:
            return 0.0
        
        # フレーム時間の計算
        n_frames = len(self.np_indices)
        frame_times_sec = np.arange(n_frames) * self.hop_length / self.sr
        
        # 指定された時間範囲内のフレームを抽出
        indices = np.where((frame_times_sec >= onset) & (frame_times_sec <= offset))[0]
        
        if len(indices) == 0:
            return 0.0
        
        # 該当フレームの平均非周期性
        avg_np = np.mean(self.np_indices[indices])
        
        return float(avg_np)

    def detect(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        音声データからノートを検出する
        
        Parameters
        ----------
        audio_data : np.ndarray
            検出する音声データ
        sr : int
            サンプリングレート
            
        Returns
        -------
        Dict[str, Any]
            検出結果
        """
        # 検出時間の測定開始
        start_time = time.time()
        
        # オーディオデータのチェック
        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("無効な音声データです")
            # 空の結果を返す
            return {
                'intervals': np.array([]).reshape(0, 2),
                'note_pitches': np.array([]),
                'frame_times': np.array([]),
                'frame_frequencies': np.array([]),
                'detector_name': self.__class__.__name__,
                'detection_time': 0.0
            }
        
        # サンプル数が少なすぎる場合もエラー処理
        if len(audio_data) < 1024:
            self.logger.warning(f"音声データのサンプル数が少なすぎます: {len(audio_data)}サンプル")
            # 空の結果を返す
            return {
                'intervals': np.array([]).reshape(0, 2),
                'note_pitches': np.array([]),
                'frame_times': np.array([]),
                'frame_frequencies': np.array([]),
                'detector_name': self.__class__.__name__,
                'detection_time': 0.0
            }
        
        # 前処理: ステレオからモノラルへ変換
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        try:
            # パラメータと音声情報のログ出力
            self.logger.debug(f"音声データ: 長さ={len(audio_data)/sr:.2f}秒, サンプリングレート={sr}Hz, サンプル数={len(audio_data)}")
            self.logger.debug(f"検出パラメータ: bands={self.bands}, f_min={self.f_min}, hop_length={self.hop_length}, frame_length={self.frame_length}")
            
            # フィルタバンク設計（まだ設計されていない場合）
            if self.filter_coefs is None:
                self.logger.debug("フィルタバンクを設計します")
                self._design_filterbank(sr)
            
            self.logger.debug("複素共振フィルタを適用し、自己微分積を計算します")
            # 複素共振フィルタを適用し、自己微分積を計算
            xi_total, z_bands, r_bands = self._apply_complex_filters(audio_data, sr)
            
            self.logger.debug(f"z_bandsの長さ: {len(z_bands)}, 各バンドのサンプル数: {[len(z) for z in z_bands[:3]]}...")
            
            self.logger.debug("曲率を計算します")
            # 曲率の計算
            kappa_bands = self._calculate_curvature(z_bands, r_bands)
            
            self.logger.debug(f"kappa_bandsの長さ: {len(kappa_bands)}, 各バンドのサンプル数: {[len(k) for k in kappa_bands[:3]]}...")
            
            self.logger.debug("エントロピーと非周期性指標を計算します")
            # エントロピーと非周期性指標の計算
            np_indices, frame_energies = self._calculate_entropy(kappa_bands, r_bands)
            
            self.logger.debug(f"np_indicesの長さ: {len(np_indices)}, frame_energiesの長さ: {len(frame_energies)}")
            
            self.logger.debug("フレーム処理を行います")
            # フレーム処理
            xi_real_frames, xi_imag_frames, frame_times = self._process_frames(xi_total)
            
            self.logger.debug(f"フレーム処理結果: xi_real_framesの長さ: {len(xi_real_frames)}, xi_imag_framesの長さ: {len(xi_imag_frames)}, frame_timesの長さ: {len(frame_times)}")
            
            self.logger.debug("オンセット・オフセットを検出します")
            # オンセット・オフセット検出
            onsets, offsets = self._detect_onsets_offsets(xi_real_frames, frame_times, sr)
            
            self.logger.debug(f"検出されたオンセット数: {len(onsets)}, オフセット数: {len(offsets)}")
            
            # 検出結果を保存（他のメソッドでの再利用のため）
            self.xi_frames = (xi_real_frames, xi_imag_frames, frame_times)
            self.np_indices = np_indices
            self.frame_energies = frame_energies
            
            self.logger.debug("ピッチを推定します")
            # ピッチ推定
            pitch_times, pitch_freqs = self._estimate_pitches(
                xi_imag_frames, frame_energies, np_indices, frame_times, sr)
            
            self.logger.debug(f"推定されたピッチポイント数: {len(pitch_times)}")
            
            self.logger.debug(f"検出結果: オンセット数={len(onsets)}, オフセット数={len(offsets)}, ピッチポイント数={len(pitch_times)}")
            
            # ノート情報の構築
            intervals = []
            note_pitches = []
            
            # オンセットとオフセットの数が一致しない場合は調整
            if len(onsets) > 0 and len(offsets) > 0:
                if len(onsets) > len(offsets):
                    # オフセットが足りない場合、最後のオンセットに対して音声の終わりを使用
                    offsets = np.append(offsets, len(audio_data) / sr)
                    self.logger.debug(f"オフセットを追加して調整: 新しいオフセット数={len(offsets)}")
                elif len(onsets) < len(offsets):
                    # オンセットが足りない場合、余分なオフセットを削除
                    offsets = offsets[:len(onsets)]
                    self.logger.debug(f"オフセットを削減して調整: 新しいオフセット数={len(offsets)}")
                
                # 各ノートの処理
                for i, onset in enumerate(onsets):
                    offset = offsets[i]
                    
                    # ノート区間
                    intervals.append([onset, offset])
                    
                    # この区間内のピッチを抽出し、中央値を音符のピッチとする
                    mask = (pitch_times >= onset) & (pitch_times <= offset)
                    if np.any(mask):
                        freqs_in_interval = pitch_freqs[mask]
                        median_freq = np.median(freqs_in_interval)
                        note_pitches.append(median_freq)
                        
                        # 診断ログ
                        self.logger.debug(f"音符#{i+1}: 時間=[{onset:.3f}, {offset:.3f}], ピッチ={median_freq:.1f}Hz, データポイント数={np.sum(mask)}")
                    else:
                        # ピッチが見つからない場合は0とする
                        note_pitches.append(0.0)
                        self.logger.warning(f"音符#{i+1}: 時間=[{onset:.3f}, {offset:.3f}], 区間内にピッチデータがありません")
            
            # フレームベースのピッチ情報の作成
            if len(pitch_times) > 0:
                frame_times_sec = pitch_times  # 既に秒単位
                frame_frequencies = pitch_freqs
            else:
                # ピッチデータがない場合は空配列
                frame_times_sec = np.array([])
                frame_frequencies = np.array([])
            
            # 検出時間の測定終了
            detection_time = time.time() - start_time
            
            # 検出結果を返す（BaseDetectorの仕様に準拠）
            intervals_array = np.array(intervals)
            note_pitches_array = np.array(note_pitches)
            
            if len(intervals) == 0:
                # 空の結果の場合、正しい形状の空配列を返す
                intervals_array = np.array([]).reshape(0, 2)
                note_pitches_array = np.array([])
            
            return {
                'intervals': intervals_array,
                'note_pitches': note_pitches_array,
                'frame_times': np.array(frame_times_sec),
                'frame_frequencies': np.array(frame_frequencies),
                'detector_name': self.__class__.__name__,
                'detection_time': detection_time
            }
            
        except Exception as e:
            # エラーが発生した場合はログに出力し、空の結果を返す
            self.logger.error(f"検出中にエラーが発生しました: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            self.logger.error(f"トレースバック情報:\n{error_traceback}")
            
            return {
                'intervals': np.array([]).reshape(0, 2),
                'note_pitches': np.array([]),
                'frame_times': np.array([]),
                'frame_frequencies': np.array([]),
                'detector_name': self.__class__.__name__,
                'detection_time': time.time() - start_time
            }
