"""
RFS/PHDフィルタに基づく統合型動的アンチパターンピッチ検出アルゴリズム (ONDE-SEVEN)

Random Finite Set (RFS) 理論とProbability Hypothesis Density (PHD) フィルタを用いて、
背景からの逸脱として音楽イベントを検出し、同時に複数のピッチを推定します。

特徴:
- リアルタイム処理に最適化されたマルチターゲット追跡アルゴリズム
- 同時発音（ポリフォニック）に対応した複数ピッチ推定
- 確率的フレームワークによる高精度なオンセット/オフセット検出
- ガウス混合モデルによる効率的な状態表現と更新
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import librosa
import numpy as np

from src.detectors.base_detector import BaseDetector
from src.detectors.modules.filterbank.base_filterbank import FilterBank
from src.detectors.modules.filterbank.cqt_filterbank import CQTFilterBank
from src.detectors.modules.filterbank.stft_filterbank import STFTFilterBank


class GaussianComponent:
    """
    PHDフィルタで使用するガウス分布成分

    Attributes:
        mean (np.ndarray): 平均ベクトル
        covariance (np.ndarray): 共分散行列
        weight (float): 成分の重み
    """

    def __init__(self, mean: np.ndarray, covariance: np.ndarray, weight: float = 1.0):
        """
        ガウス成分を初期化

        Parameters
        ----------
        mean : np.ndarray
            平均ベクトル
        covariance : np.ndarray
            共分散行列
        weight : float, optional
            成分の重み, by default 1.0
        """
        self.mean = mean
        self.covariance = covariance
        self.weight = weight

    def copy(self):
        """
        成分のコピーを作成

        Returns
        -------
        GaussianComponent
            コピーされたガウス成分
        """
        return GaussianComponent(
            mean=self.mean.copy(), covariance=self.covariance.copy(), weight=self.weight
        )


class PeakDetector:
    """
    周波数スペクトルからピークを検出するクラス

    Attributes:
        min_f0 (float): 最小検出周波数 (Hz)
        max_f0 (float): 最大検出周波数 (Hz)
        sr (int): サンプリングレート (Hz)
        peak_threshold (float): ピーク検出閾値
    """

    def __init__(
        self, min_f0: float, max_f0: float, sr: int, peak_threshold: float = 0.01
    ):
        """
        ピーク検出器を初期化

        Parameters
        ----------
        min_f0 : float
            最小検出周波数 (Hz)
        max_f0 : float
            最大検出周波数 (Hz)
        sr : int
            サンプリングレート (Hz)
        peak_threshold : float, optional
            ピーク検出閾値, by default 0.01
        """
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.sr = sr
        self.peak_threshold = peak_threshold

    def detect_peaks(
        self, magnitude_spec: np.ndarray, freq_bins: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        振幅スペクトルからピークを検出

        Parameters
        ----------
        magnitude_spec : np.ndarray
            振幅スペクトル
        freq_bins : np.ndarray
            周波数ビン (Hz)

        Returns
        -------
        List[Dict[str, float]]
            検出されたピークのリスト [{'frequency': f, 'amplitude': a}, ...]
        """
        # 周波数範囲フィルタリング
        valid_freq_mask = (freq_bins >= self.min_f0) & (freq_bins <= self.max_f0)
        valid_indices = np.where(valid_freq_mask)[0]

        if len(valid_indices) == 0:
            return []

        # 有効範囲のスペクトルとビンを取得
        valid_spec = magnitude_spec[valid_indices]
        valid_freqs = freq_bins[valid_indices]

        # ローカルピークを見つける
        # 前後の値より大きい点を特定
        peak_mask = np.zeros_like(valid_spec, dtype=bool)
        peak_mask[1:-1] = (valid_spec[1:-1] > valid_spec[:-2]) & (
            valid_spec[1:-1] > valid_spec[2:]
        )

        # 振幅閾値以上のピークのみを選択
        peak_threshold = self.peak_threshold * np.max(valid_spec)
        peak_mask = peak_mask & (valid_spec > peak_threshold)

        peak_indices = np.where(peak_mask)[0]

        # ピークをリストとして整形
        peaks = []
        for idx in peak_indices:
            freq = valid_freqs[idx]
            amp = valid_spec[idx]
            peaks.append({"frequency": freq, "amplitude": amp})

        # 振幅の降順でソート
        peaks.sort(key=lambda x: x["amplitude"], reverse=True)

        return peaks


class OnlinePHDFilter:
    """
    オンラインPHDフィルタ実装。
    Random Finite Set（RFS）理論を用いて、時間経過とともに更新される音楽特性を追跡。

    Attributes:
        state_dim (int): 状態ベクトルの次元（例：2次元の場合、[周波数, 振幅]）
        meas_dim (int): 観測ベクトルの次元（例：2次元の場合、[周波数, 振幅]）
        p_survival (float): 状態の生存確率
        p_detection (float): 状態の検出確率
        clutter_intensity (float): クラッターの強度
        process_noise (np.ndarray): プロセスノイズの共分散行列
        meas_noise (np.ndarray): 観測ノイズの共分散行列
        pruning_threshold (float): 枝刈り閾値
        merging_threshold (float): マージング閾値
        max_components (int): 最大GMM成分数
        components (List[GaussianComponent]): ガウス混合モデルの成分
        birth_components (List[GaussianComponent]): 誕生ガウス混合モデルの成分
        init_birth_intensity (float): 誕生成分の初期強度
        frame_counter (int): 処理されたフレーム数
    """

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        p_survival: float,
        p_detection: float,
        clutter_intensity: float,
        process_noise: Optional[np.ndarray] = None,
        meas_noise: Optional[np.ndarray] = None,
        pruning_threshold: float = 1e-5,
        merging_threshold: float = 5.0,
        max_components: int = 50,
        init_birth_intensity: float = 0.1,
        min_f0: float = 50.0,
        max_f0: float = 1000.0,
        sr: int = 44100,
        enable_debug: bool = False,
    ):
        """
        オンラインPHDフィルタの初期化

        Parameters
        ----------
        state_dim : int
            状態ベクトルの次元
        meas_dim : int
            観測ベクトルの次元
        p_survival : float
            状態の生存確率
        p_detection : float
            状態の検出確率
        clutter_intensity : float
            クラッターの強度
        process_noise : np.ndarray, optional
            プロセスノイズの共分散行列
        meas_noise : np.ndarray, optional
            観測ノイズの共分散行列
        pruning_threshold : float
            枝刈り閾値
        merging_threshold : float
            マージング閾値
        max_components : int
            最大GMM成分数
        init_birth_intensity : float
            誕生成分の初期強度
        min_f0 : float
            最小周波数 (Hz)
        max_f0 : float
            最大周波数 (Hz)
        sr : int
            サンプリングレート (Hz)
        enable_debug : bool
            デバッグモードを有効にするかどうか
        """
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.p_survival = p_survival
        self.p_detection = p_detection
        self.clutter_intensity = clutter_intensity
        self.pruning_threshold = pruning_threshold
        self.merging_threshold = merging_threshold
        self.max_components = max_components

        # ノイズ共分散行列の初期化（デフォルト値の設定）
        if process_noise is None:
            # 状態遷移ノイズ（プロセスノイズ）
            self.process_noise = np.eye(state_dim) * 1.0
            self.process_noise[0, 0] = 1.0  # 周波数成分のノイズ
            self.process_noise[1, 1] = 0.1  # 振幅成分のノイズ
        else:
            self.process_noise = process_noise

        if meas_noise is None:
            # 観測ノイズ
            self.meas_noise = np.eye(meas_dim) * 0.01
            self.meas_noise[0, 0] = 1.0  # 周波数観測ノイズ
            self.meas_noise[1, 1] = 0.1  # 振幅観測ノイズ
        else:
            self.meas_noise = meas_noise

        # ガウス成分リストの初期化
        self.components = []

        # 誕生成分パラメータ
        self.init_birth_intensity = init_birth_intensity
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.sr = sr

        # 誕生成分リストを初期化
        self.birth_components = []
        self.frame_counter = 0

        # デバッグモード
        self.enable_debug = enable_debug
        if self.enable_debug:
            logging.debug(
                f"OnlinePHDFilter: 初期化完了 - 状態次元={state_dim}, 観測次元={meas_dim}"
            )

    def predict(self):
        """
        予測ステップ: すべての成分を時間的に前進させる
        """
        # フレームカウンタの更新
        self.frame_counter += 1

        if self.enable_debug:
            logging.debug(
                f"フレーム {self.frame_counter} - 予測ステップ開始: 成分数={len(self.components)}"
            )

        # 生存成分の予測
        predicted_components = []
        for comp in self.components:
            # 生存確率による重みの更新
            new_weight = comp.weight * self.p_survival

            # 新しい共分散行列の計算（プロセスノイズを加える）
            new_covar = comp.covariance + self.process_noise

            # 予測成分を追加
            predicted_comp = GaussianComponent(
                mean=comp.mean.copy(),  # 状態遷移は恒等関数（音楽のピッチは短時間では一定と仮定）
                covariance=new_covar,
                weight=new_weight,
            )
            predicted_components.append(predicted_comp)

        # 予測された成分でリストを更新
        self.components = predicted_components

        if self.enable_debug:
            logging.debug(
                f"フレーム {self.frame_counter} - 予測ステップ完了: 成分数={len(self.components)}"
            )

    def add_birth_components(self, observations):
        """
        観測から誕生成分を生成

        Parameters
        ----------
        observations : List[Dict]
            観測リスト
        """
        if not observations:
            return

        if self.enable_debug:
            logging.debug(
                f"フレーム {self.frame_counter} - 誕生成分追加: 観測数={len(observations)}"
            )

        # 観測から誕生成分を生成
        for obs in observations:
            # 観測値から状態ベクトルを作成
            if isinstance(obs, dict):
                # 辞書形式の観測
                obs_vector = np.array([obs["frequency"], obs["amplitude"]])
            else:
                # 配列形式の観測
                obs_vector = np.array(obs)

            # 誕生成分の共分散行列（初期不確実性）
            birth_covar = np.eye(self.state_dim) * 10.0
            birth_covar[0, 0] = 5.0  # 周波数の不確実性
            birth_covar[1, 1] = 0.5  # 振幅の不確実性

            # 誕生成分を追加
            birth_comp = GaussianComponent(
                mean=obs_vector,
                covariance=birth_covar,
                weight=self.init_birth_intensity,
            )

            self.components.append(birth_comp)

        if self.enable_debug:
            logging.debug(
                f"フレーム {self.frame_counter} - 誕生成分追加完了: 成分総数={len(self.components)}"
            )

    def update(self, observations):
        """
        観測による更新ステップを実行

        Parameters
        ----------
        observations : List[Dict]
            観測リスト
        """
        if not self.components:
            return

        if self.enable_debug:
            logging.debug(
                f"フレーム {self.frame_counter} - 更新ステップ開始: 成分数={len(self.components)}, 観測数={len(observations) if observations else 0}"
            )

        # 観測がない場合は重み更新のみ行う
        if not observations:
            for comp in self.components:
                comp.weight *= 1.0 - self.p_detection
            return

        # 更新後の成分リスト
        updated_components = []

        # 未検出成分（観測されなかった成分）を処理
        for comp in self.components:
            # 未検出成分の重みを更新
            missed_comp = GaussianComponent(
                mean=comp.mean.copy(),
                covariance=comp.covariance.copy(),
                weight=comp.weight * (1.0 - self.p_detection),
            )
            updated_components.append(missed_comp)

        # 各観測に対して全成分を更新
        for z in observations:
            # 観測を適切な形式に変換
            if isinstance(z, dict):
                # 辞書形式の観測
                z_vector = np.array([z["frequency"], z["amplitude"]])
            else:
                # 配列形式の観測
                z_vector = np.array(z)

            # 各成分の尤度と更新成分を計算
            likelihoods = []
            updated_comp_list = []

            for comp in self.components:
                # カルマンゲイン計算
                K = comp.covariance.dot(
                    np.linalg.inv(comp.covariance + self.meas_noise)
                )

                # 新しい平均と共分散
                new_mean = comp.mean + K.dot(z_vector - comp.mean)
                new_covar = comp.covariance - K.dot(comp.covariance)

                # マハラノビス距離に基づく尤度計算
                innovation = z_vector - comp.mean
                innov_cov = comp.covariance + self.meas_noise

                # マハラノビス距離の計算
                try:
                    innov_cov_inv = np.linalg.inv(innov_cov)
                    maha_dist = innovation.T.dot(innov_cov_inv).dot(innovation)
                    likelihood = np.exp(-0.5 * maha_dist) / np.sqrt(
                        np.linalg.det(2 * np.pi * innov_cov)
                    )
                except np.linalg.LinAlgError:
                    # 共分散行列が特異点に近い場合
                    likelihood = 0.0

                # 尤度が非常に小さい場合は0に
                if likelihood < 1e-10:
                    likelihood = 0.0

                likelihoods.append(likelihood)

                # 更新された成分
                updated_comp = GaussianComponent(
                    mean=new_mean,
                    covariance=new_covar,
                    weight=comp.weight * self.p_detection * likelihood,
                )
                updated_comp_list.append(updated_comp)

            # 尤度の総和（分母）とクラッター項
            sum_likelihood = sum(
                comp.weight * like for comp, like in zip(self.components, likelihoods)
            )
            denom = self.clutter_intensity + sum_likelihood

            # 分母が非常に小さい場合はスキップ
            if denom < 1e-10:
                continue

            # 全ての更新成分を追加
            for comp in updated_comp_list:
                comp.weight /= denom
                updated_components.append(comp)

        # 成分リストを更新
        self.components = updated_components

        # 枝刈りとマージング
        self.prune_and_merge()

        if self.enable_debug:
            logging.debug(
                f"フレーム {self.frame_counter} - 更新ステップ完了: 成分数={len(self.components)}"
            )

    def prune_and_merge(self):
        """
        枝刈りとマージングを実行して成分数を制限
        """
        if not self.components:
            return

        if self.enable_debug:
            logging.debug(
                f"フレーム {self.frame_counter} - 枝刈り&マージング開始: 成分数={len(self.components)}"
            )

        # 1. 閾値以下の成分を枝刈り
        pruned_components = [
            comp for comp in self.components if comp.weight > self.pruning_threshold
        ]

        if not pruned_components:
            self.components = []
            return

        # 成分を重み降順にソート
        pruned_components.sort(key=lambda x: x.weight, reverse=True)

        # 2. 類似成分のマージング
        merged_components = []
        unmerged_indices = list(range(len(pruned_components)))

        while unmerged_indices and len(merged_components) < self.max_components:
            # 最大重みの成分を選択
            max_weight_idx = max(
                unmerged_indices, key=lambda i: pruned_components[i].weight
            )
            merged_comp = pruned_components[max_weight_idx]
            unmerged_indices.remove(max_weight_idx)

            # マージする成分のインデックスを取得
            merge_indices = []
            for i in unmerged_indices[:]:
                # マハラノビス距離を計算
                comp = pruned_components[i]
                innovation = merged_comp.mean - comp.mean
                innov_cov = merged_comp.covariance

                try:
                    maha_dist = innovation.T.dot(np.linalg.inv(innov_cov)).dot(
                        innovation
                    )
                except np.linalg.LinAlgError:
                    # 共分散行列が特異行列の場合、ユークリッド距離を使用
                    maha_dist = np.sqrt(np.sum((merged_comp.mean - comp.mean) ** 2))

                # 閾値以下の場合はマージ
                if maha_dist <= self.merging_threshold:
                    merge_indices.append(i)
                    unmerged_indices.remove(i)

            # マージする成分がある場合
            if merge_indices:
                # マージする全成分を収集
                merge_comps = [pruned_components[max_weight_idx]]
                merge_comps.extend([pruned_components[i] for i in merge_indices])

                # 総重み
                total_weight = sum(comp.weight for comp in merge_comps)

                if total_weight > 0:
                    # 重み付き平均を計算
                    new_mean = np.zeros_like(merged_comp.mean)
                    for comp in merge_comps:
                        new_mean += comp.weight * comp.mean
                    new_mean /= total_weight

                    # 重み付き共分散を計算
                    new_covar = np.zeros_like(merged_comp.covariance)
                    for comp in merge_comps:
                        diff = comp.mean - new_mean
                        new_covar += comp.weight * (
                            comp.covariance + np.outer(diff, diff)
                        )
                    new_covar /= total_weight

                    # 新しいマージ成分
                    merged_comp = GaussianComponent(
                        mean=new_mean, covariance=new_covar, weight=total_weight
                    )

            merged_components.append(merged_comp)

        # 3. 最大成分数を超える場合は重み順に切り捨て
        if len(merged_components) > self.max_components:
            merged_components.sort(key=lambda x: x.weight, reverse=True)
            merged_components = merged_components[: self.max_components]

        # 成分リストを更新
        self.components = merged_components

        if self.enable_debug:
            logging.debug(
                f"フレーム {self.frame_counter} - 枝刈り&マージング完了: 成分数={len(self.components)}"
            )

    def extract_states(self, threshold=0.5):
        """
        PHDから推定ノート数と各ノートの状態を抽出

        Parameters
        ----------
        threshold : float
            状態抽出閾値

        Returns
        -------
        Tuple[List[np.ndarray], List[float], List[bool]]
            (推定状態ベクトルのリスト, 重みのリスト, オンセットフラグのリスト)
        """
        if not self.components:
            return [], [], []

        if self.enable_debug:
            logging.debug(
                f"フレーム {self.frame_counter} - 状態抽出開始: 成分数={len(self.components)}, 閾値={threshold}"
            )

        # 重み閾値以上の成分を抽出
        extracted_states = []
        extracted_weights = []
        onset_flags = []

        for comp in self.components:
            if comp.weight > threshold:
                extracted_states.append(comp.mean)
                extracted_weights.append(comp.weight)

                # オンセットフラグ（前回見えなかったクラスタなら True）
                # 実際の実装では、過去の状態履歴と比較して判断する必要がある
                onset = False

                # ここでは簡易的にフィルタが反応してから3フレーム以内をオンセットとする
                if comp.weight > 0.8 and self.frame_counter <= 3:
                    onset = True

                onset_flags.append(onset)

        if self.enable_debug:
            logging.debug(
                f"フレーム {self.frame_counter} - 状態抽出完了: 抽出状態数={len(extracted_states)}"
            )

        return extracted_states, extracted_weights, onset_flags


class ONDESEVENDetector(BaseDetector):
    """
    RFS/PHDフィルタに基づく統合型動的アンチパターンピッチ検出アルゴリズム (ONDE-SEVEN)

    Random Finite Set理論とPHDフィルタを用いて、背景からの逸脱として音楽イベントを検出し、
    同時に複数のピッチを推定します。

    Attributes:
        p_survival (float):         生存確率(/フレーム)
        p_detection (float):        検出確率(/フレーム)
        clutter_intensity (float):  クラッターピーク強度
        sr (int):                   サンプリングレート
        hop_length (int):           ホップ長
        buffer_size (int):          処理バッファサイズ
        fft_size (int):             FFTサイズ
        min_f0 (float):             最小周波数 (Hz)
        max_f0 (float):             最大周波数 (Hz)

        # 内部状態
        phd_filter (OnlinePHDFilter): PHDフィルタ
        peak_detector (PeakDetector): ピーク検出器

        # フラグとヒストリー管理
        prev_active_pitches (Set[float]): 前フレームのアクティブピッチ
        pitch_history (List[List[float]]): 各フレームのピッチ履歴
        amplitude_history (List[List[float]]): 各フレームの振幅履歴
        onset_frames (List[int]): オンセットフレームのインデックス
        offset_frames (List[int]): オフセットフレームのインデックス
    """

    def __init__(
        self,
        p_survival: float = 0.99,  # 生存確率
        p_detection: float = 0.98,  # 検出確率
        clutter_intensity: float = 0.1,  # クラッター強度
        birth_intensity: float = 0.1,  # 誕生成分強度
        pruning_threshold: float = 1e-5,  # 枝刈り閾値
        merging_threshold: float = 5.0,  # マージ閾値
        max_components: int = 50,  # 最大成分数
        min_f0: float = 50.0,  # 最小周波数 (Hz)
        max_f0: float = 1000.0,  # 最大周波数 (Hz)
        sr: int = 44100,  # サンプリングレート
        buffer_size: int = 1024,  # 処理バッファサイズ
        fft_size: int = 2048,  # FFTサイズ
        hop_length: int = 512,  # ホップ長
        extraction_threshold: float = 0.5,  # 状態抽出閾値
        enable_debug: bool = False,
    ):
        """
        ONDE-SEVEN検出器の初期化

        Parameters
        ----------
        p_survival : float
            生存確率 (デフォルト=0.99)
        p_detection : float
            検出確率 (デフォルト=0.98)
        clutter_intensity : float
            クラッター強度 (デフォルト=0.1)
        birth_intensity : float
            誕生成分強度 (デフォルト=0.1)
        pruning_threshold : float
            枝刈り閾値 (デフォルト=1e-5)
        merging_threshold : float
            マージ閾値 (デフォルト=5.0)
        max_components : int
            最大成分数 (デフォルト=50)
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
        extraction_threshold : float
            状態抽出閾値 (デフォルト=0.5)
        enable_debug : bool
            デバッグモードを有効にするかどうか (デフォルト=False)
        """
        # BaseDetectorの初期化
        super().__init__()

        # デバッグモードの設定
        self.enable_debug = enable_debug
        if self.enable_debug:
            # デバッグモードが有効な場合、ロギングレベルをDEBUGに設定
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            logging.debug("ONDESEVENDetector: デバッグモードが有効です")

        # 基本パラメータ
        self.p_survival = p_survival
        self.p_detection = p_detection
        self.clutter_intensity = clutter_intensity
        self.birth_intensity = birth_intensity
        self.pruning_threshold = pruning_threshold
        self.merging_threshold = merging_threshold
        self.max_components = max_components
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.sr = sr
        self.buffer_size = buffer_size
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.extraction_threshold = extraction_threshold

        # PHDフィルタの初期化
        self.phd_filter = OnlinePHDFilter(
            state_dim=2,  # [f, a] (周波数, 振幅)
            meas_dim=2,  # [f, a] (周波数, 振幅)
            p_survival=self.p_survival,
            p_detection=self.p_detection,
            clutter_intensity=self.clutter_intensity,
            pruning_threshold=self.pruning_threshold,
            merging_threshold=self.merging_threshold,
            max_components=self.max_components,
            init_birth_intensity=self.birth_intensity,
            min_f0=self.min_f0,
            max_f0=self.max_f0,
            sr=self.sr,
            enable_debug=self.enable_debug,
        )

        # ピーク検出器の初期化
        self.peak_detector = PeakDetector(
            min_f0=self.min_f0, max_f0=self.max_f0, sr=self.sr
        )

        # 履歴とフラグの初期化
        self.prev_active_pitches = set()
        self.pitch_history = []
        self.amplitude_history = []
        self.onset_frames = []
        self.offset_frames = []

        if self.enable_debug:
            logging.debug(f"ONDESEVENDetector: 初期化完了")

    def process_frame(self, audio_buffer):
        """
        単一フレームを処理し、観測、PHD更新、状態抽出を行う

        Parameters
        ----------
        audio_buffer : np.ndarray
            オーディオバッファ (時間領域サンプル)

        Returns
        -------
        Dict
            ピッチ、振幅、重み等を含む辞書
        """
        # スペクトログラムを計算
        spec_frame, freq_bins = self._compute_spectrogram(audio_buffer)

        # スペクトルピークを検出（観測生成）
        peaks = self.peak_detector.detect_peaks(spec_frame, freq_bins)

        # 1. PHD予測ステップ
        self.phd_filter.predict()

        # 2. 誕生成分の追加（オンセット候補）
        self.phd_filter.add_birth_components(peaks)

        # 3. PHD更新ステップ
        self.phd_filter.update(peaks)

        # 4. 状態抽出
        states, weights, onset_flags = self.phd_filter.extract_states(
            threshold=self.extraction_threshold
        )

        # 5. 結果を保存
        active_pitches = set()
        pitches = []
        amplitudes = []

        for state in states:
            if state[0] >= self.min_f0 and state[0] <= self.max_f0:
                pitches.append(state[0])
                amplitudes.append(state[1])
                active_pitches.add(state[0])

        # オンセット/オフセットの確認
        is_onset = False
        is_offset = False

        # 前フレームにない音が出現した場合はオンセット
        new_pitches = active_pitches - self.prev_active_pitches
        if new_pitches:
            is_onset = True
            if self.enable_debug:
                logging.debug(f"オンセット検出: 新規ピッチ {new_pitches}")

        # 前フレームにあった音が消失した場合はオフセット
        disappeared_pitches = self.prev_active_pitches - active_pitches
        if disappeared_pitches:
            is_offset = True
            if self.enable_debug:
                logging.debug(f"オフセット検出: 消失ピッチ {disappeared_pitches}")

        # 前フレームのアクティブピッチを更新
        self.prev_active_pitches = active_pitches

        return {
            "pitches": pitches,
            "amplitudes": amplitudes,
            "weights": weights,
            "onset_flags": onset_flags,
            "is_onset": is_onset,
            "is_offset": is_offset,
            "active_pitches": active_pitches,
        }

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
        magnitude_spec = magnitude_spec[: self.fft_size // 2 + 1]

        # 周波数ビンを計算
        freq_bins = np.fft.rfftfreq(self.fft_size, 1.0 / self.sr)

        return magnitude_spec, freq_bins

    def detect(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        音声データからノートを検出し、検出結果を返します。

        Parameters
        ----------
        audio_data : np.ndarray
            音声データ
        sr : int
            サンプリングレート

        Returns
        -------
        Dict[str, Any]
            検出結果の辞書。以下のキーを含む必要があります：
            {
                'intervals': np.ndarray,  # 形状 (N, 2), 各行は [onset, offset] (秒)
                'note_pitches': np.ndarray,  # 形状 (N,), 各要素はHz単位のピッチ
                'frame_times': np.ndarray,  # 形状 (M,), フレーム時刻 (秒)
                'frame_frequencies': np.ndarray,  # 形状 (M,), フレーム周波数 (Hz)
                'detector_name': str,  # 検出器名
                'detection_time': float  # 検出処理時間 (秒)
            }
        """
        # 処理開始時間を記録
        start_time = time.time()

        # 設定変更が必要な場合
        if sr != self.sr:
            self.sr = sr
            # 必要に応じて他のパラメータも更新
            self.peak_detector = PeakDetector(
                min_f0=self.min_f0, max_f0=self.max_f0, sr=self.sr
            )

        # 履歴をリセット
        self.pitch_history = []
        self.amplitude_history = []
        self.onset_frames = []
        self.offset_frames = []
        self.prev_active_pitches = set()

        # フレーム分割してバッファごとに処理
        n_frames = 1 + (len(audio_data) - self.buffer_size) // self.hop_length

        # 結果格納用の配列
        frame_times = np.zeros(n_frames)
        frame_pitches = [[] for _ in range(n_frames)]
        frame_amplitudes = [[] for _ in range(n_frames)]

        # フレームごとに処理
        for i in range(n_frames):
            # フレーム時間
            frame_time = i * self.hop_length / sr
            frame_times[i] = frame_time

            # バッファの開始位置
            start = i * self.hop_length
            end = min(start + self.buffer_size, len(audio_data))

            # バッファサイズが小さすぎる場合はスキップ
            if end - start < self.buffer_size // 2:
                continue

            # バッファを取得
            buffer = audio_data[start:end]
            if len(buffer) < self.buffer_size:
                # 足りない部分はゼロパディング
                buffer = np.pad(buffer, (0, self.buffer_size - len(buffer)))

            # フレーム処理
            result = self.process_frame(buffer)

            # 結果を保存
            pitches = result["pitches"]
            amplitudes = result["amplitudes"]
            is_onset = result["is_onset"]
            is_offset = result["is_offset"]

            frame_pitches[i] = pitches
            frame_amplitudes[i] = amplitudes

            # オンセット/オフセットフレームを記録
            if is_onset:
                self.onset_frames.append(i)
            if is_offset:
                self.offset_frames.append(i)

            # 履歴に追加
            self.pitch_history.append(pitches)
            self.amplitude_history.append(amplitudes)

        # シーケンスの最後に進行中のノートがある場合はクローズ
        if self.prev_active_pitches:
            self.offset_frames.append(n_frames - 1)

        # フレーム位置を時間に変換
        onset_frames = np.array(self.onset_frames, dtype=int)
        offset_frames = np.array(self.offset_frames, dtype=int)

        # オンセット/オフセットの整合性チェック（オンセットとオフセットの数を整合）
        if len(onset_frames) > len(offset_frames):
            # 残りのオンセットにオフセットを追加
            offset_frames = np.append(offset_frames, n_frames - 1)
        elif len(onset_frames) < len(offset_frames):
            # 最初のオンセットの前のオフセットを削除
            offset_frames = offset_frames[-len(onset_frames) :]

        # 時間に変換
        onset_times = frame_times[onset_frames]
        offset_times = frame_times[offset_frames]

        # ノートごとのピッチを計算
        note_pitches = np.zeros(len(onset_frames))

        # フレームピッチをフラット化
        all_frame_pitches = []
        for i in range(n_frames):
            if frame_pitches[i]:
                # 複数ピッチの場合は最大音量のピッチを選択
                if len(frame_pitches[i]) > 1:
                    max_amp_idx = np.argmax(frame_amplitudes[i])
                    all_frame_pitches.append(frame_pitches[i][max_amp_idx])
                else:
                    all_frame_pitches.append(frame_pitches[i][0])
            else:
                all_frame_pitches.append(0.0)  # 無音フレーム

        all_frame_pitches = np.array(all_frame_pitches)

        # ノートごとのピッチを計算
        for i, (onset, offset) in enumerate(zip(onset_frames, offset_frames)):
            if onset < offset:
                # ノート範囲のピッチを取得
                note_range_pitches = []
                for j in range(onset, offset + 1):
                    if j < len(frame_pitches) and frame_pitches[j]:
                        note_range_pitches.extend(frame_pitches[j])

                # 有効なピッチのみを考慮
                note_range_pitches = [p for p in note_range_pitches if p > 0]

                if note_range_pitches:
                    # 最頻値や中央値などでピッチを決定
                    note_pitches[i] = np.median(note_range_pitches)
                else:
                    note_pitches[i] = 0.0  # 有効なピッチがない場合

        # 区間を作成
        intervals = np.column_stack((onset_times, offset_times))

        # 検出時間を計算
        detection_time = time.time() - start_time

        # 結果を返す
        return {
            "onset_times": onset_times,
            "offset_times": offset_times,
            "onset_frames": onset_frames,
            "offset_frames": offset_frames,
            "intervals": intervals,
            "note_pitches": note_pitches,
            "frame_times": frame_times,
            "frame_frequencies": all_frame_pitches,
            "detector_name": self.__class__.__name__,
            "detection_time": detection_time,
        }

    def detect_notes(
        self,
        audio: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        audio_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        オーディオから音符を検出する。オーディオデータかオーディオパスのいずれかを指定。

        Args:
            audio (np.ndarray, optional): オーディオデータ。Noneならaudio_pathを使用。
            sr (int, optional): サンプリングレート。Noneならself.srを使用。
            audio_path (str, optional): オーディオファイルパス。audioがNoneの場合に使用。

        Returns:
            Dict[str, np.ndarray]: 検出結果（intervals, pitches）
        """
        # オーディオロード
        if audio is None and audio_path is not None:
            audio, sr = librosa.load(audio_path, sr=sr if sr is not None else self.sr)
        elif audio is None and audio_path is None:
            raise ValueError(
                "オーディオデータかオーディオパスのいずれかを指定してください"
            )

        if sr is None:
            sr = self.sr

        # 検出実行
        results = self.detect(audio, sr)

        # 必要な結果だけを返す
        return {
            "intervals": results["intervals"],
            "note_pitches": results["note_pitches"],
        }

    def detect_onsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        オンセットのみを検出する簡略版メソッド

        Parameters
        ----------
        audio : np.ndarray
            音声データ
        sr : int
            サンプリングレート

        Returns
        -------
        np.ndarray
            オンセット時間配列
        """
        results = self.detect(audio, sr)
        return results["onset_times"]

    def detect_offsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        オフセットのみを検出する簡略版メソッド

        Parameters
        ----------
        audio : np.ndarray
            音声データ
        sr : int
            サンプリングレート

        Returns
        -------
        np.ndarray
            オフセット時間配列
        """
        results = self.detect(audio, sr)
        return results["offset_times"]
