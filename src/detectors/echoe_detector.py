"""
エンベロープ調和・カオス埋没理論に基づく統合ノート検出アルゴリズム (ECHOE)

このモジュールはECHOE（Envelope CHaOs-Embedding Unified Note Detection Algorithm）を
実装した検出器を提供します。従来の基本周波数(f0)推定に頼らず、振幅エンベロープの
抽象指標（協和度・カオス度・急変度）を用いて音符の検出を行います。
"""

import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
from typing import List, Dict, Tuple, Any, Optional, Union
from .base_detector import BaseDetector
import librosa


class ECHOEDetector(BaseDetector):
    """
    エンベロープ調和・カオス埋没理論に基づく統合ノート検出アルゴリズム (ECHOE)
    
    このディテクターは振幅エンベロープの抽象指標（協和度・カオス度・急変度）を用いて
    「安定(調和)」「カオス」「非調和(急変)」の3状態を評価し、音符の開始と終了を検出します。
    
    オンセットは「急激な変化」、オフセットは「カオス状態への埋没」として検出されます。
    """
    
    def __init__(self, **kwargs):
        """
        ECHOEディテクターの初期化
        
        Parameters
        ----------
        frame_length : int, optional
            分析フレームの長さ（サンプル数）、デフォルトは1024
        hop_length : int, optional
            フレーム間のホップ長（サンプル数）、デフォルトは512（50%オーバーラップ）
        look_ahead : int, optional
            先読みバッファのサイズ（フレーム数）、デフォルトは5
        history_size : int, optional
            過去バッファのサイズ（フレーム数）、デフォルトは20
        hmm_trans_matrix : np.ndarray, optional
            HMMの遷移確率行列、デフォルトはNone（自動生成）
        stability_threshold : float, optional
            安定状態(S)判定の閾値、デフォルトは0.6
        chaos_threshold : float, optional
            カオス状態(C)判定の閾値、デフォルトは0.4
        nonstable_threshold : float, optional
            非調和状態(N)判定の閾値、デフォルトは0.5
        onset_threshold : float, optional
            オンセット検出の閾値、デフォルトは0.5
        offset_min_frames : int, optional
            オフセット確定の最小フレーム数、デフォルトは3
        offset_chaos_threshold : float, optional
            オフセット時のカオス優勢度閾値、デフォルトは0.6
        offset_stability_diff : float, optional
            オフセット判定時の安定・非調和状態の差分閾値、デフォルトは0.1
        pitch_band_count : int, optional
            ピッチ推定用の周波数帯域数、デフォルトは48
        pitch_min_freq : float, optional
            ピッチ推定の最低周波数、デフォルトは50.0 Hz
        pitch_max_freq : float, optional
            ピッチ推定の最高周波数、デフォルトは2000.0 Hz
        enable_pitch_estimation : bool, optional
            ピッチ推定を有効にするかどうか、デフォルトはTrue
        use_viterbi : bool, optional
            Viterbiアルゴリズムを使用するかどうか、デフォルトはTrue
        use_gaussian_obs : bool, optional
            ガウシアンモデルによる観測確率計算を使用するかどうか、デフォルトはFalse
        """
        super().__init__(**kwargs)
        
        # 基本パラメータ
        self.frame_length = kwargs.get('frame_length', 1024)
        self.hop_length = kwargs.get('hop_length', 512)  # 50% オーバーラップが推奨値
        self.look_ahead = kwargs.get('look_ahead', 5)    # 5~10フレーム推奨
        self.history_size = kwargs.get('history_size', 20)
        
        # 状態判定パラメータ
        self.stability_threshold = kwargs.get('stability_threshold', 0.6)
        self.chaos_threshold = kwargs.get('chaos_threshold', 0.4)
        self.nonstable_threshold = kwargs.get('nonstable_threshold', 0.5)
        
        # オンセット/オフセット検出パラメータ
        self.onset_threshold = kwargs.get('onset_threshold', 0.5)
        self.offset_min_frames = kwargs.get('offset_min_frames', 3)
        self.offset_chaos_threshold = kwargs.get('offset_chaos_threshold', 0.6)
        self.offset_stability_diff = kwargs.get('offset_stability_diff', 0.1)
        
        # ピッチ推定パラメータ
        self.enable_pitch_estimation = kwargs.get('enable_pitch_estimation', True)
        self.pitch_band_count = kwargs.get('pitch_band_count', 48)
        self.pitch_min_freq = kwargs.get('pitch_min_freq', 50.0)
        self.pitch_max_freq = kwargs.get('pitch_max_freq', 2000.0)
        
        # Viterbiアルゴリズムを使用するかどうか
        self.use_viterbi = kwargs.get('use_viterbi', True)
        
        # ガウシアンモデルを使用するかどうか
        self.use_gaussian_obs = kwargs.get('use_gaussian_obs', False)
        
        # HMMの遷移確率行列（未指定の場合はデフォルト値を設定）
        self.hmm_trans_matrix = kwargs.get('hmm_trans_matrix', None)
        if self.hmm_trans_matrix is None:
            # S: 安定(調和), C: カオス, N: 非調和(急変)
            self.hmm_trans_matrix = np.array([
                [0.8, 0.15, 0.05],  # S -> S, C, N
                [0.2, 0.6, 0.2],    # C -> S, C, N
                [0.05, 0.15, 0.8]   # N -> S, C, N
            ])
        
        # バッファの初期化
        self._reset_buffers()

    def _reset_buffers(self):
        """内部バッファをリセットします"""
        self.feature_buffer = None  # 特徴量バッファ (C, K, D)
        self.state_buffer = None    # 状態バッファ (S, C, N)
        self.hmm_posterior = None   # HMM事後確率
    
    def detect(self, audio_file: Optional[str] = None, audio: Optional[np.ndarray] = None, sr: Optional[int] = None) -> Dict[str, Any]:
        """
        音声ファイルまたは音声データからノート情報を検出します
        
        Parameters
        ----------
        audio_file : str, optional
            分析する音声ファイルのパス
        audio : np.ndarray, optional
            分析する音声データ（audio_fileが指定されていない場合に使用）
        sr : int, optional
            サンプリングレート（audio指定時に必要）
            
        Returns
        -------
        Dict[str, Any]
            検出結果（オンセット、オフセット、ピッチ、音符情報を含む辞書）
        """
        # オーディオファイルが指定されている場合はロード
        if audio_file is not None:
            audio, sr = librosa.load(audio_file, sr=None)
        
        # オーディオデータのチェック
        if audio is None:
            return self._ensure_results()
            
        # バッファをリセット
        self._reset_buffers()
        
        # 特徴量計算と状態推定
        self._compute_features(audio, sr)
        
        # 状態推定にViterbiを使用するかどうかを分岐
        if self.use_viterbi:
            self._estimate_states_viterbi()
        else:
            self._estimate_states()
        
        # 音符情報の検出
        if self.enable_pitch_estimation:
            notes = self.detect_notes(audio, sr)
        else:
            # ピッチ推定なしでオンセット/オフセットのみ検出
            onsets = self.detect_onsets(audio, sr)
            offsets = self.detect_offsets(audio, sr)
            
            # ピッチ情報なしのノート情報を作成
            notes = {
                'onset_times': onsets,
                'offset_times': offsets,
                'pitches': np.zeros_like(onsets)  # ダミーのピッチ情報
            }
        
        # 結果を辞書形式で返す
        results = {
            'onsets': notes['onset_times'],
            'offsets': notes['offset_times'],
            'pitches': notes['pitches'],
            'note_intervals': np.column_stack((notes['onset_times'], notes['offset_times'])) if len(notes['onset_times']) > 0 else np.array([])
        }
        
        return results
    
    def detect_onsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        音声データからオンセット（音の開始時点）を検出します
        
        Parameters
        ----------
        audio : np.ndarray
            分析する音声データ
        sr : int
            サンプリングレート
            
        Returns
        -------
        np.ndarray
            検出されたオンセット時間（秒）の配列
        """
        # 特徴量の計算
        if self.feature_buffer is None:
            self._compute_features(audio, sr)
        
        # 状態推定
        if self.state_buffer is None:
            if self.use_viterbi:
                self._estimate_states_viterbi()
            else:
                self._estimate_states()
        
        # オンセット検出のためのパラメータ
        min_frames_between_onsets = int(0.05 * sr / self.hop_length)  # 最小50msのギャップ
        
        # オンセット検出（理論ドキュメントに準拠した条件）
        onset_frames = []
        last_onset = -min_frames_between_onsets  # 最後にオンセットを検出したフレーム
        
        for i in range(1, len(self.state_buffer)):
            # 条件1: 非調和状態(N)への遷移
            is_non_harmonic_transition = (self.state_buffer[i] == 2) and (self.state_buffer[i-1] != 2)
            
            # 条件2: (C-D)の急激な減少（微分値が負の大きな値）
            cd_diff = 0
            if i >= 2:
                # (C-D)の現在値と1フレーム前の差分
                cd_current = self.feature_buffer[i, 0] - self.feature_buffer[i, 2]
                cd_prev = self.feature_buffer[i-1, 0] - self.feature_buffer[i-1, 2]
                cd_diff = cd_current - cd_prev
            
            # 急激な低下を検出
            is_cd_falling = cd_diff < -0.3  # 閾値はパラメータ化することも可能
            
            # 条件3: 急変度が一定以上
            high_change = self.feature_buffer[i, 2] > self.onset_threshold
            
            # オンセット条件: 非調和状態への遷移、または(C-D)の急落、または急変度が高い
            if (is_non_harmonic_transition or is_cd_falling or high_change) and (i - last_onset >= min_frames_between_onsets):
                # 先読みバッファを考慮（リアルタイム処理で必要な遅延）
                if i >= self.look_ahead:
                    actual_onset_frame = i - self.look_ahead
                    onset_frames.append(actual_onset_frame)
                    last_onset = i
                else:
                    # 先読みバッファより前の場合は、そのまま使用
                    onset_frames.append(i)
                    last_onset = i
        
        # フレームインデックスを時間（秒）に変換
        onsets = np.array(onset_frames) * self.hop_length / sr
        
        print(f"検出されたオンセット数: {len(onsets)}")
        return onsets
    
    def detect_offsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        音声データからオフセット（音の終了時点）を検出します
        理論ドキュメントに準拠した「カオス埋没」条件に基づいて実装
        
        Parameters
        ----------
        audio : np.ndarray
            分析する音声データ
        sr : int
            サンプリングレート
            
        Returns
        -------
        np.ndarray
            検出されたオフセット時間（秒）の配列
        """
        # 特徴量が未計算の場合は計算する
        if self.feature_buffer is None:
            self._compute_features(audio, sr)
            if self.use_viterbi:
                self._estimate_states_viterbi()
            else:
                self._estimate_states()
        
        # カオス埋没によるオフセット検出（理論ドキュメント準拠版）
        offset_frames = []
        i = self.look_ahead  # 先読みバッファを考慮した開始位置
        
        while i < len(self.hmm_posterior) - self.offset_min_frames:
            # 条件1: カオス(C)が優勢かどうか
            chaos_dominant = np.all(self.hmm_posterior[i:i+self.offset_min_frames, 1] > self.offset_chaos_threshold)
            
            # 条件2: 安定(S)と非調和(N)の区別が困難になっているか
            # |π_S - π_N| が小さければ区別困難（埋没状態）
            stability_diff = np.all(np.abs(self.hmm_posterior[i:i+self.offset_min_frames, 0] - 
                                          self.hmm_posterior[i:i+self.offset_min_frames, 2]) < self.offset_stability_diff)
            
            # オフセット条件: カオス優勢 かつ 安定・非調和の区別困難（カオス埋没）
            if chaos_dominant and stability_diff:
                # 先読みバッファを考慮した位置にオフセットを記録
                offset_frames.append(i - self.look_ahead)
                # 連続してオフセットを検出しないように適切なスキップ
                i += self.offset_min_frames
            else:
                i += 1
        
        # フレームインデックスを時間（秒）に変換
        offsets = np.array(offset_frames) * self.hop_length / sr
        
        print(f"検出されたオフセット数: {len(offsets)}")
        return offsets
    
    def detect_pitches(self, audio: np.ndarray, sr: int, 
                      onsets: Optional[np.ndarray] = None, 
                      offsets: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        音声データからピッチ情報を検出します
        
        Parameters
        ----------
        audio : np.ndarray
            分析する音声データ
        sr : int
            サンプリングレート
        onsets : np.ndarray, optional
            検出済みのオンセット時間（秒）の配列
        offsets : np.ndarray, optional
            検出済みのオフセット時間（秒）の配列
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
            (ピッチ時間, ピッチ周波数, 信頼度) のタプル
        """
        # オンセット/オフセットが未検出の場合は検出する
        if onsets is None:
            onsets = self.detect_onsets(audio, sr)
        if offsets is None:
            offsets = self.detect_offsets(audio, sr)
        
        # ノート区間ごとに安定度に基づくピッチ推定を行う
        pitch_times = []
        pitch_freqs = []
        confidences = []
        
        # 各ノート区間に対してピッチ推定
        for i in range(len(onsets)):
            # 対応するオフセットを見つける
            offset_idx = np.searchsorted(offsets, onsets[i])
            if offset_idx >= len(offsets):
                # オフセットが見つからない場合は次のオンセットまで、または音声終了まで
                if i + 1 < len(onsets):
                    note_end = onsets[i + 1]
                else:
                    note_end = len(audio) / sr
            else:
                note_end = offsets[offset_idx]
            
            # ノート区間のフレームインデックス
            start_frame = int(onsets[i] * sr / self.hop_length)
            end_frame = int(note_end * sr / self.hop_length)
            
            # 区間が十分な長さを持つ場合のみ処理
            if end_frame - start_frame >= 3:
                # エンベロープの安定性に基づくピッチ推定
                pitches, times, confs = self._estimate_pitch_for_segment(
                    audio, sr, start_frame, end_frame)
                
                pitch_times.extend(times)
                pitch_freqs.extend(pitches)
                confidences.extend(confs)
        
        return np.array(pitch_times), np.array(pitch_freqs), np.array(confidences)
    
    def detect_notes(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        音声データから音符情報（開始時間、終了時間、ピッチ）を検出します
        
        Parameters
        ----------
        audio : np.ndarray
            分析する音声データ
        sr : int
            サンプリングレート
            
        Returns
        -------
        Dict[str, np.ndarray]
            音符情報を含む辞書（'onset_times', 'offset_times', 'pitches' のキーを持つ）
        """
        # オンセットとオフセットを検出
        onsets = self.detect_onsets(audio, sr)
        offsets = self.detect_offsets(audio, sr)
        
        # ノート単位でマッチングしてピッチを割り当て
        notes = self._match_notes(audio, sr, onsets, offsets)
        
        return notes
    
    def _compute_features(self, audio: np.ndarray, sr: int):
        """
        音声データから協和度(C)、カオス度(K)、急変度(D)の特徴量を計算します
        
        Parameters
        ----------
        audio : np.ndarray
            分析する音声データ
        sr : int
            サンプリングレート
        """
        # フレーム数の計算
        n_frames = 1 + (len(audio) - self.frame_length) // self.hop_length
        
        # 特徴量バッファの初期化 (C, K, D)
        self.feature_buffer = np.zeros((n_frames, 3))
        
        # 過去のエネルギー値を保持する変数
        prev_energies = []
        energy_window = 5  # エネルギー変化を評価するウィンドウサイズ
        
        # デバッグ用に値の範囲を追跡
        c_values = []
        k_values = []
        d_values = []
        
        # スペクトル平坦度計算用のFFTサイズ
        n_fft = 2 ** int(np.ceil(np.log2(self.frame_length)))
        
        # 各フレームで特徴量を計算
        for i in range(n_frames):
            # フレーム取得
            start = i * self.hop_length
            end = start + self.frame_length
            frame = audio[start:end]
            
            # RMS エネルギー計算
            frame_energy = np.sqrt(np.mean(frame ** 2))
            
            # 微弱信号の場合は特別扱い
            if frame_energy < 1e-5:
                self.feature_buffer[i] = [0.1, 0.5, 0.0]  # 低協和度、中カオス度、低急変度
                continue
            
            # 振幅エンベロープの抽出
            envelope = np.abs(signal.hilbert(frame))
            envelope_energy = np.sum(envelope ** 2) / len(envelope)
            
            # 協和度(C): 自己相関のピーク値を使用
            acf = self._compute_autocorrelation(envelope)
            
            # 自己相関が無効（全てのラグでほぼ0）の場合は低い協和度を設定
            if np.max(acf) < 1e-5:
                harmony_score = 0.1
            else:
                # 最初のピークはラグ0なので除外
                # ラグが小さすぎるピークは無視 (最低周波数に相当するラグ以上を探索)
                min_lag = max(1, int(sr / self.frame_length / 20))  # 例: 20Hz以上の周期性を探索
                if len(acf) > min_lag + 1:
                    # より低周波の周期性を優先するため、ある程度の範囲でピークを探す
                    max_lag = min(len(acf) - 1, int(sr / self.frame_length / 0.5))  # 0.5Hz以上
                    peak_idx = min_lag + np.argmax(acf[min_lag:max_lag])
                    harmony_score = acf[peak_idx]
                    
                    # 自己相関ピークの鮮明さも考慮（ピークが明確かどうか）
                    if peak_idx > 0 and peak_idx < len(acf) - 1:
                        peak_prominence = harmony_score - 0.5 * (acf[peak_idx-1] + acf[peak_idx+1])
                        harmony_score = max(0.1, min(1.0, harmony_score * (1 + peak_prominence)))
                else:
                    harmony_score = 0.2
            
            # 結果を保存
            self.feature_buffer[i, 0] = harmony_score
            c_values.append(harmony_score)
            
            # カオス度(K): スペクトル平坦度を使用
            # スペクトル平坦度 = 幾何平均/算術平均（値が1に近いほどホワイトノイズに近く、カオス度が高い）
            mag_spectrum = np.abs(np.fft.rfft(frame, n_fft))
            mag_spectrum = mag_spectrum[1:]  # DC成分除去
            
            if np.mean(mag_spectrum) < 1e-10:
                chaos_score = 0.5  # 信号が弱い場合は中程度のカオス度
            else:
                # スペクトル平坦度の計算
                log_mean = np.mean(np.log(mag_spectrum + 1e-12))
                linear_mean = np.mean(mag_spectrum)
                spectral_flatness = np.exp(log_mean) / (linear_mean + 1e-12)
                
                # スペクトル平坦度を0-1の範囲にスケーリング
                # 0に近いほど単一周波数（安定）、1に近いほどホワイトノイズ（カオス）
                chaos_score = min(1.0, spectral_flatness * 2.0)
                
                # エンベロープのスペクトル平坦度も考慮
                env_spectrum = np.abs(np.fft.rfft(envelope, n_fft))
                env_spectrum = env_spectrum[1:]  # DC成分除去
                if np.mean(env_spectrum) > 1e-10:
                    env_log_mean = np.mean(np.log(env_spectrum + 1e-12))
                    env_linear_mean = np.mean(env_spectrum)
                    env_flatness = np.exp(env_log_mean) / (env_linear_mean + 1e-12)
                    # エンベロープのカオス度も加味
                    chaos_score = 0.7 * chaos_score + 0.3 * min(1.0, env_flatness * 2.0)
            
            self.feature_buffer[i, 1] = chaos_score
            k_values.append(chaos_score)
            
            # 急変度(D): エネルギー差分とスペクトル変化を組み合わせる
            prev_energies.append(envelope_energy)
            if len(prev_energies) > energy_window:
                prev_energies.pop(0)
            
            if i > 0:
                # エネルギー差分ベースの急変度
                energy_diff = abs(envelope_energy - prev_energies[-2])
                
                # 相対変化量を考慮（小さいエネルギーでも大きな相対変化は急変とみなす）
                relative_change = energy_diff / (prev_energies[-2] + 1e-5)
                
                # スペクトル変化を計算（オプション）
                if i >= 1:
                    prev_frame = audio[(i-1)*self.hop_length:(i-1)*self.hop_length+self.frame_length]
                    spec1 = np.abs(np.fft.rfft(frame))
                    spec2 = np.abs(np.fft.rfft(prev_frame))
                    if len(spec1) > 0 and len(spec2) > 0:
                        spec_diff = np.sum(np.abs(spec1 - spec2)) / (np.sum(spec1 + spec2) + 1e-5)
                        # スペクトル変化も急変度に加味
                        spectral_change = min(1.0, spec_diff * 5.0)
                    else:
                        spectral_change = 0.0
                else:
                    spectral_change = 0.0
                
                # 最終的な急変度は、エネルギー変化とスペクトル変化の重み付き和
                # エネルギーの相対変化が小さくても、スペクトル変化が大きければ急変とみなす
                energy_change = min(1.0, relative_change * 5.0)
                change_score = 0.7 * energy_change + 0.3 * spectral_change
                
                # 急激な立ち上がりをより強調
                if relative_change > 0.2 and envelope_energy > np.mean(prev_energies[:-1]) * 1.5:
                    change_score = max(change_score, 0.7)  # 急激な増加は強く評価
                
                self.feature_buffer[i, 2] = change_score
                d_values.append(change_score)
            else:
                # 最初のフレームは差分が計算できないので低い値を設定
                self.feature_buffer[i, 2] = 0.1
                d_values.append(0.1)
        
        # デバッグ出力（実際の値の分布を確認）
        print(f"協和度(C) 範囲: {min(c_values):.3f}～{max(c_values):.3f}, 平均: {np.mean(c_values):.3f}")
        print(f"カオス度(K) 範囲: {min(k_values):.3f}～{max(k_values):.3f}, 平均: {np.mean(k_values):.3f}")
        print(f"急変度(D) 範囲: {min(d_values):.3f}～{max(d_values):.3f}, 平均: {np.mean(d_values):.3f}")
    
    def _compute_autocorrelation(self, x: np.ndarray) -> np.ndarray:
        """
        信号の自己相関関数を計算します（FFTを使用した高速実装）
        
        Parameters
        ----------
        x : np.ndarray
            自己相関を計算する信号
            
        Returns
        -------
        np.ndarray
            正規化された自己相関関数
        """
        n = len(x)
        # パディングしてFFTサイズを2のべき乗に
        fft_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        
        # 信号のゼロ平均化
        x_centered = x - np.mean(x)
        
        # FFTを使用した自己相関計算
        X = fftpack.fft(x_centered, fft_size)
        power_spectrum = np.abs(X) ** 2
        acf = np.real(fftpack.ifft(power_spectrum))
        
        # サイズをもとの信号長に戻し、正規化
        acf = acf[:n]
        
        # 0除算を防ぐため、分母が極めて小さい場合は特別扱い
        if acf[0] < 1e-10:
            return np.zeros_like(acf)
        
        acf /= acf[0]
        
        return acf
    
    def _estimate_states(self):
        """
        特徴量から状態(S, C, N)を推定します
        HMMを使用した手法B（高精度版）を実装
        """
        n_frames = len(self.feature_buffer)
        n_states = 3  # S, C, N
        
        # HMM事後確率の初期化
        self.hmm_posterior = np.zeros((n_frames, n_states))
        
        # 初期状態確率（各状態の特徴が均等になるように設定）
        # 無条件に等確率ではなく、最初のフレームの特徴に基づいて設定
        if n_frames > 0:
            init_obs_prob = self._compute_observation_probability(self.feature_buffer[0])
            pi = init_obs_prob  # 最初のフレームの観測確率を初期状態確率として使用
        else:
            pi = np.ones(n_states) / n_states
        
        # 状態バッファの初期化
        self.state_buffer = np.zeros(n_frames, dtype=int)
        
        # 遷移確率行列の調整（より緩やかな遷移を許容）
        # 状態変化を起こりやすくするため、対角成分を少し下げる
        transition_matrix = np.array(self.hmm_trans_matrix).copy()
        # 対角成分を少し小さく（0.8→0.7など）、他への遷移確率を上げる
        for i in range(n_states):
            if transition_matrix[i, i] > 0.7:
                diff = transition_matrix[i, i] - 0.7
                transition_matrix[i, i] = 0.7
                # 差分を他の遷移に分配
                remaining = np.sum(transition_matrix[i, :]) - transition_matrix[i, i]
                if remaining > 0:
                    for j in range(n_states):
                        if j != i:
                            transition_matrix[i, j] += diff * transition_matrix[i, j] / remaining
        
        # フォワードアルゴリズムで事後確率を計算
        for t in range(n_frames):
            # 観測確率の計算
            b = self._compute_observation_probability(self.feature_buffer[t])
            
            if t == 0:
                # 初期フレーム
                alpha = pi * b
            else:
                # 遷移確率を使用した更新
                alpha = np.zeros(n_states)
                for j in range(n_states):
                    for i in range(n_states):
                        alpha[j] += self.hmm_posterior[t-1, i] * transition_matrix[i, j]
                    alpha[j] *= b[j]
            
            # 正規化
            alpha_sum = np.sum(alpha)
            if alpha_sum > 0:
                self.hmm_posterior[t] = alpha / alpha_sum
            else:
                # 観測確率をそのまま使用
                self.hmm_posterior[t] = b
            
            # 最も確率の高い状態を選択
            self.state_buffer[t] = np.argmax(self.hmm_posterior[t])
        
        # デバッグ出力
        states_count = np.bincount(self.state_buffer, minlength=3)
        print(f"状態分布 - S:{states_count[0]}, C:{states_count[1]}, N:{states_count[2]}")
    
    def _compute_observation_probability(self, feature: np.ndarray) -> np.ndarray:
        """
        特徴量ベクトルに対する各状態の観測確率を計算します
        
        Parameters
        ----------
        feature : np.ndarray
            特徴量ベクトル (C, K, D)
            
        Returns
        -------
        np.ndarray
            各状態(S, C, N)の観測確率
        """
        # 協和度(C)、カオス度(K)、急変度(D)
        C, K, D = feature
        
        # 各状態の観測確率をより柔軟なモデルで計算
        # 安定状態(S): 高協和性が主要因、カオスや急変が低いほど良い
        # 単純な積ではなく、協和度を重視した重み付け
        p_S = 0.7 * C + 0.2 * (1 - K) + 0.1 * (1 - D)
        
        # カオス状態(C): カオス度が主要因、協和度はある程度許容
        p_C = 0.6 * K + 0.3 * (1 - D) + 0.1 * (1 - C)
        
        # 非調和状態(N): 急変度が主要因
        p_N = 0.7 * D + 0.2 * (1 - C) + 0.1 * K
        
        # より明確な状態分類のために、状態間の差を強調
        # 安定とカオスが近い場合、もっとも強い特徴を持つ状態を優先
        max_val = max(p_S, p_C, p_N)
        if max_val == p_S:
            p_S = max(p_S, 0.6)  # 協和度が支配的なら、さらに強調
        elif max_val == p_C:
            p_C = max(p_C, 0.6)  # カオス度が支配的なら、さらに強調
        elif max_val == p_N:
            p_N = max(p_N, 0.6)  # 急変度が支配的なら、さらに強調
        
        # 確率の正規化
        p = np.array([p_S, p_C, p_N])
        p = np.clip(p, 1e-10, 1.0)  # 数値の安定化
        p /= np.sum(p)
        
        return p
    
    def _estimate_pitch_for_segment(self, audio: np.ndarray, sr: int, 
                                   start_frame: int, end_frame: int) -> Tuple[List[float], List[float], List[float]]:
        """
        指定されたセグメントのピッチを推定します
        
        Parameters
        ----------
        audio : np.ndarray
            分析する音声データ
        sr : int
            サンプリングレート
        start_frame : int
            開始フレームインデックス
        end_frame : int
            終了フレームインデックス
            
        Returns
        -------
        Tuple[List[float], List[float], List[float]]
            (ピッチ周波数リスト, 時間リスト, 信頼度リスト)
        """
        # セグメントの抽出
        start_sample = start_frame * self.hop_length
        end_sample = min(end_frame * self.hop_length + self.frame_length, len(audio))
        segment = audio[start_sample:end_sample]
        
        # セグメントが短すぎる場合は信頼度の低いピッチを返す
        if len(segment) < self.frame_length:
            # 50ms未満のセグメントは信頼度低め
            default_pitch = 440.0  # 仮のピッチ値
            default_time = (start_frame + end_frame) * 0.5 * self.hop_length / sr
            return [default_pitch], [default_time], [0.2]
        
        # 周波数帯域に分割してエンベロープの安定性を分析
        pitches = []
        times = []
        confidences = []
        
        # 周波数帯域の設定（対数スケール）
        bands = np.logspace(np.log10(self.pitch_min_freq), 
                          np.log10(self.pitch_max_freq), 
                          self.pitch_band_count + 1)
        
        # 全体の振幅RMS値を計算（弱い部分を無視するため）
        segment_rms = np.sqrt(np.mean(segment ** 2))
        if segment_rms < 1e-5:
            # 無音に近いセグメントは信頼度低めの推定値を返す
            mid_freq = np.sqrt(self.pitch_min_freq * self.pitch_max_freq)  # 幾何平均
            mid_time = (start_frame + end_frame) * 0.5 * self.hop_length / sr
            return [mid_freq], [mid_time], [0.1]
        
        # セグメント内の各時点でピッチ推定
        # 簡略化のため、フレームごとではなく適度な間隔で推定
        num_points = min(10, end_frame - start_frame)
        step = max(1, (end_frame - start_frame) // num_points)
        
        for frame_idx in range(start_frame, end_frame, step):
            time_sec = frame_idx * self.hop_length / sr
            
            # 現在のフレームから小さなセグメントを取得
            curr_start = (frame_idx - start_frame) * self.hop_length
            curr_end = min(curr_start + self.frame_length, len(segment))
            if curr_end - curr_start < self.frame_length // 2:
                continue  # フレームが短すぎる場合はスキップ
            
            frame_segment = segment[curr_start:curr_end]
            
            # フレームが弱すぎる場合は無視
            frame_energy = np.sqrt(np.mean(frame_segment ** 2))
            if frame_energy < segment_rms * 0.3:  # 全体の30%未満のエネルギーはスキップ
                continue
            
            # 周波数帯域ごとの安定性スコアを計算
            stability_scores = np.zeros(self.pitch_band_count)
            peak_lags = np.zeros(self.pitch_band_count)
            spectral_peaks = []  # スペクトル分析によるピーク周波数
            
            # スペクトル分析（FFT）によるピーク検出
            n_fft = min(2048, len(frame_segment))
            if n_fft > 32:  # 最低限のFFTサイズチェック
                window = np.hanning(n_fft)
                if len(frame_segment) >= n_fft:
                    windowed = frame_segment[:n_fft] * window
                    spect = np.abs(np.fft.rfft(windowed))
                    freqs = np.fft.rfftfreq(n_fft, 1.0/sr)
                    
                    # ピッチの範囲内のスペクトルピークを検出
                    in_range_idx = np.where((freqs >= self.pitch_min_freq) & (freqs <= self.pitch_max_freq))[0]
                    if len(in_range_idx) > 1:
                        in_range_spect = spect[in_range_idx]
                        in_range_freqs = freqs[in_range_idx]
                        
                        # ピーク検出（最大5つまで）
                        peak_idx = signal.find_peaks(in_range_spect, height=np.max(in_range_spect)*0.3, distance=3)[0]
                        if len(peak_idx) > 0:
                            peak_values = in_range_spect[peak_idx]
                            sorted_idx = np.argsort(-peak_values)[:5]  # 上位5つのピーク
                            for idx in sorted_idx:
                                spectral_peaks.append((in_range_freqs[peak_idx[idx]], peak_values[idx]))
            
            # バンドパス処理によるピッチ推定
            for i in range(self.pitch_band_count):
                # バンドパスフィルタの適用
                low_freq = bands[i]
                high_freq = bands[i+1]
                
                # バターワースフィルタの次数
                order = 4
                
                # 正規化カットオフ周波数
                nyquist = 0.5 * sr
                low = low_freq / nyquist
                high = high_freq / nyquist
                
                # フィルタ適用
                if low < 1.0 and high < 1.0:  # 周波数が有効範囲内か確認
                    try:
                        b, a = signal.butter(order, [low, high], btype='band')
                        filtered = signal.filtfilt(b, a, frame_segment)
                        
                        # フィルタ後の信号が弱すぎる場合はスキップ
                        filt_energy = np.sqrt(np.mean(filtered ** 2))
                        if filt_energy < frame_energy * 0.1:  # 元の10%未満のエネルギーはスキップ
                            continue
                        
                        # エンベロープ抽出
                        envelope = np.abs(signal.hilbert(filtered))
                        
                        # 自己相関による安定性評価
                        acf = self._compute_autocorrelation(envelope)
                        if len(acf) > 1:
                            # 低周波優先でピーク探索
                            min_lag = max(1, int(sr / high_freq / 4))
                            max_lag = min(len(acf) - 1, int(sr / low_freq * 1.5))
                            if max_lag > min_lag:
                                try:
                                    # ピークの明確さも評価（周辺との差）
                                    peak_idx = min_lag + np.argmax(acf[min_lag:max_lag])
                                    peak_val = acf[peak_idx]
                                    nearby_vals = acf[max(0, peak_idx-3):min(len(acf), peak_idx+4)]
                                    nearby_vals = np.delete(nearby_vals, 3)  # ピーク自体を除く
                                    peak_prominence = peak_val - np.mean(nearby_vals)
                                    
                                    # スコアにはピーク値とその明確さを両方考慮
                                    clarity_factor = 1.0 + peak_prominence * 2.0
                                    stability_scores[i] = peak_val * clarity_factor
                                    
                                    # ピーク位置から周波数を推定（逆数）
                                    if peak_idx > 0:
                                        estimated_freq = sr / peak_idx
                                        # そのバンドの範囲内かチェック
                                        if low_freq * 0.8 <= estimated_freq <= high_freq * 1.2:
                                            peak_lags[i] = peak_idx
                                except:
                                    # エラーが発生した場合は低いスコアを設定
                                    stability_scores[i] = 0.0
                    except:
                        # フィルタ適用に失敗した場合は無視
                        pass
            
            # バンドパス+自己相関によるピッチ候補とFFTピークを組み合わせる
            combined_candidates = []
            
            # 自己相関の上位候補
            if np.max(stability_scores) > 0.2:  # 最低信頼度閾値
                # 上位3つのバンドを選択
                best_band_indices = np.argsort(-stability_scores)[:3]
                for idx in best_band_indices:
                    if stability_scores[idx] > 0.2:
                        # 自己相関ラグから周波数を推定
                        if peak_lags[idx] > 0:
                            estimated_freq = sr / peak_lags[idx]
                            combined_candidates.append((estimated_freq, stability_scores[idx], 'acf'))
                        else:
                            # ラグがない場合はバンドの中心周波数を使用
                            center_freq = np.sqrt(bands[idx] * bands[idx+1])
                            combined_candidates.append((center_freq, stability_scores[idx] * 0.5, 'band'))
            
            # FFTピークを追加
            for freq, magnitude in spectral_peaks:
                # マグニチュードを0-1に正規化
                norm_magnitude = magnitude / (np.max([p[1] for p in spectral_peaks]) + 1e-10)
                confidence = min(1.0, norm_magnitude * 0.8)  # 最大0.8の信頼度
                combined_candidates.append((freq, confidence, 'fft'))
            
            # 候補が見つかれば処理
            if combined_candidates:
                # 信頼度でソート
                combined_candidates.sort(key=lambda x: -x[1])
                
                # 類似周波数をマージ
                merged_candidates = []
                for freq, conf, source in combined_candidates:
                    # すでに似た周波数が追加されていないかチェック
                    is_new = True
                    for i, (existing_freq, existing_conf, _) in enumerate(merged_candidates):
                        # 周波数比が1.05（半音程度）以内ならマージ
                        freq_ratio = max(freq, existing_freq) / min(freq, existing_freq)
                        if freq_ratio < 1.05:
                            # 高い信頼度を持つ方を優先し、周波数を加重平均
                            weight1 = existing_conf
                            weight2 = conf
                            new_freq = (existing_freq * weight1 + freq * weight2) / (weight1 + weight2)
                            new_conf = max(existing_conf, conf)
                            merged_candidates[i] = (new_freq, new_conf, source)
                            is_new = False
                            break
                    
                    if is_new and len(merged_candidates) < 3:  # 最大3つまで
                        merged_candidates.append((freq, conf, source))
                
                # 最も信頼度の高い候補を選択
                if merged_candidates:
                    best_freq, best_conf, _ = merged_candidates[0]
                    # 最低信頼度チェック（0.25以上だけ採用）
                    if best_conf > 0.25:
                        pitches.append(best_freq)
                        times.append(time_sec)
                        confidences.append(best_conf)
        
        # セグメント全体の代表ピッチを1つ決める場合は以下を使用
        if len(pitches) == 0:
            # 候補が見つからない場合のデフォルト値
            mid_freq = np.sqrt(self.pitch_min_freq * self.pitch_max_freq)
            mid_time = (start_frame + end_frame) * 0.5 * self.hop_length / sr
            return [mid_freq], [mid_time], [0.1]
        
        print(f"セグメント内で検出されたピッチ数: {len(pitches)}")
        return pitches, times, confidences
    
    def _match_notes(self, audio: np.ndarray, sr: int, 
                    onsets: np.ndarray, offsets: np.ndarray) -> Dict[str, np.ndarray]:
        """
        オンセットとオフセットをマッチングして音符情報を生成し、
        各音符にピッチを割り当てます
        
        Parameters
        ----------
        audio : np.ndarray
            分析する音声データ
        sr : int
            サンプリングレート
        onsets : np.ndarray
            検出されたオンセット時間（秒）の配列
        offsets : np.ndarray
            検出されたオフセット時間（秒）の配列
            
        Returns
        -------
        Dict[str, np.ndarray]
            音符情報を含む辞書（'onset_times', 'offset_times', 'pitches' のキーを持つ）
        """
        matched_onsets = []
        matched_offsets = []
        note_pitches = []
        
        # オンセットがない場合、過剰な検出を減らし最小限のオンセットを自動生成
        if len(onsets) == 0:
            # シンプルなエネルギーベースでいくつかのオンセットを生成
            audio_frames = []
            frame_size = self.frame_length
            hop_size = self.hop_length
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i+frame_size]
                energy = np.sqrt(np.mean(frame ** 2))
                audio_frames.append(energy)
            
            # エネルギーの急激な増加を検出
            energy_diff = np.zeros_like(audio_frames)
            for i in range(1, len(audio_frames)):
                if audio_frames[i-1] > 0:
                    energy_diff[i] = audio_frames[i] / audio_frames[i-1]
                else:
                    energy_diff[i] = 1.0
            
            # 閾値以上のエネルギー増加をオンセットとする
            auto_onsets = []
            for i in range(1, len(energy_diff)):
                if (energy_diff[i] > 1.5 and  # 50%以上のエネルギー増加
                    audio_frames[i] > 0.05 * max(audio_frames)):  # 最大の5%以上のエネルギー
                    auto_onsets.append(i * hop_size / sr)
            
            # 間隔が近すぎるオンセットをフィルタリング
            if auto_onsets:
                filtered_onsets = [auto_onsets[0]]
                min_interval = 0.1  # 100ms
                for onset in auto_onsets[1:]:
                    if onset - filtered_onsets[-1] >= min_interval:
                        filtered_onsets.append(onset)
                
                onsets = np.array(filtered_onsets)
            else:
                # それでも見つからない場合、音声を大まかに分割
                if len(audio) > 0:
                    total_duration = len(audio) / sr
                    if total_duration > 1.0:
                        # 1秒以上なら2〜3分割
                        num_segments = max(2, min(3, int(total_duration)))
                        onsets = np.linspace(0, total_duration * 0.6, num_segments)
                    else:
                        # 短い音声なら先頭だけ
                        onsets = np.array([0.0])
                else:
                    return {
                        'onset_times': np.array([]),
                        'offset_times': np.array([]),
                        'pitches': np.array([])
                    }
        
        # オフセットがない場合は音声の終了時または次のオンセットを使用
        if len(offsets) == 0:
            # 音声長に基づいてデフォルトの持続時間を設定
            audio_duration = len(audio) / sr
            default_note_duration = min(1.0, audio_duration / max(1, len(onsets)))
            
            auto_offsets = []
            for i, onset in enumerate(onsets):
                if i < len(onsets) - 1:
                    # 次のオンセットの少し手前
                    offset = onsets[i+1] - 0.05
                    # オンセットより後になるように
                    offset = max(onset + 0.1, offset)
                else:
                    # 最後のノートは音声終了時または一定時間後
                    offset = min(audio_duration, onset + default_note_duration)
                
                auto_offsets.append(offset)
            
            offsets = np.array(auto_offsets)
        
        print(f"マッチング前のオンセット数: {len(onsets)}, オフセット数: {len(offsets)}")
        
        # 各オンセットに対応するオフセットを見つける
        last_offset_idx = 0
        for onset_idx, onset_time in enumerate(onsets):
            # このオンセットより後の最初のオフセットを探す
            offset_idx = last_offset_idx
            while offset_idx < len(offsets) and offsets[offset_idx] <= onset_time:
                offset_idx += 1
            
            # 対応するオフセットが見つかった場合
            if offset_idx < len(offsets):
                # 次のオンセットがある場合、そのオンセットかオフセットの早い方を終了時間とする
                if onset_idx + 1 < len(onsets) and onsets[onset_idx + 1] < offsets[offset_idx]:
                    offset_time = onsets[onset_idx + 1] - 0.01  # 1ms前
                else:
                    offset_time = offsets[offset_idx]
                    last_offset_idx = offset_idx
                
                # 音符が短すぎないか確認
                note_duration = offset_time - onset_time
                if note_duration < 0.05:  # 50ms未満は短すぎる
                    continue
                
                # ピッチ推定
                start_frame = int(onset_time * sr / self.hop_length)
                end_frame = int(offset_time * sr / self.hop_length)
                
                # 十分な長さを持つノートのみ処理
                min_frames = 2
                if end_frame - start_frame >= min_frames:
                    # ノート区間のピッチ推定
                    pitches, times, confidences = self._estimate_pitch_for_segment(
                        audio, sr, start_frame, end_frame)
                    
                    if len(pitches) > 0:
                        # 信頼度加重平均でピッチを決定
                        weighted_pitches = np.array(pitches) * np.array(confidences)
                        avg_pitch = np.sum(weighted_pitches) / np.sum(confidences) if np.sum(confidences) > 0 else pitches[0]
                        
                        matched_onsets.append(onset_time)
                        matched_offsets.append(offset_time)
                        note_pitches.append(avg_pitch)
        
        print(f"最終的に検出された音符数: {len(matched_onsets)}")
        return {
            'onset_times': np.array(matched_onsets),
            'offset_times': np.array(matched_offsets),
            'pitches': np.array(note_pitches)
        }
    
    def _estimate_states_viterbi(self):
        """
        先読みバッファを活用したViterbiアルゴリズムによる状態(S, C, N)の推定
        
        理論ドキュメントに準拠した高精度版実装で、特徴量から計算される観測確率と
        遷移確率行列を使用して、最も確からしい状態列を推定します。
        """
        n_frames = len(self.feature_buffer)
        n_states = 3  # S, C, N
        
        # 特徴量が存在しない場合は処理を行わない
        if n_frames == 0:
            self.state_buffer = np.array([])
            self.hmm_posterior = np.array([]).reshape(0, n_states)
            return
        
        # 1) 全フレームの観測確率 b[t, j] を先に計算
        b = np.zeros((n_frames, n_states))
        for t in range(n_frames):
            if self.use_gaussian_obs:
                b[t] = self._compute_observation_probability_gaussian(self.feature_buffer[t])
            else:
                b[t] = self._compute_observation_probability(self.feature_buffer[t])
        
        # 2) Viterbiの δ[t, j], ψ[t, j] を確保
        delta = np.zeros((n_frames, n_states))
        psi = np.zeros((n_frames, n_states), dtype=int)
        
        # 初期化
        # 最初のフレームの観測確率を初期状態確率として使用
        delta[0] = b[0]
        # 正規化
        delta_sum = np.sum(delta[0])
        if delta_sum > 0:
            delta[0] /= delta_sum
        
        # 3) 前向きにdeltaを更新
        transition_matrix = np.array(self.hmm_trans_matrix).copy()
        for t in range(1, n_frames):
            for j in range(n_states):
                # i->j の遷移を考慮し、最大となるiを探す
                candidates = delta[t-1] * transition_matrix[:, j]
                best_i = np.argmax(candidates)
                delta[t, j] = candidates[best_i] * b[t, j]
                psi[t, j] = best_i
            
            # 正規化（数値安定性のため）
            delta_sum = np.sum(delta[t])
            if delta_sum > 0:
                delta[t] /= delta_sum
        
        # 4) 最終フレームで最大確率状態を探し、逆順に辿る
        state_seq = np.zeros(n_frames, dtype=int)
        state_seq[-1] = np.argmax(delta[-1])
        for t in reversed(range(n_frames-1)):
            state_seq[t] = psi[t+1, state_seq[t+1]]
        
        # 結果を保存
        self.state_buffer = state_seq
        self.hmm_posterior = delta  # 正規化された事後確率として使用
        
        # デバッグ出力
        states_count = np.bincount(self.state_buffer, minlength=3)
        print(f"Viterbi状態分布 - S:{states_count[0]}, C:{states_count[1]}, N:{states_count[2]}")
    
    def _compute_observation_probability_gaussian(self, feature: np.ndarray) -> np.ndarray:
        """
        特徴量ベクトルに対する各状態の観測確率を3次元ガウシアンモデルで計算します
        
        Parameters
        ----------
        feature : np.ndarray
            特徴量ベクトル (C, K, D)
            
        Returns
        -------
        np.ndarray
            各状態(S, C, N)の観測確率
        """
        # ガウシアンの平均ベクトル（C, K, D）
        # 安定状態: 高協和、低カオス、低急変
        # カオス状態: 中協和、高カオス、低急変
        # 非調和状態: 低協和、中カオス、高急変
        mu_S = np.array([0.8, 0.2, 0.1])  # 安定状態の平均
        mu_C = np.array([0.4, 0.8, 0.2])  # カオス状態の平均
        mu_N = np.array([0.2, 0.5, 0.8])  # 非調和状態の平均
        
        # 共分散行列（簡略化のため対角行列を使用）
        sigma_S = np.array([0.2, 0.2, 0.1])  # 安定状態の分散
        sigma_C = np.array([0.3, 0.2, 0.2])  # カオス状態の分散
        sigma_N = np.array([0.2, 0.3, 0.2])  # 非調和状態の分散
        
        # 各状態の観測確率を計算
        log_p = np.zeros(3)
        
        # 安定状態(S)の観測確率
        diff_S = feature - mu_S
        log_p[0] = -0.5 * np.sum((diff_S ** 2) / sigma_S)
        
        # カオス状態(C)の観測確率
        diff_C = feature - mu_C
        log_p[1] = -0.5 * np.sum((diff_C ** 2) / sigma_C)
        
        # 非調和状態(N)の観測確率
        diff_N = feature - mu_N
        log_p[2] = -0.5 * np.sum((diff_N ** 2) / sigma_N)
        
        # 対数確率から通常の確率に変換し、正規化
        p = np.exp(log_p - np.max(log_p))  # 数値安定性のため最大値を引く
        p_sum = np.sum(p)
        if p_sum > 0:
            p /= p_sum
        
        return p 