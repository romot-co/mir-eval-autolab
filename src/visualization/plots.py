"""
可視化処理の共通モジュール

このモジュールでは、検出結果や評価結果を可視化するための共通関数を提供します。
波形やピッチデータのプロット、評価メトリクスの可視化などの機能があります。
"""

import os
import numpy as np
import matplotlib
# バックエンドをAggに設定（GUIを使わないため）
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Union, Optional, Any
import logging  # 追加: loggingモジュール

# 表示言語設定（日本語対応）
try:
    import japanize_matplotlib
except ImportError:
    pass

# カラーパレット設定
COLORS = {
    'reference': '#3498db',   # 青：参照データ
    'detection': '#e74c3c',   # 赤：検出データ
    'waveform': '#2c3e50',    # 濃紺：波形
    'background': '#ecf0f1',  # 薄灰色：背景
    'grid': '#bdc3c7',        # 灰色：グリッド
    'highlight': '#f39c12',   # オレンジ：ハイライト
    'text': '#2c3e50',        # 濃紺：テキスト
    'axes': '#7f8c8d'         # グレー：軸
}

# 透明度設定
ALPHA = {
    'reference': 0.7,
    'detection': 0.7,
    'waveform': 0.8,
    'grid': 0.3,
    'fill': 0.2
}

# プロットスタイル設定
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.alpha'] = ALPHA['grid']
plt.rcParams['grid.color'] = COLORS['grid']
plt.rcParams['axes.labelcolor'] = COLORS['text']
plt.rcParams['text.color'] = COLORS['text']
plt.rcParams['axes.edgecolor'] = COLORS['axes']
plt.rcParams['xtick.color'] = COLORS['text']
plt.rcParams['ytick.color'] = COLORS['text']


def setup_japanese_font() -> None:
    """
    日本語フォントの設定を行います。
    japanize_matplotlibがインストールされている場合は自動的に設定されます。
    そうでない場合は手動で日本語フォントを設定します。
    """
    try:
        # Macの場合はヒラギノを優先
        if os.name == 'posix':
            # ヒラギノフォントを優先的に使用
            font_candidates = ['Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'Hiragino Maru Gothic Pro', 'AppleGothic']
            for font in font_candidates:
                try:
                    plt.rcParams['font.family'] = font
                    break
                except Exception:
                    continue
        # Windowsの場合
        elif os.name == 'nt':
            plt.rcParams['font.family'] = 'Yu Gothic'
    except Exception:
        # デフォルトのサンセリフフォントを使用
        plt.rcParams['font.family'] = 'sans-serif'


def hz_to_midi(freq: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Hz単位の周波数をMIDIノート番号に変換します。
    
    Parameters
    ----------
    freq : float or ndarray
        Hz単位の周波数
        
    Returns
    -------
    float or ndarray
        MIDIノート番号
    """
    # A4 (440Hz) = MIDIノート番号69
    return 69 + 12 * np.log2(freq / 440.0)


def midi_to_note_name(midi_number: int) -> str:
    """
    MIDIノート番号を音名（例：A4）に変換します。
    
    Parameters
    ----------
    midi_number : int
        MIDIノート番号
        
    Returns
    -------
    str
        音名（例：A4）
    """
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = notes[midi_number % 12]
    return f"{note}{octave}"


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    辞書から安全にキーの値を取得します。キーが存在しない場合はデフォルト値を返します。
    
    Parameters
    ----------
    data : Dict[str, Any]
        辞書
    key : str
        キー
    default : Any, optional
        デフォルト値, by default None
        
    Returns
    -------
    Any
        キーに対応する値、またはデフォルト値
    """
    return data.get(key, default)


def setup_plot_style(ax: Axes, title: str = None) -> None:
    """
    プロットのスタイルを設定します。
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        プロットの軸
    title : str, optional
        タイトル, by default None
    """
    if title:
        ax.set_title(title)
    ax.set_facecolor(COLORS['background'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['axes'])
    ax.spines['bottom'].set_color(COLORS['axes'])
    ax.tick_params(colors=COLORS['text'])


def plot_waveform(ax: Axes, waveform: np.ndarray, sr: int, title: str = "波形") -> None:
    """
    波形をプロットします。
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        プロットの軸
    waveform : np.ndarray
        波形データ
    sr : int
        サンプリングレート
    title : str, optional
        タイトル, by default "波形"
    """
    # 時間軸の計算
    time = np.arange(len(waveform)) / sr
    
    # 波形プロット
    ax.plot(time, waveform, color=COLORS['waveform'], alpha=ALPHA['waveform'])
    
    # グラフの設定
    setup_plot_style(ax, title)
    ax.set_xlabel('時間 (秒)')
    ax.set_ylabel('振幅')
    ax.set_ylim(-1.1, 1.1)


def plot_pitch_data(ax: Axes, time: np.ndarray, pitch: np.ndarray, 
                   title: str = "ピッチ検出結果", show_midi: bool = True) -> None:
    """
    ピッチデータをプロットします。
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        プロットの軸
    time : np.ndarray
        時間データ
    pitch : np.ndarray
        ピッチデータ (Hz)
    title : str, optional
        タイトル, by default "ピッチ検出結果"
    show_midi : bool, optional
        MIDIノート番号を表示するかどうか, by default True
    """
    valid_idx = pitch > 0  # ピッチが0より大きい箇所を有効とする
    
    # ピッチプロット
    ax.plot(time[valid_idx], pitch[valid_idx], 'o', 
            color=COLORS['detection'], alpha=ALPHA['detection'], markersize=2)
    
    # グラフの設定
    setup_plot_style(ax, title)
    ax.set_xlabel('時間 (秒)')
    
    if show_midi:
        # Hz軸とMIDIノート番号軸の両方を表示
        ax_midi = ax.twinx()
        ax_midi.set_ylabel('周波数 (Hz)')
        
        # ピッチの範囲からMIDIノート番号の範囲を設定
        min_freq = np.min(pitch[valid_idx]) if np.any(valid_idx) else 100
        max_freq = np.max(pitch[valid_idx]) if np.any(valid_idx) else 1000
        
        # 安全マージンを追加
        min_freq = max(20, min_freq * 0.8)
        max_freq = min(8000, max_freq * 1.2)
        
        min_midi = hz_to_midi(min_freq)
        max_midi = hz_to_midi(max_freq)
        
        # MIDIノート番号の軸を整数に制限
        midi_ticks = np.arange(int(min_midi), int(max_midi) + 1)
        midi_tick_freqs = 440 * 2**((midi_ticks - 69) / 12)
        
        # 軸の設定
        ax.set_yscale('log')
        ax.set_ylim(min_freq, max_freq)
        ax_midi.set_ylim(min_midi, max_midi)
        
        # MIDIノート番号と対応する音名を表示
        midi_labels = [f"{int(m)} ({midi_to_note_name(int(m))})" if m % 12 == 0 else str(int(m)) 
                     for m in midi_ticks]
        ax_midi.set_yticks(midi_ticks)
        ax_midi.set_yticklabels(midi_labels)
        
        # スタイル設定
        ax_midi.spines['top'].set_visible(False)
        ax_midi.spines['right'].set_color(COLORS['axes'])
        ax_midi.tick_params(axis='y', colors=COLORS['text'])
    else:
        ax.set_ylabel('周波数 (Hz)')


def plot_detection_results(audio_data: np.ndarray, sr: int, 
                          detection_result: Dict[str, Any], 
                          reference_data: Optional[Dict[str, Any]] = None,
                          title: str = "検出結果", 
                          figsize: Tuple[int, int] = (12, 8),
                          show: bool = True,
                          save_path: Optional[str] = None) -> Figure:
    """
    音声検出結果をプロットします。
    波形と検出されたノートのオンセット/オフセットを表示します。
    参照データが提供された場合は、それも表示します。
    
    Parameters
    ----------
    audio_data : np.ndarray
        音声データ
    sr : int
        サンプリングレート
    detection_result : Dict[str, Any]
        検出結果の辞書
        {'intervals': [[onset, offset], ...], 'note_pitches': [...], 'times': [...], 'freqs': [...]}
    reference_data : Optional[Dict[str, Any]], optional
        参照データの辞書
        {'intervals': [[onset, offset], ...], 'note_pitches': [...]}
    title : str, optional
        プロットのタイトル, by default "検出結果"
    figsize : Tuple[int, int], optional
        図のサイズ, by default (12, 8)
    show : bool, optional
        プロットを表示するかどうか, by default True
    save_path : Optional[str], optional
        保存先のパス, by default None
        
    Returns
    -------
    matplotlib.figure.Figure
        プロットの図オブジェクト
    """
    # 安全に値を取得するヘルパー関数
    def safe_get(d, key, default=None):
        if d is None:
            return default
        return d.get(key, default)
    
    # 図の作成
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=16)
    
    # 評価サマリーがあれば表示
    evaluation_summary = safe_get(detection_result, 'evaluation_summary', None)
    if evaluation_summary:
        fig.text(0.5, 0.94, evaluation_summary, ha='center', fontsize=11)
    
    fig.tight_layout(pad=4.0)
    
    # 波形プロット
    plot_waveform(axes[0], audio_data, sr, "波形")
    
    # 検出結果のプロット（上の波形図にオーバーレイ）
    det_intervals = safe_get(detection_result, 'intervals', np.array([]).reshape(0, 2))
    
    for interval in det_intervals:
        onset, offset = interval[0], interval[1]
        # ノートの区間をハイライト
        axes[0].axvspan(onset, offset, alpha=0.2, color=COLORS['detection'])
        # オンセットとオフセットのマーカー
        axes[0].axvline(x=onset, color=COLORS['detection'], linestyle='-', alpha=0.7)
        axes[0].axvline(x=offset, color=COLORS['detection'], linestyle='--', alpha=0.7)
    
    # 参照データがある場合はプロット
    if reference_data is not None:
        ref_intervals = safe_get(reference_data, 'intervals', np.array([]).reshape(0, 2))
        for interval in ref_intervals:
            # 参照の区間をハイライト
            axes[0].axvspan(interval[0], interval[1], alpha=0.2, color=COLORS['reference'])
            # オンセットとオフセットのマーカー
            axes[0].axvline(x=interval[0], color=COLORS['reference'], linestyle='-', alpha=0.7)
            axes[0].axvline(x=interval[1], color=COLORS['reference'], linestyle='--', alpha=0.7)
    
    # 凡例の追加
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['detection'], alpha=0.2, label='検出区間'),
        Patch(facecolor=COLORS['reference'], alpha=0.2, label='参照区間')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')
    
    # ピッチプロット（検出結果と参照、MIDI風表示）
    det_note_pitches = safe_get(detection_result, 'note_pitches', np.array([]))
    det_freqs = safe_get(detection_result, 'freqs', None)
    det_times = safe_get(detection_result, 'times', None)
    
    # 検出結果のピッチプロット（MIDI風）
    if len(det_intervals) > 0 and len(det_note_pitches) > 0:
        # MIDI風の矩形表示（オンセットからオフセットまで水平線を引く）
        for interval, pitch in zip(det_intervals, det_note_pitches):
            onset, offset = interval[0], interval[1]
            # 矩形表示用の水平線（オンセットからオフセットまで）
            axes[1].plot([onset, offset], [pitch, pitch], '-', 
                       color=COLORS['detection'], alpha=0.7, linewidth=2)
            # オンセット位置にマーカー
            axes[1].plot(onset, pitch, 'o', color=COLORS['detection'], alpha=0.7)
    
    # 連続的なピッチ曲線（freqs）を表示
    if det_freqs is not None and det_times is not None:
        # ピッチ値が有効な部分のみプロット
        valid_idx = det_freqs > 0
        if np.any(valid_idx):
            axes[1].plot(det_times[valid_idx], det_freqs[valid_idx], 
                      color=COLORS['detection'], alpha=0.4, linewidth=1)
    
    # 参照データのピッチプロット
    if reference_data is not None:
        ref_intervals = safe_get(reference_data, 'intervals', np.array([]).reshape(0, 2))
        # 参照データのピッチキーを統一
        ref_pitches = safe_get(reference_data, 'note_pitches', None)
        
        # 古い形式のキーをチェック（後方互換性のため）
        if ref_pitches is None:
            ref_pitches = safe_get(reference_data, 'pitches', None)
        
        if ref_pitches is None and 'detection' in reference_data:
            # detection内部のref_pitchesを確認
            ref_pitches = safe_get(reference_data['detection'], 'ref_pitches', np.array([]))
        
        if ref_pitches is None:  # それでも見つからなければ空配列
            ref_pitches = np.array([])
            
        ref_times = safe_get(reference_data, 'times', None)
        ref_freqs = safe_get(reference_data, 'freqs', None)
        
        # 参照ピッチのMIDI風表示
        if len(ref_intervals) > 0 and len(ref_pitches) > 0:
            for interval, pitch in zip(ref_intervals, ref_pitches):
                onset, offset = interval[0], interval[1]
                axes[1].plot([onset, offset], [pitch, pitch], '-', 
                           color=COLORS['reference'], alpha=0.7, linewidth=2)
                axes[1].plot(onset, pitch, 'o', color=COLORS['reference'], alpha=0.7)
        
        # 参照の連続的なピッチ曲線を表示
        if ref_freqs is not None and ref_times is not None:
            valid_idx = ref_freqs > 0
            if np.any(valid_idx):
                axes[1].plot(ref_times[valid_idx], ref_freqs[valid_idx], 
                          color=COLORS['reference'], alpha=0.4, linewidth=1)
    
    # ピッチプロットの設定
    axes[1].set_xlabel('時間 (秒)')
    axes[1].set_ylabel('周波数 (Hz)')
    axes[1].grid(True, alpha=0.3)
    
    # Y軸の対数スケール
    axes[1].set_yscale('log')
    
    # Y軸の範囲設定（最小値と最大値を自動調整）
    all_pitches = []
    if len(det_note_pitches) > 0:
        all_pitches.extend(det_note_pitches)
    if reference_data is not None and len(ref_pitches) > 0:
        all_pitches.extend(ref_pitches)
    
    if all_pitches:
        min_pitch = max(20, min(all_pitches) * 0.8)  # 最小値は20Hz以上
        max_pitch = min(20000, max(all_pitches) * 1.2)  # 最大値は20kHz以下
        axes[1].set_ylim(min_pitch, max_pitch)
    else:
        # デフォルト範囲
        axes[1].set_ylim(50, 2000)
    
    # プロットの表示または保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig 