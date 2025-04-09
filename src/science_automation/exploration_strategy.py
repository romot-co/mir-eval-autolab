# src/science_automation/exploration_strategy.py

import logging
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ExplorationStrategy:
    """
    アルゴリズム改善の探索戦略を管理し、提案するクラス。
    セッション履歴とパフォーマンスに基づいて、活用(exploitation)と探索(exploration)の
    フェーズを切り替え、次のアクションを提案します。
    """

    def __init__(self, history: List[Dict[str, Any]], current_performance: float = 0.0):
        """
        探索戦略を初期化します。

        Args:
            history: セッション履歴のイベントリスト。
            current_performance: 現在のパフォーマンス指標（例: F値）。
        """
        self.history = history or []
        self.current_performance = current_performance
        self.stagnation_count = 0
        # 履歴から現在のフェーズと戦略履歴を復元する方が望ましいが、ここでは単純化
        self.exploration_phase = "exploitation"  # 初期フェーズは活用
        self.strategy_history: List[Dict[str, Any]] = [] # 実行した戦略アクションの履歴
        self.last_hypothesis: Optional[Dict[str, Any]] = None
        self._load_state_from_history() # 履歴から状態を復元

    def _load_state_from_history(self):
        """履歴から停滞カウントや戦略履歴などを復元する（簡易版）"""
        stagnation_threshold = 0.005 # 改善とみなす閾値
        last_perf = 0.0
        perf_history = []

        # 評価履歴からパフォーマンス推移を抽出
        for event in self.history:
            if event.get("type") == "evaluation_complete":
                data = event.get("data", {})
                summary = data.get("summary")
                if isinstance(summary, dict):
                    detector_name = next(iter(summary), None)
                    if detector_name:
                        note_metrics = summary[detector_name].get("note")
                        if isinstance(note_metrics, dict) and "f_measure" in note_metrics:
                            perf_history.append(note_metrics["f_measure"])

        # 停滞カウントを計算
        if len(perf_history) > 1:
            for i in range(len(perf_history) - 1, 0, -1):
                if perf_history[i] - perf_history[i-1] < stagnation_threshold:
                    self.stagnation_count += 1
                else:
                    break # 改善が見られたらカウント停止
            self.current_performance = perf_history[-1] # 最新のパフォーマンスをセット

        # 戦略変更履歴をロード
        for event in self.history:
             if event.get("type") == "strategy_change":
                 strategy_data = event.get("data", {}).get("strategy")
                 if isinstance(strategy_data, dict):
                     self.strategy_history.append(strategy_data)
                     # 最後に提案された戦略に基づいてフェーズを設定
                     self.exploration_phase = strategy_data.get("phase", self.exploration_phase)

        logger.debug(f"Strategy state loaded: stagnation={self.stagnation_count}, phase='{self.exploration_phase}', current_perf={self.current_performance:.4f}")


    def detect_stagnation(self, new_performance: float, threshold: float = 0.005) -> bool:
        """
        パフォーマンスの停滞を検出します。

        Args:
            new_performance: 新しいパフォーマンス指標。
            threshold: 改善と見なす閾値。

        Returns:
            bool: 停滞が検出されたかどうか。
        """
        improvement = new_performance - self.current_performance

        if improvement < threshold:
            self.stagnation_count += 1
            logger.info(f"パフォーマンス停滞検出: 改善 {improvement:.4f} < {threshold:.4f} (連続 {self.stagnation_count} 回)")
        else:
            self.stagnation_count = 0
            logger.info(f"パフォーマンス改善検出: 改善 {improvement:.4f} >= {threshold:.4f}")

        self.current_performance = new_performance
        # 停滞と判定する回数を設定 (例: 3回)
        stagnation_limit = 3
        is_stagnating = self.stagnation_count >= stagnation_limit
        if is_stagnating:
            logger.warning(f"パフォーマンスが {stagnation_limit} 回連続で停滞しています。")
        return is_stagnating

    def extract_recent_improvements(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        直近の改善履歴（LLMによる変更概要）を抽出します。

        Args:
            limit: 取得する履歴の最大数。

        Returns:
            List[Dict[str, Any]]: 改善履歴のリスト。
        """
        improvements = []
        for event in reversed(self.history):
            # 修正: improvement_iteration_completed から取得
            if event.get("type") == "improvement_iteration_completed" and len(improvements) < limit:
                data = event.get("data", {}).get("improvement_data", {})
                changes_summary = data.get("changes_summary")
                if changes_summary:
                    improvements.append({
                        "timestamp": event.get("timestamp", 0),
                        "iteration": data.get("iteration_number"),
                        "changes_summary": changes_summary,
                        "f_measure_change": data.get("improvement", 0.0)
                    })
        return list(reversed(improvements)) # 時系列順に戻す

    def suggest_strategy_change(self) -> Dict[str, Any]:
        """
        現在の状態に基づいて戦略変更を提案します。
        停滞した場合に呼び出されます。

        Returns:
            Dict[str, Any]: 次のアクションを含む戦略変更の提案。
                           例: {"phase": "exploration", "action": "generate_hypothesis", ...}
        """
        recent_improvements = self.extract_recent_improvements()
        context_summary = "\n".join([f"- Iter {imp.get('iteration', '?')}: {imp.get('changes_summary', '')} ({imp.get('f_measure_change'):+.4f})" for imp in recent_improvements])

        if self.exploration_phase == "exploitation":
            # 活用フェーズで停滞 -> 探索フェーズへ移行し、まずは仮説生成
            self.exploration_phase = "exploration"
            next_action = "generate_hypothesis"
            explanation = "活用フェーズでの改善が停滞したため、探索フェーズに移行し、新しい仮説を生成します。"
        else: # exploration フェーズで停滞
            # 異なる探索戦略を試す
            available_exploration_actions = [
                {"name": "parameter_optimization", "action": "optimize_parameters", "explanation": "有望そうなパラメータ空間を探索します。"},
                {"name": "focused_code_improvement", "action": "improve_code_focused", "explanation": "特定の弱点や仮説に基づいてコード改善を行います。"},
                # {"name": "algorithm_redesign", "action": "redesign_algorithm", "explanation": "アルゴリズムのコア部分を再設計します"}, # これはLLMには難しいかも
                # {"name": "feature_engineering", "action": "engineer_features", "explanation": "特徴量の工学的設計に焦点を当てます"}, # 同上
            ]

            # 試行済みの戦略を除外
            tried_actions = [s.get("action") for s in self.strategy_history if s.get("phase") == "exploration"]
            next_strategies = [s for s in available_exploration_actions if s["action"] not in tried_actions]

            if not next_strategies:
                # 一周したらパラメータ最適化に戻るか、活用フェーズに戻る
                logger.warning("全ての探索戦略を試しました。パラメータ最適化に戻ります。")
                next_action = "optimize_parameters"
                explanation = "全ての探索戦略を試したため、再度パラメータ最適化を行います。"
                # self.exploration_phase = "exploitation" # オプション: 活用に戻る
            else:
                selected_strategy = next_strategies[0]
                next_action = selected_strategy["action"]
                explanation = f"新しい探索アプローチ '{selected_strategy['name']}' を試みます: {selected_strategy['explanation']}"

        strategy = {
            "phase": self.exploration_phase,
            "action": next_action,
            "explanation": explanation,
            "context_summary": context_summary
        }
        self.strategy_history.append(strategy) # 提案された戦略を記録
        logger.info(f"戦略変更を提案: Action='{strategy['action']}' - {strategy['explanation']}")
        return strategy

    def set_hypothesis(self, hypothesis: Optional[Dict[str, Any]]):
        """
        現在検証中の仮説を設定またはクリアします。

        Args:
            hypothesis: 検証する仮説の情報辞書、または None。
        """
        self.last_hypothesis = hypothesis
        if hypothesis:
            logger.info(f"新しい仮説を設定: {hypothesis.get('title', 'タイトルなし')}")
        else:
             logger.info("現在の仮説をクリアしました。")

    def get_current_hypothesis(self) -> Optional[Dict[str, Any]]:
        """
        現在検証中の仮説を取得します。

        Returns:
            Optional[Dict[str, Any]]: 現在の仮説、または存在しない場合は None。
        """
        return self.last_hypothesis

    def get_next_action(self, new_performance: Optional[float] = None) -> Dict[str, Any]:
        """
        次の改善アクションを決定します。

        Args:
            new_performance: 最新の評価パフォーマンス（指定された場合、停滞検出に使用）。

        Returns:
            Dict[str, Any]: 次のアクション提案。
                           例: {"action": "improve_code", "phase": "exploitation", ...}
        """
        if new_performance is not None:
            is_stagnating = self.detect_stagnation(new_performance)
            if is_stagnating:
                # 停滞していたら戦略を変更
                return self.suggest_strategy_change()

        # 停滞していない、またはパフォーマンス情報がない場合
        if self.exploration_phase == "exploitation":
            # 活用フェーズ：通常のコード改善
            return {
                "phase": "exploitation",
                "action": "improve_code", # 標準的な改善アクション
                "explanation": "現在の戦略で改善を試みます（活用フェーズ）。",
                "hypothesis": self.get_current_hypothesis() # 現在の仮説があれば含める
            }
        else: # exploration フェーズ
            # 前回提案された戦略アクションがあればそれを返す
            if self.strategy_history:
                 last_suggested = self.strategy_history[-1]
                 # 同じアクションを繰り返さないように考慮が必要かもしれないが、まずは前回提案を返す
                 # ただし、仮説生成後は focused_code_improvement に移行すべき
                 if last_suggested.get("action") == "generate_hypothesis":
                      # 仮説生成の次は、その仮説に基づいた改善を行う
                      focused_action = {
                           "phase": "exploration",
                           "action": "improve_code_focused",
                           "explanation": "生成された仮説に基づいてコード改善を行います。",
                           "hypothesis": self.get_current_hypothesis()
                      }
                      self.strategy_history.append(focused_action) # 実行するアクションも記録
                      return focused_action
                 else:
                    return last_suggested
            else:
                # 履歴がない場合はデフォルトで仮説生成から開始
                return self.suggest_strategy_change()