"""
コード変更影響グラフ生成モジュール

このモジュールはコード変更の影響を分析して可視化する機能を提供します。
AST解析を使用して依存関係を抽出し、ネットワークグラフで表示します。
"""

import ast
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
import difflib
import logging

# ロギングの設定
logger = logging.getLogger(__name__)

class CodeChangeAnalyzer:
    """コード変更の影響を分析して可視化するためのクラス"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.changed_elements = set()
        self.original_elements = set()
        self.improved_elements = set()
    
    def analyze_code_change(self, original_code: str, improved_code: str) -> nx.DiGraph:
        """オリジナルコードと改善コードを比較し、変更点とその影響を分析"""
        # AST解析で関数/クラス/変数の依存関係を抽出
        self._build_dependency_graph(original_code, "original")
        self._build_dependency_graph(improved_code, "improved")
        
        # 差分検出
        diff = difflib.unified_diff(
            original_code.splitlines(), improved_code.splitlines(), lineterm=''
        )
        changed_lines = self._parse_diff(diff)
        
        # 変更された要素の特定
        self.changed_elements = self._identify_changed_elements(changed_lines)
        
        # 影響を受ける要素の特定と重み付け
        self._analyze_impact(self.changed_elements)
        
        return self.graph
    
    def visualize_impact(self, output_path: str = None, title: str = "Code Change Impact"):
        """変更影響グラフを可視化"""
        plt.figure(figsize=(14, 10))
        
        # ノードの位置計算（階層レイアウト）
        pos = nx.spring_layout(self.graph)
        
        # ノードグループ（オリジナル/変更/影響）ごとに色分け
        original_nodes = [n for n, d in self.graph.nodes(data=True) 
                         if d.get('type') == 'original' and n not in self.changed_elements]
        changed_nodes = list(self.changed_elements)
        impacted_nodes = [n for n, d in self.graph.nodes(data=True) 
                         if d.get('impact_level', 0) > 0 and n not in self.changed_elements]
        
        # 影響度に基づくエッジの太さと色の設定
        edge_widths = [d.get('impact_weight', 1) for u, v, d in self.graph.edges(data=True)]
        
        # グラフ描画
        nx.draw_networkx_nodes(self.graph, pos, nodelist=original_nodes, node_color='lightblue', node_size=300)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=changed_nodes, node_color='red', node_size=500)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=impacted_nodes, node_color='orange', node_size=400)
        nx.draw_networkx_edges(self.graph, pos, width=edge_widths, edge_color='gray', alpha=0.7)
        nx.draw_networkx_labels(self.graph, pos)
        
        plt.title(title)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"影響グラフを保存しました: {output_path}")
            return output_path
        else:
            plt.show()
            logger.info("影響グラフを表示しました")
            return None
    
    def _build_dependency_graph(self, code: str, code_type: str):
        """ASTを使用してコードの依存関係グラフを構築"""
        try:
            tree = ast.parse(code)
            analyzer = DependencyVisitor(code_type)
            analyzer.visit(tree)
            
            # ノードの追加
            for node_name, node_info in analyzer.nodes.items():
                if node_name not in self.graph:
                    self.graph.add_node(node_name, **node_info)
                else:
                    # 既存ノードの属性を更新
                    for key, value in node_info.items():
                        self.graph.nodes[node_name][key] = value
            
            # エッジの追加
            for source, targets in analyzer.edges.items():
                for target in targets:
                    if source in self.graph and target in self.graph:
                        self.graph.add_edge(source, target, relationship="depends_on")
            
            # 要素セットに追加
            if code_type == "original":
                self.original_elements.update(analyzer.nodes.keys())
            else:
                self.improved_elements.update(analyzer.nodes.keys())
                
        except SyntaxError as e:
            logger.error(f"コードの構文解析に失敗しました: {e}")
    
    def _parse_diff(self, diff) -> List[Tuple[int, str, str]]:
        """差分情報を解析して変更行を抽出"""
        changed_lines = []
        current_file = None
        line_num = 0
        
        for line in diff:
            if line.startswith('+++') or line.startswith('---'):
                continue
            elif line.startswith('@@'):
                # 行番号情報の解析
                parts = line.split(' ')
                if len(parts) >= 2:
                    line_info = parts[1]
                    if line_info.startswith('-'):
                        line_num = abs(int(line_info.split(',')[0]))
                    else:
                        line_num = 0
            elif line.startswith('+'):
                changed_lines.append((line_num, 'add', line[1:]))
                line_num += 1
            elif line.startswith('-'):
                changed_lines.append((line_num, 'remove', line[1:]))
                # 削除行はカウントしない
            else:
                line_num += 1
        
        return changed_lines
    
    def _identify_changed_elements(self, changed_lines: List[Tuple[int, str, str]]) -> Set[str]:
        """変更行から変更された要素（関数、クラス、変数）を特定"""
        changed_elements = set()
        
        # 変更行に含まれる要素を特定
        for line_num, change_type, line_content in changed_lines:
            line_content = line_content.strip()
            
            # クラス定義の変更
            if line_content.startswith('class '):
                class_name = line_content.split('class ')[1].split('(')[0].strip()
                changed_elements.add(class_name)
            
            # 関数定義の変更
            elif line_content.startswith('def '):
                func_name = line_content.split('def ')[1].split('(')[0].strip()
                changed_elements.add(func_name)
            
            # クラスメソッドやインスタンス変数の変更
            elif "self." in line_content:
                for part in line_content.split("self.")[1:]:
                    attr_name = part.split('(')[0].split(' ')[0].split('=')[0].strip()
                    if attr_name:
                        changed_elements.add(attr_name)
        
        # グラフ内の要素と照合
        final_changed_elements = set()
        for element in changed_elements:
            for node in self.graph.nodes():
                if element in node:
                    final_changed_elements.add(node)
        
        return final_changed_elements
    
    def _analyze_impact(self, changed_elements: Set[str]):
        """変更された要素の影響を分析し、グラフに反映"""
        # 各変更要素から到達可能なノードを特定
        impacted_elements = set()
        
        for element in changed_elements:
            # BFSで影響を受ける要素を特定
            if element in self.graph:
                descendants = nx.descendants(self.graph, element)
                impacted_elements.update(descendants)
                
                # 変更要素自身に影響レベルを設定
                self.graph.nodes[element]['impact_level'] = 3
                self.graph.nodes[element]['changed'] = True
        
        # 影響を受ける要素に影響レベルを設定
        for element in impacted_elements:
            # 直接の依存関係ほど影響が大きい
            distance = 1
            for changed in changed_elements:
                if changed in self.graph and element in self.graph:
                    try:
                        path_length = nx.shortest_path_length(self.graph, changed, element)
                        distance = min(distance, path_length)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
            
            # 距離に基づく影響レベルの計算（近いほど値が大きい）
            impact_level = max(3 - distance + 1, 0)
            self.graph.nodes[element]['impact_level'] = impact_level
            
            # エッジの重みを影響レベルに応じて設定
            for u, v, d in self.graph.in_edges(element, data=True):
                d['impact_weight'] = impact_level / 3.0 * 2 + 1  # 1～3の範囲でスケーリング


class DependencyVisitor(ast.NodeVisitor):
    """ASTを巡回して依存関係を抽出するビジターパターン実装"""
    
    def __init__(self, code_type):
        self.nodes = {}  # ノード名とその属性 {name: {attributes}}
        self.edges = {}  # 依存関係 {source: [targets]}
        self.current_context = []  # コンテキストスタック
        self.code_type = code_type
    
    def visit_ClassDef(self, node):
        class_name = node.name
        self.nodes[class_name] = {
            'type': self.code_type,
            'node_type': 'class',
            'lineno': node.lineno
        }
        
        # スコープスタックに現在のクラスを追加
        self.current_context.append(class_name)
        
        # 基底クラスとの依存関係を追加
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_name = base.id
                self._add_dependency(class_name, base_name)
        
        # クラス本体を巡回
        for item in node.body:
            self.visit(item)
        
        # スコープから抜ける
        self.current_context.pop()
    
    def visit_FunctionDef(self, node):
        # 現在のコンテキストに応じて関数名を修飾
        if self.current_context:
            parent = self.current_context[-1]
            func_name = f"{parent}.{node.name}"
        else:
            func_name = node.name
        
        self.nodes[func_name] = {
            'type': self.code_type,
            'node_type': 'function',
            'lineno': node.lineno
        }
        
        # スコープスタックに現在の関数を追加
        self.current_context.append(func_name)
        
        # 関数内で使用されている名前を収集
        for item in node.body:
            self.visit(item)
        
        # スコープから抜ける
        self.current_context.pop()
    
    def visit_Attribute(self, node):
        # インスタンス変数やメソッド呼び出しを検出
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            attr_name = node.attr
            
            # 現在のコンテキストから親クラスを取得
            if self.current_context:
                for ctx in reversed(self.current_context):
                    if '.' not in ctx:  # クラス名として扱う
                        class_name = ctx
                        full_attr_name = f"{class_name}.{attr_name}"
                        
                        # 属性をノードとして追加
                        if full_attr_name not in self.nodes:
                            self.nodes[full_attr_name] = {
                                'type': self.code_type,
                                'node_type': 'attribute',
                                'lineno': getattr(node, 'lineno', 0)
                            }
                        
                        # 現在のコンテキストと属性の間に依存関係を追加
                        current = self.current_context[-1]
                        self._add_dependency(current, full_attr_name)
                        break
        
        # 子ノードを再帰的に巡回
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # 関数呼び出しの依存関係を抽出
        if self.current_context:
            current = self.current_context[-1]
            
            # 直接の関数名呼び出し
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                # 現在のコンテキストから関数呼び出しへの依存関係を追加
                self._add_dependency(current, func_name)
            
            # メソッド呼び出し (obj.method())
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    obj_name = node.func.value.id
                    method_name = node.func.attr
                    
                    # 'self'メソッド呼び出しの場合
                    if obj_name == 'self':
                        for ctx in reversed(self.current_context):
                            if '.' not in ctx:  # クラス名として扱う
                                class_name = ctx
                                full_method_name = f"{class_name}.{method_name}"
                                self._add_dependency(current, full_method_name)
                                break
                    else:
                        # 外部オブジェクトのメソッド呼び出し
                        full_method_name = f"{obj_name}.{method_name}"
                        self._add_dependency(current, full_method_name)
        
        # 子ノードを再帰的に巡回
        self.generic_visit(node)
    
    def _add_dependency(self, source, target):
        """依存関係を追加"""
        if source not in self.edges:
            self.edges[source] = set()
        self.edges[source].add(target) 