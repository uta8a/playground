# --- 必要なインポート ---
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class Node:
    """HNSWのグラフを構成するノードを表すクラス"""

    def __init__(self, vector: np.ndarray, layer_idx: int):
        self.vector = vector
        self.layer_idx = layer_idx
        self.neighborhood: Set["Node"] = set()

    def add_neighborhood(self, q: "Node"):
        if self.layer_idx == q.layer_idx:
            self.neighborhood.add(q)


class HNSW:
    """Hierarchical Navigable Small World (HNSW) のグラフ構造を管理するクラス"""

    def __init__(self, M: int, M_max: int, ef_construction: int, mL: float):
        self.nodes: Set[Node] = set()
        self.highest_layer_num: int = 0
        self.entry_point: Node = None
        self.M = M
        self.M_max = M_max
        self.ef_construction = ef_construction
        self.mL = mL

    def _calc_similarity(self, a: Node, b: Node) -> float:
        v1 = a.vector
        v2 = b.vector
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _select_neighbors(self, q: Node, candidates: Set[Node], m: int) -> Set[Node]:
        return set(
            sorted(candidates, key=lambda x: self._calc_similarity(q, x), reverse=True)[:m]
        )

    def _search_layer(self, q: Node, ep: Node, ef: int) -> Set[Node]:
        nodes_visited = set([ep])
        candidates = set([ep])
        neighbors = set([ep])

        while candidates:
            c = max(candidates, key=lambda x: self._calc_similarity(q, x))
            f = min(neighbors, key=lambda x: self._calc_similarity(q, x))

            if self._calc_similarity(q, c) < self._calc_similarity(q, f):
                break

            for e in c.neighborhood:
                if e not in nodes_visited:
                    nodes_visited.add(e)
                    f = min(neighbors, key=lambda x: self._calc_similarity(q, x))
                    if (
                        self._calc_similarity(q, e) > self._calc_similarity(q, f)
                        or len(neighbors) < ef
                    ):
                        candidates.add(e)
                        neighbors.add(e)
                        if len(neighbors) > ef:
                            f = min(neighbors, key=lambda x: self._calc_similarity(q, x))
                            neighbors.remove(f)

            candidates.remove(c)

        return neighbors

    def knn_search(self, vector: np.ndarray, k: int, ef: int) -> Set[Node]:
        q = Node(vector, 0)
        ep = self.entry_point

        for _ in range(self.highest_layer_num, 0, -1):
            candidates = self._search_layer(q, ep, 1)
            ep = max(candidates, key=lambda x: self._calc_similarity(q, x))

        candidates = self._search_layer(q, ep, ef)
        return self._select_neighbors(q, candidates, k)

    def insert(self, vector: np.ndarray) -> None:
        l_max = math.floor(-math.log(np.random.uniform()) * self.mL)

        if self.entry_point is None:
            for lc in range(l_max, -1, -1):
                q = Node(vector, lc)
                self.nodes.add(q)
                if lc == l_max:
                    self.entry_point = q
            self.highest_layer_num = l_max
            return

        ep = self.entry_point
        candidates = {}

        for lc in range(self.highest_layer_num, l_max, -1):
            q = Node(vector, lc)
            candidates = self._search_layer(q, ep, 1)
            ep = max(candidates, key=lambda x: self._calc_similarity(q, x))

        for lc in range(min(self.highest_layer_num, l_max), -1, -1):
            q = Node(vector, lc)
            self.nodes.add(q)

            candidates = self._search_layer(q, ep, self.ef_construction)
            neighbors = self._select_neighbors(q, candidates, self.M)

            for e in neighbors:
                q.add_neighborhood(e)
                e.add_neighborhood(q)

                if len(e.neighborhood) > self.M_max:
                    e.neighborhood = self._select_neighbors(e, e.neighborhood, self.M_max)

        if l_max > self.highest_layer_num:
            self.entry_point = q
            self.highest_layer_num = l_max


# --- ログ用データ構造 ---
@dataclass
class StepEvent:
    algo: str
    kind: str
    step: int
    payload: Dict[str, Any]


class InstrumentedHNSW(HNSW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dist_count = 0
        self._events: List[StepEvent] = []
        self._active_algo = "hnsw"
        self._active_layer: Optional[int] = None

    def reset_trace(self):
        self._dist_count = 0
        self._events = []
        self._active_layer = None

    def events(self) -> List[StepEvent]:
        return list(self._events)

    def dist_count(self) -> int:
        return self._dist_count

    def _sim(self, q: Node, x: Node, layer: Optional[int] = None, note: str = "") -> float:
        self._dist_count += 1
        self._events.append(
            StepEvent(
                algo=self._active_algo,
                kind="dist",
                step=self._dist_count,
                payload={
                    "q_id": id(q),
                    "x_id": id(x),
                    "layer": layer,
                    "note": note,
                },
            )
        )
        return super()._calc_similarity(q, x)

    def _search_layer_instrumented(self, q: Node, ep: Node, ef: int, layer: Optional[int] = None) -> Set[Node]:
        self._active_layer = layer
        nodes_visited = set([ep])
        candidates = set([ep])
        neighbors = set([ep])

        while candidates:
            c = max(candidates, key=lambda x: self._sim(q, x, layer, "pick_best"))
            f = min(neighbors, key=lambda x: self._sim(q, x, layer, "pick_worst"))

            if self._sim(q, c, layer, "check_stop") < self._sim(q, f, layer, "check_stop"):
                break

            self._events.append(
                StepEvent(
                    algo=self._active_algo,
                    kind="move",
                    step=self._dist_count,
                    payload={"from_id": id(ep), "to_id": id(c), "layer": layer},
                )
            )

            for e in c.neighborhood:
                if e not in nodes_visited:
                    nodes_visited.add(e)
                    f = min(neighbors, key=lambda x: self._sim(q, x, layer, "update"))
                    if (self._sim(q, e, layer, "compare") > self._sim(q, f, layer, "compare")) or len(neighbors) < ef:
                        candidates.add(e)
                        neighbors.add(e)
                        if len(neighbors) > ef:
                            f = min(neighbors, key=lambda x: self._sim(q, x, layer, "trim"))
                            neighbors.remove(f)

            candidates.remove(c)
            ep = c

        return neighbors

    def knn_search(self, vector: np.ndarray, k: int, ef: int) -> Set[Node]:
        q = Node(vector, 0)
        ep = self.entry_point

        for lc in range(self.highest_layer_num, 0, -1):
            self._events.append(StepEvent(self._active_algo, "layer_down", self._dist_count, {"from": lc, "to": lc-1}))
            candidates = self._search_layer_instrumented(q, ep, 1, layer=lc)
            ep = max(candidates, key=lambda x: self._sim(q, x, lc, "choose_ep"))

        candidates = self._search_layer_instrumented(q, ep, ef, layer=0)
        topk = set(sorted(candidates, key=lambda x: self._sim(q, x, 0, "final_sort"), reverse=True)[:k])

        self._events.append(
            StepEvent(self._active_algo, "result", self._dist_count, {"k": k, "result_ids": [id(n) for n in topk]})
        )
        return topk


# --- Manim Scene ---
from manim import *


class CompareLinearVsHNSW(Scene):
    def construct(self):
        # ---- 設定 ----
        INTERVAL = 1/15  # 15 FPSの最小単位（60FPSなら1/60）
        MAX_DURATION = 60.0  # 動画の最大長さ（秒）
        MAX_STEPS = int(MAX_DURATION / INTERVAL)  # 最大ステップ数
        step_count = 0  # ステップカウント

        # ---- データ生成 ----
        np.random.seed(42)
        N = 100
        dim = 2
        points = [np.random.randn(dim) for _ in range(N)]
        points = [p / (np.linalg.norm(p) + 1e-9) * 0.8 for p in points]
        query = np.array([0.7, 0.3], dtype=float)
        query = query / (np.linalg.norm(query) + 1e-9)

        # ---- 実際のHNSWインデックスを構築 ----
        hnsw = InstrumentedHNSW(M=4, M_max=6, ef_construction=10, mL=1/math.log(4))
        for p in points:
            hnsw.insert(p)

        # レイヤーごとにノードを分類
        layers: Dict[int, List[Node]] = {}
        for node in hnsw.nodes:
            layer = node.layer_idx
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)

        # ---- パネル配置 ----
        title = Text("Linear Search vs HNSW", font_size=32)
        title.to_edge(UP)
        self.add(title)

        left_box = Rectangle(width=6.2, height=5.2).to_edge(LEFT).shift(DOWN*0.2)
        right_box = Rectangle(width=6.2, height=5.2).to_edge(RIGHT).shift(DOWN*0.2)
        self.add(left_box, right_box)

        left_label = Text("Linear Search", font_size=26).next_to(left_box, UP)
        right_label = Text("HNSW Graph (Layer 0)", font_size=26).next_to(right_box, UP)
        self.add(left_label, right_label)

        # カウンタ
        lin_counter = Integer(0).scale(0.8)
        hnsw_counter = Integer(0).scale(0.8)
        lin_text = Text("dist =", font_size=22)
        hnsw_text = Text("dist =", font_size=22)

        lin_counter_group = VGroup(lin_text, lin_counter).arrange(RIGHT, buff=0.2).next_to(left_box, DOWN)
        hnsw_counter_group = VGroup(hnsw_text, hnsw_counter).arrange(RIGHT, buff=0.2).next_to(right_box, DOWN)
        self.add(lin_counter_group, hnsw_counter_group)

        # ---- 座標変換関数 ----
        def to_panel_coords(p: np.ndarray, panel_rect: Rectangle, scale=2.5) -> np.ndarray:
            x, y = p[0], p[1]
            cx, cy = panel_rect.get_center()[0], panel_rect.get_center()[1]
            return np.array([cx + x*scale, cy + y*scale, 0.0])

        # ---- 左パネル: 線形探索用の点群 ----
        left_dots = []
        for i, p in enumerate(points):
            d = Dot(point=to_panel_coords(p, left_box), radius=0.01, color=BLUE)
            left_dots.append(d)
        left_group = VGroup(*left_dots)
        self.add(left_group)

        # 左: クエリ
        q_dot_left = Dot(point=to_panel_coords(query, left_box), radius=0.05, color=RED)
        self.add(q_dot_left)

        # ---- 右パネル: HNSWグラフ（Layer 0、エッジ付き） ----
        layer0_nodes = layers.get(0, [])
        
        node_id_to_dot: Dict[int, Dot] = {}
        right_dots = []
        for node in layer0_nodes:
            pos = to_panel_coords(node.vector, right_box)
            d = Dot(point=pos, radius=0.01, color=BLUE)
            right_dots.append(d)
            node_id_to_dot[id(node)] = d

        # エッジを描画
        edges = []
        drawn_edges: Set[Tuple[int, int]] = set()
        for node in layer0_nodes:
            for neighbor in node.neighborhood:
                if neighbor.layer_idx == 0:
                    edge_key = tuple(sorted([id(node), id(neighbor)]))
                    if edge_key not in drawn_edges:
                        drawn_edges.add(edge_key)
                        pos1 = to_panel_coords(node.vector, right_box)
                        pos2 = to_panel_coords(neighbor.vector, right_box)
                        line = Line(pos1, pos2, stroke_width=1.5, color=GRAY, stroke_opacity=0.6)
                        edges.append(line)

        # エッジを先に追加
        for edge in edges:
            self.add(edge)
        for d in right_dots:
            self.add(d)

        # 右: クエリ
        q_dot_right = Dot(point=to_panel_coords(query, right_box), radius=0.05, color=RED)
        self.add(q_dot_right)

        # ---- ハイライト用 ----
        lin_highlight = Circle(radius=0.06, color=YELLOW).set_stroke(width=3)
        lin_highlight.set_fill(opacity=0)
        lin_highlight.move_to(left_dots[0].get_center())
        self.add(lin_highlight)

        hnsw_highlight = Circle(radius=0.06, color=YELLOW).set_stroke(width=3)
        hnsw_highlight.set_fill(opacity=0)
        if right_dots:
            hnsw_highlight.move_to(right_dots[0].get_center())
        self.add(hnsw_highlight)

        # 探索パス用
        path_lines = VGroup()
        self.add(path_lines)

        # ---- HNSW探索のイベントを取得 ----
        hnsw.reset_trace()
        _ = hnsw.knn_search(query, k=1, ef=5)
        hnsw_events = hnsw.events()

        # ---- アニメーション: 線形探索 ----
        lin_step = 0
        for i, p in enumerate(points):
            if step_count >= MAX_STEPS:
                break
            lin_step += 1
            step_count += 1
            self.play(
                lin_counter.animate.set_value(lin_step),
                lin_highlight.animate.move_to(left_dots[i].get_center()),
                run_time=INTERVAL
            )

        # 最も近い点をハイライト
        best_idx = 0
        best_sim = -1
        for i, p in enumerate(points):
            sim = np.dot(p, query) / (np.linalg.norm(p) * np.linalg.norm(query))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        
        left_dots[best_idx].set_color(GREEN)
        if step_count < MAX_STEPS:
            self.play(lin_highlight.animate.move_to(left_dots[best_idx].get_center()), run_time=0.3)
            step_count += int(0.3 / INTERVAL) + 1
            self.wait(0.3)
            step_count += int(0.3 / INTERVAL) + 1

        # ---- HNSW探索のアニメーション ----
        hnsw_step = 0
        prev_node_id = None
        
        for event in hnsw_events:
            if step_count >= MAX_STEPS:
                break
            if event.kind == "dist":
                hnsw_step += 1
                x_id = event.payload.get("x_id")
                if x_id in node_id_to_dot:
                    target_dot = node_id_to_dot[x_id]
                    
                    if prev_node_id is not None and prev_node_id in node_id_to_dot:
                        prev_dot = node_id_to_dot[prev_node_id]
                        path_line = Line(
                            prev_dot.get_center(), 
                            target_dot.get_center(),
                            stroke_width=2,
                            color=ORANGE
                        )
                        path_lines.add(path_line)
                        self.play(
                            Create(path_line),
                            hnsw_counter.animate.set_value(hnsw_step),
                            hnsw_highlight.animate.move_to(target_dot.get_center()),
                            run_time=INTERVAL
                        )
                    else:
                        self.play(
                            hnsw_counter.animate.set_value(hnsw_step),
                            hnsw_highlight.animate.move_to(target_dot.get_center()),
                            run_time=INTERVAL
                        )
                    step_count += 1
                    prev_node_id = x_id

        if step_count < MAX_STEPS:
            self.wait(0.3)
            step_count += int(0.3 / INTERVAL) + 1
        
        # ---- 最終比較テキスト ----
        final_text = Text(
            f"Linear: {lin_step} dist  vs  HNSW: {hnsw_step} dist",
            font_size=24
        ).to_edge(DOWN)
        if step_count < MAX_STEPS:
            self.play(Write(final_text), run_time=0.5)
            self.wait(min(1.0, (MAX_STEPS - step_count) * INTERVAL))
