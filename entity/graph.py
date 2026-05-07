import os
import math
import itertools
import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_comm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class WeightedGraph:
    def __init__(self, graph_id="root"):
        self.id = graph_id
        self.nodes = set()
        self.edges = {} # Key: tuple(sorted((u, v))), Value: weight

    def _get_edge_key(self, u, v):
        return tuple(sorted((u, v)))

    def add_edge(self, u, v, weight=1):
        if u == v: return # 忽略自环
        self.nodes.add(u)
        self.nodes.add(v)
        key = self._get_edge_key(u, v)
        self.edges[key] = self.edges.get(key, 0) + weight

    def build_from_list(self, keywords):
        """从关键词列表构建全连接子图"""
        if len(keywords) < 2:
            for k in keywords: self.nodes.add(k)
            return
        for u, v in itertools.combinations(keywords, 2):
            self.add_edge(u, v, weight=1)

    def merge_from(self, other_graph):
        """合并另一个图对象"""
        if not other_graph: return
        self.nodes.update(other_graph.nodes)
        for (u, v), weight in other_graph.edges.items():
            key = (u, v)
            self.edges[key] = self.edges.get(key, 0) + weight

    def visualize(self, output_file):
        """CiteSpace 风格可视化 — 黑色背景、年轮节点、紫色中心性环、白色标签"""
        if not self.nodes:
            print("图为空，跳过可视化。")
            return

        print(f"正在生成可视化图谱 (CiteSpace 风格): {output_file}")

        # ── 字体设置 ──
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

        # ── 1. 构建 NetworkX 图 ──
        G = nx.Graph()
        for (u, v), w in self.edges.items():
            G.add_edge(u, v, weight=w)

        if len(G.nodes()) == 0:
            print("NetworkX 图为空，跳过可视化。")
            return

        # ── 2. 计算图指标 ──
        weighted_degrees = dict(G.degree(weight='weight'))
        max_wd = max(weighted_degrees.values()) if weighted_degrees else 1

        # 介数中心性
        centrality = nx.betweenness_centrality(G, weight='weight')

        # 社区发现 (Louvain)
        try:
            communities = list(nx_comm.louvain_communities(
                G, weight='weight', resolution=1.5, seed=42))
        except Exception:
            communities = list(nx_comm.greedy_modularity_communities(
                G, weight='weight'))

        # 按社区大小排序（大社区在前，分配更显眼的颜色）
        communities.sort(key=len, reverse=True)

        # ── 3. CiteSpace 配色 ──
        CITESPACE_COLORS = [
            '#E53935',  # 红
            '#1E88E5',  # 蓝
            '#43A047',  # 绿
            '#FF9800',  # 橙
            '#8E24AA',  # 紫
            '#00ACC1',  # 青
            '#FDD835',  # 黄
            '#D81B60',  # 粉红
            '#7CB342',  # 黄绿
            '#5E35B1',  # 深紫
            '#00897B',  # 蓝绿
            '#F4511E',  # 深橙
            '#3949AB',  # 靛蓝
            '#C0CA33',  # 柠檬绿
            '#6D4C41',  # 棕
        ]

        node_to_color = {}
        node_to_comm_idx = {}
        for i, comm in enumerate(communities):
            color = CITESPACE_COLORS[i % len(CITESPACE_COLORS)]
            for node in comm:
                node_to_color[node] = color
                node_to_comm_idx[node] = i

        # ── 4. 边剪枝：只保留权重前 N 的边，避免密集区糊成一片 ──
        all_weights = sorted([d['weight'] for _, _, d in G.edges(data=True)], reverse=True)
        max_edges_to_draw = min(len(all_weights), max(2000, int(len(G.nodes()) * 8)))
        weight_cutoff = all_weights[max_edges_to_draw - 1] if max_edges_to_draw < len(all_weights) else 0

        # ── 5. 力导向布局 ──
        num_nodes = len(G.nodes())
        k_val = 2.5 / (num_nodes ** 0.5) if num_nodes > 0 else 0.1
        pos = nx.spring_layout(G, k=k_val, iterations=400, seed=42, weight='weight')

        # 将布局居中归一化到 [-1, 1]
        pos_arr = np.array(list(pos.values()))
        center = pos_arr.mean(axis=0)
        scale = np.abs(pos_arr - center).max()
        if scale > 0:
            pos = {n: (p - center) / scale for n, p in pos.items()}

        # 裁剪离群点：将超出 2.5 标准差的极端坐标拉回来
        pos_arr = np.array(list(pos.values()))
        for axis in [0, 1]:
            q1, q3 = np.percentile(pos_arr[:, axis], [5, 95])
            iqr = q3 - q1
            lo, hi = q1 - 2.0 * iqr, q3 + 2.0 * iqr
            for n in pos:
                coords = list(pos[n])
                coords[axis] = max(lo, min(hi, coords[axis]))
                pos[n] = np.array(coords)

        # ── 6. 创建画布 ──
        fig, ax = plt.subplots(figsize=(36, 24), dpi=200)
        fig.patch.set_facecolor('#0A0A0A')
        ax.set_facecolor('#0A0A0A')
        ax.axis('off')

        # ── 7. 绘制边 ──
        max_w = all_weights[0] if all_weights else 1
        for (u, v, d) in G.edges(data=True):
            if d['weight'] < weight_cutoff:
                continue
            x_coords = [pos[u][0], pos[v][0]]
            y_coords = [pos[u][1], pos[v][1]]
            weight_ratio = d['weight'] / max_w

            same_comm = node_to_comm_idx.get(u) == node_to_comm_idx.get(v)
            if same_comm:
                edge_color = node_to_color.get(u, '#555555')
                edge_alpha = 0.12 + weight_ratio * 0.25
                line_width = 0.15 + weight_ratio * 2.0
            else:
                edge_color = '#333333'
                edge_alpha = 0.03 + weight_ratio * 0.10
                line_width = 0.1 + weight_ratio * 0.8

            ax.plot(x_coords, y_coords,
                    color=edge_color, linewidth=line_width,
                    alpha=edge_alpha, solid_capstyle='round', zorder=1)

        # ── 8. 节点尺寸计算 ──
        # 使用 log 压缩极端值，避免个别超大节点
        all_x = [pos[n][0] for n in G.nodes()]
        all_y = [pos[n][1] for n in G.nodes()]
        coord_range = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 0.01)

        log_wds = {n: math.log1p(wd) for n, wd in weighted_degrees.items()}
        max_log_wd = max(log_wds.values()) if log_wds else 1

        base_radius = coord_range * 0.006
        max_extra_radius = coord_range * 0.028

        # ── 9. 绘制节点 (CiteSpace 年轮效果) ──
        for node in G.nodes():
            x, y = pos[node]
            wd_ratio = log_wds[node] / max_log_wd
            radius = base_radius + wd_ratio * max_extra_radius
            comm_color = node_to_color.get(node, '#FFFFFF')

            # 解析颜色并生成深浅变体
            r_c, g_c, b_c = mcolors.to_rgb(comm_color)
            darker = (r_c * 0.4, g_c * 0.4, b_c * 0.4)
            lighter = (min(1, r_c * 1.3), min(1, g_c * 1.3), min(1, b_c * 1.3))

            # 9a. 紫色中心性外圈 (centrality >= 0.03)
            if centrality.get(node, 0) >= 0.03:
                c_val = centrality[node]
                purple_lw = 1.0 + c_val * 15
                purple_ring = plt.Circle(
                    (x, y), radius * 1.35,
                    fill=False, edgecolor='#B040FF',
                    linewidth=purple_lw, alpha=0.75, zorder=2)
                ax.add_patch(purple_ring)

            # 9b. 外发光圈 (社区色微光)
            glow = plt.Circle(
                (x, y), radius * 1.2,
                facecolor=comm_color, edgecolor='none',
                alpha=0.10, zorder=3)
            ax.add_patch(glow)

            # 9c. 外环 (较亮色)
            outer = plt.Circle(
                (x, y), radius,
                facecolor=lighter, edgecolor='#FFFFFF',
                linewidth=0.2, alpha=0.85, zorder=4)
            ax.add_patch(outer)

            # 9d. 中间环 (主色)
            mid = plt.Circle(
                (x, y), radius * 0.72,
                facecolor=comm_color, edgecolor='none',
                alpha=0.90, zorder=5)
            ax.add_patch(mid)

            # 9e. 内环 (深色)
            inner = plt.Circle(
                (x, y), radius * 0.45,
                facecolor=darker, edgecolor='none',
                alpha=0.85, zorder=6)
            ax.add_patch(inner)

            # 9f. 中心点 (最暗)
            core = plt.Circle(
                (x, y), radius * 0.18,
                facecolor='#000000', edgecolor=comm_color,
                linewidth=0.3, alpha=0.9, zorder=7)
            ax.add_patch(core)

        # ── 10. 绘制标签 ──
        # 只标注 top N 重要节点，避免标签覆盖
        sorted_nodes = sorted(G.nodes(),
                              key=lambda n: weighted_degrees[n], reverse=True)
        # 标注数量：节点少则全标，多则限制
        label_count = min(len(sorted_nodes), max(50, int(num_nodes * 0.25)))
        nodes_to_label = set(sorted_nodes[:label_count])

        # 高中心性节点也强制标注
        for n, c in centrality.items():
            if c >= 0.03:
                nodes_to_label.add(n)

        # 简单标签碰撞检测：记录已占用位置，跳过太近的标签
        placed_labels = []  # (x, y, width_est)

        for node in sorted_nodes:  # 按重要性顺序标注，重要的优先
            if node not in nodes_to_label:
                continue

            x, y = pos[node]
            wd_ratio = log_wds[node] / max_log_wd
            radius = base_radius + wd_ratio * max_extra_radius
            fontsize = 5.0 + wd_ratio * 16
            font_weight = 'bold' if wd_ratio > 0.30 else 'normal'

            y_offset = radius + coord_range * 0.006

            # 碰撞检测：估计标签占据的宽度
            label_width_est = len(node) * fontsize * coord_range * 0.00006
            label_y = y + y_offset

            too_close = False
            for px, py, pw in placed_labels:
                if abs(label_y - py) < coord_range * 0.012 and \
                   abs(x - px) < (label_width_est + pw) * 0.6:
                    too_close = True
                    break

            if too_close and wd_ratio < 0.20:
                continue  # 跳过不太重要的重叠标签

            placed_labels.append((x, label_y, label_width_est))

            # 黑色描边（增强可读性）
            for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                ax.text(x + dx * coord_range * 0.0008,
                        label_y + dy * coord_range * 0.0008,
                        node, fontsize=fontsize, color='#000000',
                        fontweight=font_weight, fontfamily='sans-serif',
                        ha='center', va='bottom', alpha=0.85, zorder=8)

            # 主文字（白色）
            ax.text(x, label_y, node,
                    fontsize=fontsize, color='#FFFFFF',
                    fontweight=font_weight, fontfamily='sans-serif',
                    ha='center', va='bottom', alpha=0.95, zorder=9)

        # ── 11. 统计信息 ──
        try:
            modularity_q = nx_comm.modularity(
                G, communities, weight='weight')
        except Exception:
            modularity_q = 0.0

        stats_text = (
            f"N = {len(G.nodes())}    "
            f"E = {len(G.edges())}    "
            f"Q = {modularity_q:.4f}    "
            f"Communities = {len(communities)}"
        )
        ax.text(0.01, 0.01, stats_text,
                transform=ax.transAxes,
                fontsize=8, color='#888888',
                fontfamily='monospace',
                verticalalignment='bottom', zorder=10)

        # ── 12. 自动裁剪留白并保存 ──
        ax.set_xlim(min(all_x) - coord_range * 0.08,
                    max(all_x) + coord_range * 0.08)
        ax.set_ylim(min(all_y) - coord_range * 0.08,
                    max(all_y) + coord_range * 0.12)

        fig.savefig(output_file, bbox_inches='tight',
                    facecolor='#0A0A0A', edgecolor='none', pad_inches=0.3)
        plt.close(fig)
        print(f"CiteSpace 风格图谱已保存: {output_file}")


def test_graph_logic():
    g1 = WeightedGraph("1")
    g1.build_from_list(["apple", "banana", "cherry"])
    
    g2 = WeightedGraph("2")
    g2.build_from_list(["apple", "banana"])
    
    g1.merge_from(g2)
    
    # 验证边权重：apple-banana 应该为 2 (g1贡献1, g2贡献1)
    weight = g1.edges.get(tuple(sorted(("apple", "banana"))))
    assert weight == 2
    print("Graph logic test passed!")

if __name__ == "__main__":
    test_graph_logic()