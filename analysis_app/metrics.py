"""确定性聚类评估工具：面积加权统计、轮廓系数、标签对齐与调整后兰德指数。"""
import math
from collections import Counter


def weighted_mean(values, weights):
    total = math.fsum(weights)
    if total <= 0:
        raise ValueError('权重总和必须为正')
    return math.fsum(value * weight for value, weight in zip(values, weights)) / total


def weighted_variance(values, weights):
    mean = weighted_mean(values, weights)
    total = math.fsum(weights)
    return math.fsum(weight * (value - mean) ** 2 for value, weight in zip(values, weights)) / total


def weighted_median(values, weights):
    pairs = sorted(zip(values, weights))
    half = math.fsum(weights) / 2
    accrued = 0.0
    for value, weight in pairs:
        accrued += weight
        if accrued >= half:
            return value
    return pairs[-1][0]


def euclidean(first, second):
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(first, second)))


def squared_distance(first, second):
    return math.fsum((a - b) ** 2 for a, b in zip(first, second))


def silhouette_samples(vectors, labels, weights):
    """逐样本轮廓系数；权重为网格代表面积。单成员簇按约定记 0。"""
    n = len(vectors)
    clusters = sorted(set(labels))
    if len(clusters) < 2:
        return [0.0] * n
    distance = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            value = euclidean(vectors[i], vectors[j])
            distance[i][j] = value
            distance[j][i] = value
    samples = []
    for i in range(n):
        own = labels[i]
        own_weight = math.fsum(weights[j] for j in range(n) if labels[j] == own and j != i)
        if own_weight <= 0:
            samples.append(0.0)
            continue
        a = math.fsum(weights[j] * distance[i][j]
                      for j in range(n) if labels[j] == own and j != i) / own_weight
        b = None
        for cluster in clusters:
            if cluster == own:
                continue
            cluster_weight = math.fsum(weights[j] for j in range(n) if labels[j] == cluster)
            if cluster_weight <= 0:
                continue
            mean_distance = math.fsum(weights[j] * distance[i][j]
                                      for j in range(n) if labels[j] == cluster) / cluster_weight
            b = mean_distance if b is None else min(b, mean_distance)
        if b is None:
            samples.append(0.0)
            continue
        denominator = max(a, b)
        samples.append((b - a) / denominator if denominator > 0 else 0.0)
    return samples


def adjusted_rand(reference, target):
    """调整后兰德指数。两侧同为单一簇或同为全互异时划分等价，记 1。"""
    if len(reference) != len(target):
        raise ValueError('标签长度不一致')
    n = len(reference)
    if n == 0:
        return 1.0

    def comb2(count):
        return count * (count - 1) / 2

    total = comb2(n)
    if total == 0:
        return 1.0
    cells = Counter(zip(reference, target))
    sum_cells = math.fsum(comb2(count) for count in cells.values())
    sum_ref = math.fsum(comb2(count) for count in Counter(reference).values())
    sum_tgt = math.fsum(comb2(count) for count in Counter(target).values())
    expected = sum_ref * sum_tgt / total
    denominator = (sum_ref + sum_tgt) / 2 - expected
    if denominator == 0:
        return 1.0
    return (sum_cells - expected) / denominator


def hungarian_max(table):
    """最大权二分匹配（匈牙利算法），返回 {行号: 列号}（0 起始）；非方阵按零补齐。"""
    rows = len(table)
    cols = len(table[0]) if rows else 0
    size = max(rows, cols)
    cost = [[0.0] * (size + 1) for _ in range(size + 1)]
    for i in range(rows):
        for j in range(cols):
            cost[i + 1][j + 1] = -float(table[i][j])
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0][j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]
    assignment = {}
    for j in range(1, size + 1):
        if p[j] and p[j] <= rows and j <= cols:
            assignment[p[j] - 1] = j - 1
    return assignment


def align_labels(reference, target):
    """把 target 的簇编号对齐到基准 reference，返回 {target标签: reference标签}。

    重采样得到的簇编号可能对调，必须先在混淆矩阵上做最大权匹配，
    生态人员才能追踪同一类；未匹配到的 target 标签不出现在映射中。
    """
    ref_labels = sorted(set(reference))
    tgt_labels = sorted(set(target))
    ref_index = {label: i for i, label in enumerate(ref_labels)}
    tgt_index = {label: i for i, label in enumerate(tgt_labels)}
    table = [[0] * len(tgt_labels) for _ in ref_labels]
    for ref, tgt in zip(reference, target):
        table[ref_index[ref]][tgt_index[tgt]] += 1
    assignment = hungarian_max(table)
    return {tgt_labels[column]: ref_labels[row] for row, column in assignment.items()}


def aligned_agreement(reference, target):
    """target 对齐到 reference 之后的一致率。"""
    if not reference:
        return 1.0
    mapping = align_labels(reference, target)
    matched = sum(1 for ref, tgt in zip(reference, target) if mapping.get(tgt) == ref)
    return matched / len(reference)
