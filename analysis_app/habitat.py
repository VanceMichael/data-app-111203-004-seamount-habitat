"""海山生境分群：连续地形与离散底质联合聚类，产物可保存并重新载入。

weight 语义：weight 为网格代表面积（平方千米），训练与解释全程按面积加权，
避免调查网格数量代替真实面积。缺值填补、尺度变换与类别字典只在训练海山上拟合。
"""
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from .io import fingerprint, save_json
from .metrics import (euclidean, squared_distance, weighted_mean,
                      weighted_median, weighted_variance)
from .schema import PRODUCT

NUMERIC_FEATURES = ['depth', 'slope', 'roughness']
CATEGORICAL_FEATURE = 'substrate'
UNKNOWN_SUBSTRATE = '未知'
DEFAULT_SEED = 20240611
DEFAULT_N_INIT = 4
DEFAULT_MAX_ITER = 50
DEFAULT_MIN_MARGIN = 0.1
DEFAULT_CATEGORICAL_WEIGHT = 1.0

WEIGHT_SEMANTICS = (
    'weight 为网格代表面积（平方千米）：训练时作为样本权重参与缺值填补、尺度变换、'
    '聚类中心与离散度计算，解释产物中的占比均按代表面积汇总，'
    '避免调查网格数量代替真实面积；同一网格的重复观测合并时代表面积取均值。'
)

ABSTAIN_RULE = (
    '满足任一条件即判为无可靠归属（cluster 为 null）：'
    '出现训练未见的底质类别；最近簇中心距离超过该簇训练期最大距离；'
    '置信边际 (次近距离-最近距离)/次近距离 低于 min_margin。'
)

PREPROCESS_NOTE = '缺值填补、尺度变换与类别字典均只在训练海山上拟合，并随模型产物保存'


def aggregate_cells(records):
    """按网格合并重复观测：数值取均值，底质取众数（并列取字典序较小者），代表面积取均值。"""
    buckets = {}
    for row in sorted(records, key=lambda item: item['observation_id']):
        bucket = buckets.setdefault(row['cell'], {
            'cell': row['cell'], 'seamount': row['seamount'], 'observations': 0,
            'depth': [], 'slope': [], 'roughness': [], 'substrate': [], 'weight': []})
        if row['seamount'] != bucket['seamount']:
            raise ValueError('网格归属海山不一致：' + row['cell'])
        bucket['observations'] += 1
        for name in NUMERIC_FEATURES:
            if row[name] is not None:
                bucket[name].append(row[name])
        bucket['substrate'].append(row[CATEGORICAL_FEATURE] or UNKNOWN_SUBSTRATE)
        if row['weight'] is not None:
            bucket['weight'].append(row['weight'])
    cells = []
    for key in sorted(buckets):
        bucket = buckets[key]
        if not bucket['weight']:
            raise ValueError('网格缺少代表面积：' + key)
        weight = math.fsum(bucket['weight']) / len(bucket['weight'])
        if weight <= 0:
            raise ValueError('代表面积必须为正：' + key)
        substrate = sorted(Counter(bucket['substrate']).items(),
                           key=lambda item: (-item[1], item[0]))[0][0]
        cell = {'cell': key, 'seamount': bucket['seamount'],
                'observations': bucket['observations'],
                'substrate': substrate, 'weight': weight}
        for name in NUMERIC_FEATURES:
            values = bucket[name]
            cell[name] = math.fsum(values) / len(values) if values else None
        cells.append(cell)
    return cells


def fit_preprocessor(cells, categorical_weight=DEFAULT_CATEGORICAL_WEIGHT):
    """在训练网格上拟合缺值填补、尺度变换与类别字典；全部统计按代表面积加权。"""
    imputation = {}
    scaler = {}
    for name in NUMERIC_FEATURES:
        observed = [(cell[name], cell['weight']) for cell in cells if cell[name] is not None]
        if observed:
            values = [value for value, _ in observed]
            weights = [weight for _, weight in observed]
            median = weighted_median(values, weights)
            mean = weighted_mean(values, weights)
            variance = weighted_variance(values, weights)
        else:
            median, mean, variance = 0.0, 0.0, 0.0
        std = math.sqrt(variance)
        imputation[name] = median
        scaler[name] = {'mean': mean, 'std': std if std > 0 else 1.0}
    dictionary = sorted({cell['substrate'] for cell in cells})
    return {'imputation': imputation, 'scaler': scaler, 'dictionary': dictionary,
            'categorical_weight': categorical_weight, 'note': PREPROCESS_NOTE}


def transform_cell(pre, cell):
    """把合并后的网格变换为特征向量；返回 (向量, 信息标记)。"""
    vector = []
    flags = []
    for name in NUMERIC_FEATURES:
        value = cell[name]
        if value is None:
            value = pre['imputation'][name]
            flags.append('缺值已填补:' + name)
        scaler = pre['scaler'][name]
        vector.append((value - scaler['mean']) / scaler['std'])
    dictionary = pre['dictionary']
    substrate = cell['substrate']
    if substrate in dictionary:
        vector.extend(pre['categorical_weight'] if item == substrate else 0.0
                      for item in dictionary)
    else:
        vector.extend([0.0] * len(dictionary))
        flags.append('未见底质类别:' + str(substrate))
    return vector, flags


def _weighted_pick(weights, rng):
    total = math.fsum(weights)
    if total <= 0:
        return max(range(len(weights)), key=lambda i: (weights[i], -i))
    marker = rng.random() * total
    accrued = 0.0
    for i, weight in enumerate(weights):
        accrued += weight
        if accrued > marker:
            return i
    return len(weights) - 1


def _kmeanspp(vectors, weights, k, rng):
    """面积加权的 k-means++ 初始化；种子固定，结果确定。"""
    centers = [list(vectors[_weighted_pick(weights, rng)])]
    d2 = [squared_distance(vector, centers[0]) for vector in vectors]
    while len(centers) < k:
        probs = [weight * dist for weight, dist in zip(weights, d2)]
        if math.fsum(probs) <= 0:
            # 所有网格都与既有中心重合：复制权重最大者的中心，空簇事后丢弃
            index = max(range(len(vectors)), key=lambda i: (weights[i], -i))
        else:
            index = _weighted_pick(probs, rng)
        centers.append(list(vectors[index]))
        d2 = [min(dist, squared_distance(vector, centers[-1]))
              for dist, vector in zip(d2, vectors)]
    return centers


def _nearest(vector, centers):
    return min(range(len(centers)),
               key=lambda i: (squared_distance(vector, centers[i]), i))


def _recenter(vectors, labels, weights, centers):
    dimension = len(vectors[0])
    sums = [[0.0] * dimension for _ in centers]
    totals = [0.0] * len(centers)
    for vector, label, weight in zip(vectors, labels, weights):
        totals[label] += weight
        for d in range(dimension):
            sums[label][d] += weight * vector[d]
    return [[sums[c][d] / totals[c] if totals[c] > 0 else centers[c][d]
             for d in range(dimension)] for c in range(len(centers))]


def _weighted_kmeans_single(vectors, weights, k, rng, max_iter):
    centers = _kmeanspp(vectors, weights, k, rng)
    labels = [None] * len(vectors)
    for _ in range(max_iter):
        new_labels = [_nearest(vector, centers) for vector in vectors]
        if new_labels == labels:
            break
        labels = new_labels
        centers = _recenter(vectors, labels, weights, centers)
    labels = [_nearest(vector, centers) for vector in vectors]
    inertia = math.fsum(weight * squared_distance(vector, centers[label])
                        for vector, label, weight in zip(vectors, labels, weights))
    return centers, labels, inertia


def fit_on_cells(cells, k, seed=DEFAULT_SEED, n_init=DEFAULT_N_INIT,
                 max_iter=DEFAULT_MAX_ITER, categorical_weight=DEFAULT_CATEGORICAL_WEIGHT):
    """在给定网格上拟合完整流水线（预处理 + 面积加权 k 均值），k 为请求的簇数。

    空簇被丢弃，保留簇按面积占比降序重编号；全部网格特征相同时收敛为单簇。
    """
    if k < 1:
        raise ValueError('k 必须为正整数')
    if not cells:
        raise ValueError('没有可用网格')
    pre = fit_preprocessor(cells, categorical_weight)
    vectors = [transform_cell(pre, cell)[0] for cell in cells]
    weights = [cell['weight'] for cell in cells]
    rng = random.Random(seed)
    best = None
    for _ in range(max(1, n_init)):
        result = _weighted_kmeans_single(vectors, weights, k, rng, max_iter)
        if best is None or result[2] < best[2] - 1e-9:
            best = result
    centers, _, _ = best
    labels = [_nearest(vector, centers) for vector in vectors]
    cluster_weight = [0.0] * len(centers)
    for label, weight in zip(labels, weights):
        cluster_weight[label] += weight
    order = sorted(range(len(centers)), key=lambda c: (-cluster_weight[c], centers[c]))
    keep = [c for c in order if cluster_weight[c] > 0]
    remap = {old: new for new, old in enumerate(keep)}
    centers = [centers[c] for c in keep]
    labels = [remap[label] for label in labels]
    total = math.fsum(weights)
    shares = [cluster_weight[c] / total for c in keep]
    max_dist = [0.0] * len(centers)
    for vector, label in zip(vectors, labels):
        dist = euclidean(vector, centers[label])
        if dist > max_dist[label]:
            max_dist[label] = dist
    inertia = math.fsum(weight * squared_distance(vector, centers[label])
                        for vector, label, weight in zip(vectors, labels, weights))
    return {'pre': pre, 'centers': centers, 'labels': labels, 'vectors': vectors,
            'weights': weights, 'area_share': shares, 'max_dist': max_dist,
            'inertia': inertia}


def assign_labels(pre, centers, cells):
    """用既有预处理与簇中心给新网格打标签（不做弃权判断）。"""
    return [_nearest(transform_cell(pre, cell)[0], centers) for cell in cells]


def _describe_cluster(cluster, members, area, total, stats, substrate_shares, seamount_shares):
    parts = [f'簇{cluster + 1}：代表面积占比 {area / total:.1%}'
             f'（{len(members)} 个网格，合计 {area:.1f} 平方千米）']
    depth = stats['depth']
    if depth['mean'] is not None:
        parts.append(f"深度均值约 {depth['mean']:.0f} 米"
                     f"（范围 {depth['minimum']:.0f}～{depth['maximum']:.0f} 米）")
    slope = stats['slope']
    if slope['mean'] is not None:
        parts.append(f"坡度均值约 {slope['mean']:.1f} 度")
    roughness = stats['roughness']
    if roughness['mean'] is not None:
        parts.append(f"粗糙度均值约 {roughness['mean']:.2f}")
    top_substrate = sorted(substrate_shares.items(), key=lambda item: (-item[1], item[0]))[:2]
    parts.append('底质以 ' + '、'.join(f'{name}（面积占比 {share:.1%}）'
                                       for name, share in top_substrate) + ' 为主')
    top_seamounts = sorted(seamount_shares.items(), key=lambda item: (-item[1], item[0]))[:3]
    parts.append('主要分布海山：' + '、'.join(f'{name}（{share:.1%}）'
                                             for name, share in top_seamounts))
    return '。'.join(parts) + '。'


def build_profiles(cells, labels):
    """按代表面积汇总每个簇的生境特征，并生成中文说明。"""
    total = math.fsum(cell['weight'] for cell in cells)
    profiles = []
    for cluster in sorted(set(labels)):
        members = [cell for cell, label in zip(cells, labels) if label == cluster]
        area = math.fsum(cell['weight'] for cell in members)
        substrate_area = {}
        seamount_area = {}
        for cell in members:
            substrate_area[cell['substrate']] = substrate_area.get(cell['substrate'], 0.0) + cell['weight']
            seamount_area[cell['seamount']] = seamount_area.get(cell['seamount'], 0.0) + cell['weight']
        substrate_shares = {name: value / area for name, value in sorted(substrate_area.items())}
        seamount_shares = {name: value / area for name, value in sorted(seamount_area.items())}
        stats = {}
        for name in NUMERIC_FEATURES:
            observed = [(cell[name], cell['weight']) for cell in members if cell[name] is not None]
            if observed:
                values = [value for value, _ in observed]
                weights = [weight for _, weight in observed]
                stats[name] = {'mean': weighted_mean(values, weights),
                               'minimum': min(values), 'maximum': max(values),
                               'observed': len(values)}
            else:
                stats[name] = {'mean': None, 'minimum': None, 'maximum': None, 'observed': 0}
        profiles.append({'cluster': cluster, 'label': f'簇{cluster + 1}',
                         'cells': len(members), 'area': area, 'area_share': area / total,
                         'features': stats, 'substrate_shares': substrate_shares,
                         'seamount_shares': seamount_shares,
                         'description': _describe_cluster(cluster, members, area, total,
                                                          stats, substrate_shares,
                                                          seamount_shares)})
    return profiles


def artifact_fingerprint(artifact):
    payload = {key: value for key, value in artifact.items() if key != 'fingerprint'}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class HabitatModel:
    """可保存、可重新载入的生境分群模型；同一模型再次载入后归属一致。"""

    def __init__(self, artifact):
        self.artifact = artifact

    @classmethod
    def train(cls, records, k, selection=None, seed=DEFAULT_SEED, n_init=DEFAULT_N_INIT,
              max_iter=DEFAULT_MAX_ITER, categorical_weight=DEFAULT_CATEGORICAL_WEIGHT,
              min_margin=DEFAULT_MIN_MARGIN):
        cells = aggregate_cells(records)
        fit = fit_on_cells(cells, k, seed=seed, n_init=n_init, max_iter=max_iter,
                           categorical_weight=categorical_weight)
        artifact = {
            'product': PRODUCT,
            'module': 'habitat',
            'schema_version': 1,
            'weight_semantics': WEIGHT_SEMANTICS,
            'features': {'numeric': list(NUMERIC_FEATURES),
                         'categorical': [CATEGORICAL_FEATURE],
                         'categorical_weight': categorical_weight},
            'preprocessing': fit['pre'],
            'training': {'seamounts': sorted({cell['seamount'] for cell in cells}),
                         'cells': len(cells), 'observations': len(records),
                         'input_fingerprint': fingerprint(records),
                         'seed': seed, 'n_init': n_init, 'max_iter': max_iter},
            'clusters': {'k': len(fit['centers']), 'k_requested': k,
                         'centers': fit['centers'], 'area_share': fit['area_share'],
                         'max_train_distance': fit['max_dist'], 'inertia': fit['inertia']},
            'diagnostics': {'min_margin': min_margin, 'rule': ABSTAIN_RULE},
            'selection': selection,
            'profiles': build_profiles(cells, fit['labels']),
        }
        artifact['fingerprint'] = artifact_fingerprint(artifact)
        return cls(artifact)

    @classmethod
    def load(cls, path):
        try:
            artifact = json.loads(Path(path).read_text(encoding='utf-8'))
        except json.JSONDecodeError as error:
            raise ValueError('模型文件不是有效 JSON：' + str(error)) from None
        if not isinstance(artifact, dict) or artifact.get('module') != 'habitat' \
                or artifact.get('schema_version') != 1:
            raise ValueError('模型文件不是有效的 habitat 产物')
        return cls(artifact)

    def save(self, path):
        save_json(path, self.artifact)

    def predict(self, records):
        """逐网格给出归属与置信诊断；无可靠归属的网格 cluster 为 null。"""
        cells = aggregate_cells(records)
        pre = self.artifact['preprocessing']
        centers = self.artifact['clusters']['centers']
        max_dist = self.artifact['clusters']['max_train_distance']
        min_margin = self.artifact['diagnostics']['min_margin']
        assignments = []
        for cell in cells:
            vector, notes = transform_cell(pre, cell)
            distances = [euclidean(vector, center) for center in centers]
            nearest = min(range(len(centers)), key=lambda i: (distances[i], i))
            own = distances[nearest]
            others = [dist for i, dist in enumerate(distances) if i != nearest]
            second = min(others) if others else None
            margin = 1.0 if not second or second <= 0 else (second - own) / second
            flags = [note for note in notes if note.startswith('缺值已填补')]
            reasons = [note for note in notes if note.startswith('未见底质类别')]
            if own > max_dist[nearest] + 1e-9:
                reasons.append('超出训练距离范围')
            if margin < min_margin:
                reasons.append('置信边际不足')
            assignments.append({
                'cell': cell['cell'], 'seamount': cell['seamount'],
                'observations': cell['observations'], 'weight': cell['weight'],
                'cluster': None if reasons else nearest, 'nearest_cluster': nearest,
                'confidence': margin, 'distance': own, 'margin': margin,
                'flags': flags, 'abstain_reasons': reasons})
        return {'product': PRODUCT, 'module': 'habitat', 'schema_version': 1,
                'model_fingerprint': self.artifact['fingerprint'],
                'input_fingerprint': fingerprint(records),
                'weight_semantics': WEIGHT_SEMANTICS,
                'cells': len(assignments),
                'unassigned': sum(1 for item in assignments if item['cluster'] is None),
                'assignments': assignments}
