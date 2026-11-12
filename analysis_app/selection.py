"""按海山整组留出的外推验证与簇数选择。

不随机拆分网格：每一折留出一整座海山，折内重新拟合预处理与聚类，
重采样簇先对齐到基准簇，再计算稳定度。
"""
import math
import statistics

from .habitat import (DEFAULT_CATEGORICAL_WEIGHT, DEFAULT_MAX_ITER,
                      DEFAULT_N_INIT, DEFAULT_SEED, assign_labels, fit_on_cells)
from .metrics import (adjusted_rand, align_labels, aligned_agreement,
                      silhouette_samples, weighted_mean)

SELECTION_RULE = (
    '候选 k 的得分 = 归一化训练轮廓系数、留出稳定度与小簇规模三者的算术平均：'
    '留出稳定度为按 seamount 整组留出、重采样簇与基准簇对齐后的调整后兰德指数均值；'
    '小簇规模为最小簇面积占比×k（封顶 1）。取最高分，并列取较小 k。'
    '只按海山整组留出，不随机拆分网格。'
)

SELECTION_NOTE = '留出稳定度按 seamount 整组留出计算；只有一座海山时退化为轮廓系数与小簇规模'


def loso_folds(cells):
    """留一海山交叉验证：每一折的测试集是一整座海山的全部网格。"""
    seamounts = sorted({cell['seamount'] for cell in cells})
    for seamount in seamounts:
        train = [cell for cell in cells if cell['seamount'] != seamount]
        test = [cell for cell in cells if cell['seamount'] == seamount]
        yield seamount, train, test


def choose_k(candidates):
    """取得分最高的候选；并列取较小 k。"""
    best = None
    for candidate in candidates:
        if best is None or candidate['score'] > best['score'] + 1e-12:
            best = candidate
    if best is None:
        raise ValueError('没有可用的候选簇数')
    return best['k']


def evaluate_k(cells, k, seed=DEFAULT_SEED, n_init=DEFAULT_N_INIT,
               max_iter=DEFAULT_MAX_ITER, categorical_weight=DEFAULT_CATEGORICAL_WEIGHT):
    """评估单个候选簇数：训练轮廓系数、整组留出稳定度与小簇规模。"""
    full = fit_on_cells(cells, k, seed=seed, n_init=n_init, max_iter=max_iter,
                        categorical_weight=categorical_weight)
    samples = silhouette_samples(full['vectors'], full['labels'], full['weights'])
    silhouette = weighted_mean(samples, full['weights'])
    size_balance = min(1.0, min(full['area_share']) * k)
    seamounts = sorted({cell['seamount'] for cell in cells})
    folds = []
    if len(seamounts) >= 2:
        for seamount, train_cells, test_cells in loso_folds(cells):
            fold = fit_on_cells(train_cells, k, seed=seed, n_init=n_init,
                                max_iter=max_iter, categorical_weight=categorical_weight)
            reference = assign_labels(full['pre'], full['centers'], test_cells)
            resampled = assign_labels(fold['pre'], fold['centers'], test_cells)
            mapping = align_labels(reference, resampled)
            aligned = [mapping.get(label) for label in resampled]
            folds.append({'seamount': seamount,
                          'adjusted_rand': adjusted_rand(reference, aligned),
                          'aligned_agreement': aligned_agreement(reference, resampled)})
        stability = statistics.fmean(fold['adjusted_rand'] for fold in folds)
    else:
        stability = None
    components = [(silhouette + 1) / 2, size_balance]
    if stability is not None:
        components.append(max(0.0, stability))
    return {'k': k, 'silhouette': silhouette, 'stability': stability,
            'size_balance': size_balance, 'score': statistics.fmean(components),
            'folds': folds}


def select_k(cells, k_max=6, seed=DEFAULT_SEED, n_init=DEFAULT_N_INIT,
             max_iter=DEFAULT_MAX_ITER, categorical_weight=DEFAULT_CATEGORICAL_WEIGHT):
    """在候选簇数中选择兼顾留出稳定性与小簇规模的 k，返回 (k, 选簇报告)。"""
    upper = min(k_max, len(cells) - 1)
    if upper < 2:
        raise ValueError('网格数量不足，无法自动选择簇数')
    candidates = [evaluate_k(cells, k, seed=seed, n_init=n_init, max_iter=max_iter,
                             categorical_weight=categorical_weight)
                  for k in range(2, upper + 1)]
    chosen = choose_k(candidates)
    report = {'rule': SELECTION_RULE, 'note': SELECTION_NOTE,
              'candidates': candidates, 'chosen_k': chosen}
    return chosen, report
