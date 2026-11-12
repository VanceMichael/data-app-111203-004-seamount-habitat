import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analysis_app.habitat import (HabitatModel, WEIGHT_SEMANTICS, aggregate_cells)
from analysis_app.io import load_records
from analysis_app.metrics import (adjusted_rand, align_labels, aligned_agreement)
from analysis_app.selection import choose_k, loso_folds, select_k

REGIMES = ((200.0, 4.0, 0.2, 'sand'), (500.0, 10.0, 0.6, 'mixed'), (800.0, 16.0, 1.0, 'rock'))


def make_record(obs, cell, seamount, depth, slope=5.0, roughness=0.1,
                substrate='sand', weight=1.0):
    return {'observation_id': obs, 'cell': cell, 'seamount': seamount, 'depth': depth,
            'slope': slope, 'roughness': roughness, 'substrate': substrate,
            'survey': 'V0', 'weight': weight}


def synthetic_records(seamounts=('A', 'B', 'C', 'D'), cells_per=6):
    """三座分离良好的深度 regime 轮换出现，扰动确定，不依赖随机源。"""
    records = []
    for si, seamount in enumerate(seamounts):
        for i in range(cells_per):
            depth, slope, roughness, substrate = REGIMES[i % len(REGIMES)]
            jitter = ((si * 7 + i * 3) % 5) * 0.4
            records.append(make_record(f'{seamount}-{i}', f'{seamount}-G{i}', seamount,
                                       depth + jitter, slope=slope + jitter * 0.1,
                                       roughness=roughness, substrate=substrate,
                                       weight=1.0 + si))
    return records


class HabitatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = synthetic_records()
        cls.model = HabitatModel.train(cls.records, k=3)

    def test_unseen_substrate_abstains(self):
        new = [make_record('n1', 'X1', 'E', 210.0, slope=4.2, roughness=0.2,
                           substrate='gravel')]
        result = self.model.predict(new)
        entry = result['assignments'][0]
        self.assertIsNone(entry['cluster'])
        self.assertTrue(any('未见底质类别' in reason
                            for reason in entry['abstain_reasons']))
        self.assertEqual(result['unassigned'], 1)
        # 类别字典不被新数据扩展
        self.assertEqual(self.model.artifact['preprocessing']['dictionary'],
                         ['mixed', 'rock', 'sand'])

    def test_far_extrapolation_abstains(self):
        new = [make_record('n1', 'X1', 'E', 5000.0, slope=40.0, roughness=9.0,
                           substrate='sand')]
        entry = self.model.predict(new)['assignments'][0]
        self.assertIsNone(entry['cluster'])
        self.assertIn('超出训练距离范围', entry['abstain_reasons'])

    def test_duplicate_cells_aggregated(self):
        records = [make_record('o1', 'G1', 'A', 100.0),
                   make_record('o2', 'G1', 'A', 200.0),
                   make_record('o3', 'G2', 'A', 300.0)]
        cells = aggregate_cells(records)
        self.assertEqual(len(cells), 2)
        merged = cells[0]
        self.assertEqual(merged['cell'], 'G1')
        self.assertEqual(merged['observations'], 2)
        self.assertEqual(merged['depth'], 150.0)
        conflict = records + [make_record('o4', 'G1', 'B', 100.0)]
        with self.assertRaisesRegex(ValueError, '海山不一致'):
            aggregate_cells(conflict)

    def test_area_weight_dominates(self):
        # 10 个轻网格（各 1 平方千米）对 1 个重网格（1000 平方千米）：
        # 网格计数上重网格只占 1/11，面积占比应接近 1，聚类中心由面积决定
        records = [make_record(f'l{i}', f'L{i}', 'A', 100.0 + i, weight=1.0)
                   for i in range(10)]
        records.append(make_record('h0', 'H0', 'A', 1000.0, slope=20.0,
                                   roughness=2.0, substrate='rock', weight=1000.0))
        model = HabitatModel.train(records, k=2)
        heavy = [p for p in model.artifact['profiles'] if p['cells'] == 1][0]
        self.assertGreater(heavy['area_share'], 0.9)
        self.assertAlmostEqual(heavy['features']['depth']['mean'], 1000.0)

    def test_label_permutation_aligned(self):
        reference = [0, 0, 1, 1, 2, 2]
        permuted = [2, 2, 0, 0, 1, 1]
        mapping = align_labels(reference, permuted)
        self.assertEqual(mapping, {2: 0, 0: 1, 1: 2})
        aligned = [mapping[label] for label in permuted]
        self.assertEqual(adjusted_rand(reference, aligned), 1.0)
        self.assertEqual(aligned_agreement(reference, permuted), 1.0)

    def test_reseeded_models_align(self):
        other = HabitatModel.train(self.records, k=3, seed=99)
        cells = aggregate_cells(self.records)
        base = self.model.predict(self.records)['assignments']
        theirs = other.predict(self.records)['assignments']
        reference = [entry['nearest_cluster'] for entry in base]
        target = [entry['nearest_cluster'] for entry in theirs]
        self.assertGreaterEqual(aligned_agreement(reference, target), 0.95)
        self.assertEqual(len(cells), len(reference))

    def test_identical_features_collapse(self):
        records = [make_record(f'o{i}', f'G{i}', 'A', 300.0, slope=5.0,
                               roughness=0.5, substrate='sand') for i in range(6)]
        model = HabitatModel.train(records, k=3)
        self.assertEqual(model.artifact['clusters']['k'], 1)
        result = model.predict(records)
        self.assertEqual([entry['cluster'] for entry in result['assignments']], [0] * 6)
        json.dumps(model.artifact, ensure_ascii=False, allow_nan=False)

    def test_reload_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            self.model.save(path)
            loaded = HabitatModel.load(path)
            again = HabitatModel.load(path)
        self.assertEqual(loaded.artifact, self.model.artifact)
        expected = self.model.predict(self.records)
        self.assertEqual(loaded.predict(self.records), expected)
        self.assertEqual(again.predict(self.records), expected)

    def test_deterministic_training(self):
        retrained = HabitatModel.train(self.records, k=3)
        self.assertEqual(retrained.artifact, self.model.artifact)

    def test_holdout_folds_keep_seamounts_intact(self):
        cells = aggregate_cells(self.records)
        folds = list(loso_folds(cells))
        self.assertEqual([name for name, _, _ in folds], ['A', 'B', 'C', 'D'])
        for name, train, test in folds:
            self.assertTrue(all(cell['seamount'] != name for cell in train))
            self.assertTrue(all(cell['seamount'] == name for cell in test))
            self.assertEqual(len(train) + len(test), len(cells))

    def test_choose_k_balances_components(self):
        candidates = [{'k': 2, 'score': 0.7}, {'k': 3, 'score': 0.9},
                      {'k': 4, 'score': 0.9}, {'k': 5, 'score': 0.6}]
        self.assertEqual(choose_k(candidates), 3)  # 并列取较小 k
        # 轮廓系数最高但综合得分落败时不被选
        candidates = [{'k': 2, 'score': 0.8}, {'k': 4, 'score': 0.75}]
        self.assertEqual(choose_k(candidates), 2)

    def test_select_k_reports_all_components(self):
        k, report = select_k(aggregate_cells(self.records), k_max=4)
        self.assertEqual(k, 3)
        self.assertEqual(report['chosen_k'], choose_k(report['candidates']))
        for candidate in report['candidates']:
            self.assertIsNotNone(candidate['silhouette'])
            self.assertIsNotNone(candidate['stability'])
            self.assertIsNotNone(candidate['size_balance'])
            self.assertEqual(len(candidate['folds']), 4)
            for fold in candidate['folds']:
                self.assertIn('adjusted_rand', fold)
                self.assertIn('aligned_agreement', fold)

    def test_weight_semantics_recorded(self):
        artifact = self.model.artifact
        self.assertEqual(artifact['weight_semantics'], WEIGHT_SEMANTICS)
        self.assertIn('代表面积', artifact['weight_semantics'])
        self.assertIn('网格数量', artifact['weight_semantics'])
        result = self.model.predict(self.records)
        self.assertEqual(result['weight_semantics'], WEIGHT_SEMANTICS)

    def test_profiles_are_chinese_and_complete(self):
        profiles = self.model.artifact['profiles']
        self.assertEqual(len(profiles), 3)
        total = sum(profile['area_share'] for profile in profiles)
        self.assertAlmostEqual(total, 1.0)
        for profile in profiles:
            text = profile['description']
            self.assertIn('簇', text)
            self.assertIn('底质', text)
            self.assertIn('平方千米', text)
            self.assertIn('海山', text)

    def test_missing_numeric_imputed(self):
        records = synthetic_records(seamounts=('A', 'B'), cells_per=4)
        records[0]['depth'] = None
        model = HabitatModel.train(records, k=2)
        self.assertIsNotNone(model.artifact['preprocessing']['imputation']['depth'])
        result = model.predict(records)
        flagged = [entry for entry in result['assignments'] if entry['flags']]
        self.assertTrue(any('缺值已填补' in flag
                            for entry in flagged for flag in entry['flags']))

    def test_cli_train_then_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / 'model.json'
            assign_path = Path(directory) / 'assign.json'
            assign2_path = Path(directory) / 'assign2.json'
            train = subprocess.run(
                [sys.executable, '-m', 'analysis_app', 'habitat', '--k', '3',
                 '--model', str(model_path), '--assignments', str(assign_path)],
                text=True, capture_output=True)
            self.assertEqual(train.returncode, 0, train.stderr)
            artifact = json.loads(train.stdout)
            self.assertEqual(artifact['clusters']['k'], 3)
            self.assertEqual(json.loads(model_path.read_text()), artifact)
            predict = subprocess.run(
                [sys.executable, '-m', 'analysis_app', 'habitat', '--predict',
                 '--model', str(model_path), '--assignments', str(assign2_path)],
                text=True, capture_output=True)
            self.assertEqual(predict.returncode, 0, predict.stderr)
            self.assertEqual(json.loads(assign_path.read_text()),
                             json.loads(assign2_path.read_text()))
            self.assertEqual(json.loads(predict.stdout),
                             json.loads(assign_path.read_text()))

    def test_cli_predict_missing_model_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run(
                [sys.executable, '-m', 'analysis_app', 'habitat', '--predict',
                 '--model', str(Path(directory) / 'missing.json')],
                capture_output=True)
            self.assertEqual(run.returncode, 2)

    def test_fixture_selection_uses_group_holdout(self):
        records = load_records('data/observations.csv')
        k, report = select_k(aggregate_cells(records), k_max=3)
        self.assertIn(k, (2, 3))
        self.assertEqual(len(report['candidates']), 2)
        for candidate in report['candidates']:
            self.assertEqual(len(candidate['folds']), 4)
            self.assertIsNotNone(candidate['stability'])


if __name__ == '__main__':
    unittest.main()
