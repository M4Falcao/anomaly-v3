import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import common as cm  # noqa: E402
import evaluate as ev  # noqa: E402
import masking as mk  # noqa: E402


def blob(h=448, w=448):
    m = np.zeros((h, w), bool)
    cv2.circle(m.view(np.uint8), (150, 200), 40, 1, -1)
    m[100:120, 300:340] = True
    return m


class TestStrategies(unittest.TestCase):
    def test_parse_and_names(self):
        self.assertEqual(mk.parse_variant('dil32__black'), dict(strategy='dil32', fill='black', feather=0))
        self.assertEqual(mk.parse_variant('hull+dil16__imagenet_mean__f8')['feather'], 8)
        self.assertEqual(mk.variant_name('bbox10', 'blur', 4), 'bbox10__blur__f4')
        self.assertIsNone(mk.parse_variant('none')['strategy'])
        for bad in ('dil__black', 'dil32__pink', 'dil32', 'foo__black'):
            with self.assertRaises(ValueError):
                mk.parse_variant(bad)

    def test_monotonic_and_containment(self):
        base = blob()
        prev = base
        for s in ('dil8', 'dil16', 'dil32', 'dil64'):
            m = mk.strategy_mask(base, s)
            self.assertTrue((m >= prev).all(), s)
            prev = m
        hull = mk.strategy_mask(base, 'hull')
        self.assertTrue((hull >= base).all())
        self.assertTrue((mk.strategy_mask(base, 'hull+dil16') >= hull).all())
        bbox = mk.strategy_mask(base, 'bbox10')
        self.assertTrue((bbox >= hull).all())

    def test_dilation_radius(self):
        base = np.zeros((448, 448), bool)
        base[224, 224] = True
        m = mk.strategy_mask(base, 'dil16')
        self.assertTrue(m[224, 224 + 16])
        self.assertFalse(m[224, 224 + 18])

    def test_empty_base_keeps_everything(self):
        self.assertTrue(mk.strategy_mask(np.zeros((448, 448), bool), 'dil32').all())

    def test_fill_and_feather(self):
        img = np.full((300, 600, 3), 200, np.uint8)
        grid = np.zeros((448, 448), bool)
        grid[:, :224] = True
        out = mk.apply_fill(img, grid, 'black')
        self.assertTrue((out[:, :290] == 200).all() and (out[:, 310:] == 0).all())
        mean = mk.apply_fill(img, grid, 'imagenet_mean')
        self.assertEqual(tuple(mean[0, -1]), mk.IMAGENET_MEAN_RGB[::-1])
        feath = mk.apply_fill(img, grid, 'black', feather=8)
        self.assertTrue((feath[:, :299] == 200).all())
        self.assertTrue(0 < feath[0, 305, 0] < 200)

    def test_contour_band(self):
        band = mk.contour_band(blob(), 8)
        self.assertTrue(band.any())
        self.assertFalse(band[200, 150])
        self.assertFalse(mk.contour_band(np.ones((448, 448), bool)).any())


class TestHistogramMetrics(unittest.TestCase):
    def test_matches_sklearn(self):
        from sklearn.metrics import average_precision_score, roc_auc_score
        rng = np.random.default_rng(1)
        neg = rng.normal(0, 1, 200000)
        pos = rng.normal(1.5, 1, 3000)
        scores = np.concatenate([neg, pos])
        labels = np.concatenate([np.zeros_like(neg), np.ones_like(pos)])
        edges = ev.quantile_edges(scores[rng.integers(0, len(scores), 50000)])
        hp, hn = ev.bincount(pos, edges), ev.bincount(neg, edges)
        self.assertAlmostEqual(ev.auroc_from_hist(hp, hn), roc_auc_score(labels, scores), places=3)
        self.assertAlmostEqual(ev.ap_from_hist(hp, hn), average_precision_score(labels, scores), places=2)
        thr = ev.threshold_at_fpr(hn, edges, 0.01)
        self.assertAlmostEqual((neg >= thr).mean(), 0.01, delta=0.002)
        self.assertAlmostEqual(ev.fpr_at(hn, edges, thr), (neg >= thr).mean(), delta=0.002)

    def test_paired_bootstrap_sign(self):
        rng = np.random.default_rng(2)
        n, nb = 60, 32
        labels = np.array([0] * 40 + [1] * 20)
        pos_good = np.zeros((n, nb), int)
        neg = rng.integers(0, 50, (n, nb))
        neg[:, nb // 2:] = 0
        pos_good[labels == 1, -4:] = 10
        pos_bad = np.zeros((n, nb), int)
        pos_bad[labels == 1, :4] = 10
        out = ev.paired_bootstrap(pos_bad, neg, pos_good, neg, labels, n_boot=200)
        self.assertGreater(out['pixel']['ci_low'], 0)

    def test_image_auroc_bootstrap(self):
        rng = np.random.default_rng(3)
        labels = np.array([0] * 80 + [1] * 30)
        good = labels * 2.0 + rng.normal(0, 1, len(labels))
        bad = rng.normal(0, 1, len(labels))
        out = ev.image_auroc_bootstrap(labels, bad, good, n_boot=200)
        self.assertGreater(out['ci_low'], 0)
        same = ev.image_auroc_bootstrap(labels, good, good, n_boot=50)
        self.assertEqual((same['ci_low'], same['ci_high']), (0.0, 0.0))


class TestSelectionRule(unittest.TestCase):
    def test_rule(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('sel', os.path.join(ROOT, '05_select_strategies.py'))
        sel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sel)
        rows = []
        for s, area, lost in (('raw', .2, .10), ('dil16', .3, .04), ('dil32', .4, .02), ('dil64', .6, .01), ('bbox10', .7, .0)):
            rows.append(dict(strategy=s, partition='train_good', area_kept=area, gt_lost_frac='', n_lost50=0, n_lost95=0))
            rows.append(dict(strategy=s, partition='val', area_kept=area, gt_lost_frac=lost, n_lost50=0, n_lost95=0))
        finalists, reliable, _ = sel.select(rows, 0.05, 2, ['dil64', 'bbox20'])
        self.assertEqual((finalists, reliable), (['dil16', 'dil32'], True))
        for r in rows:
            if r['partition'] == 'val':
                r['gt_lost_frac'] = 0.5
        self.assertEqual(sel.select(rows, 0.05, 2, ['dil64', 'bbox20'])[:2], (['dil64', 'bbox20'], False))


class TestMaterialize(unittest.TestCase):
    def test_same_tree_and_split(self):
        tmp = tempfile.mkdtemp()
        try:
            orig, masks, data = (os.path.join(tmp, d) for d in ('orig', 'masks', 'data'))
            cls = 'vari-grip'
            rng = np.random.default_rng(0)
            for split, defect, n in (('train', 'good', 3), ('test', 'good', 4), ('test', 'rust', 3)):
                for i in range(n):
                    img = rng.integers(60, 255, (120, 160, 3), dtype=np.uint8)
                    os.makedirs(os.path.join(orig, cls, split, defect), exist_ok=True)
                    cv2.imwrite(os.path.join(orig, cls, split, defect, f'{defect}{i}.jpg'), img)
                    m = np.zeros((120, 160), np.uint8)
                    m[30:90, 40:120] = 255
                    cm.write_png(os.path.join(masks, split, defect, f'{defect}{i}.png'), m)
                    if defect == 'rust':
                        g = np.zeros((120, 160), np.uint8)
                        g[5:15, 5:15] = 255
                        cm.write_png(os.path.join(orig, cls, 'ground_truth', defect, f'{defect}{i}.png'), g)
            cm.write_json(os.path.join(masks, 'manifest.json'), dict(class_name=cls))
            root = mk.materialize(cls, 'dil16__black', masks, orig_root=orig, data_root=data)
            for split in ('train', 'test'):
                for defect in os.listdir(os.path.join(orig, cls, split)):
                    self.assertEqual(os.listdir(os.path.join(orig, cls, split, defect)),
                                     os.listdir(os.path.join(root, cls, split, defect)))
            self.assertEqual(sorted(os.listdir(os.path.join(root, cls, 'ground_truth', 'rust'))),
                             ['rust0.png', 'rust1.png', 'rust2.png'])
            out = cv2.imread(os.path.join(root, cls, 'test', 'good', 'good0.jpg'))
            self.assertLess(out[:5, :5].max(), 12)
            self.assertGreater(out[60, 80].mean(), 40)
            self.assertEqual(mk.materialize(cls, 'dil16__black', masks, orig_root=orig, data_root=data), root)
            self.assertEqual(mk.materialize(cls, 'none', masks, orig_root=orig, data_root=data), orig)

            import protocol as pt
            inv_o = pt.test_inventory(orig, cls)
            inv_m = pt.test_inventory(root, cls)
            self.assertEqual([r for r, _, _ in inv_o], [r for r, _, _ in inv_m])
            labels = [lab for _, lab, _ in inv_o]
            self.assertEqual(sum(labels), 3)
            self.assertTrue(all(gt for _, lab, gt in inv_m if lab))
            val, hold = pt.split_indices(labels, 0.4, 42)
            self.assertEqual(sorted(val + hold), list(range(len(labels))))
        finally:
            shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
