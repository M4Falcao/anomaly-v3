import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import common as cm  # noqa: E402


def synthetic_pair(h=120, w=160, seed=0):
    rng = np.random.default_rng(seed)
    orig = rng.integers(40, 255, size=(h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (40, 60), 25, 1, -1)
    cv2.rectangle(mask, (95, 20), (140, 100), 1, -1)
    nobg = orig * mask[..., None]
    return orig, nobg, mask.astype(bool)


class TestMaskOps(unittest.TestCase):
    def test_mask_from_nobg_recovers_objects(self):
        _, nobg, mask = synthetic_pair()
        self.assertGreater(cm.iou(cm.mask_from_nobg(nobg), mask), 0.98)

    def test_jpeg_noise_and_dark_pixels(self):
        _, nobg, mask = synthetic_pair()
        noisy = nobg.copy()
        noisy[~mask] = 6
        noisy[60, 40] = 0
        self.assertGreater(cm.iou(cm.mask_from_nobg(noisy), mask), 0.98)

    def test_small_components_removed(self):
        _, nobg, mask = synthetic_pair()
        nobg[5, 5] = 200
        self.assertFalse(cm.mask_from_nobg(nobg)[5, 5])

    def test_polygon_roundtrip_one_instance_per_component(self):
        _, _, mask = synthetic_pair()
        polys = cm.mask_to_polygons(mask)
        self.assertEqual(len(polys), 2)
        for p in polys:
            self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertGreater(cm.iou(cm.rasterize_polygons(polys, mask.shape), mask), 0.95)
        lines = cm.polygons_to_yolo_lines(polys)
        self.assertTrue(all(l.startswith('0 ') and len(l.split()) % 2 == 1 for l in lines))

    def test_union_instances(self):
        a = np.zeros((2, 10, 10), bool)
        a[0, :5] = True
        a[1, 5:, 5:] = True
        u = cm.union_instance_masks(a, (10, 10))
        self.assertEqual(u.sum(), 50 + 25)
        self.assertEqual(cm.union_instance_masks([], (4, 4)).sum(), 0)
        with self.assertRaises(ValueError):
            cm.union_instance_masks(a, (8, 8))


class TestBuildDataset(unittest.TestCase):
    def test_conversion_end_to_end(self):
        tmp = tempfile.mkdtemp()
        try:
            orig_root, nobg_root, out_root = (os.path.join(tmp, d) for d in ('orig', 'nobg', 'out'))
            cls = 'vari-grip'
            for i in range(6):
                orig, nobg, _ = synthetic_pair(seed=i)
                for root, img in ((orig_root, orig), (nobg_root, nobg)):
                    d = os.path.join(root, cls, 'train', 'good')
                    os.makedirs(d, exist_ok=True)
                    cv2.imwrite(os.path.join(d, f'img{i}.png'), img)
            orig, _, _ = synthetic_pair(seed=99)
            cv2.imwrite(os.path.join(orig_root, cls, 'train', 'good', 'only_orig.png'), orig)
            cmd = [sys.executable, os.path.join(ROOT, '01_build_yolo_dataset.py'), '--classes', cls,
                   '--orig_root', orig_root, '--nobg_root', nobg_root, '--out_root', out_root, '--val_frac', '0.34']
            subprocess.run(cmd, check=True, capture_output=True)
            base = os.path.join(out_root, cls)
            n_train = len(os.listdir(os.path.join(base, 'labels', 'train')))
            n_val = len(os.listdir(os.path.join(base, 'labels', 'val')))
            self.assertEqual((n_train, n_val), (4, 2))
            self.assertTrue(os.path.isfile(os.path.join(base, 'data.yaml')))
            with open(os.path.join(base, 'conversion_report.csv'), encoding='utf-8') as f:
                report = f.read()
            self.assertIn('missing_in_nobg', report)
            img = cv2.imread(os.path.join(base, 'images', 'train', os.listdir(os.path.join(base, 'images', 'train'))[0]))
            # A imagem copiada deve ser a original (com fundo), não a sem fundo.
            self.assertLess((img.max(axis=2) <= 10).mean(), 0.01)
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(cmd, check=True, capture_output=True)
        finally:
            shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
