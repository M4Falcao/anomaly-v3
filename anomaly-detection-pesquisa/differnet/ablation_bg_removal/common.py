"""Utilidades compartilhadas da ablação de remoção de fundo (D-19).

Converte imagens sem fundo (pixels de fundo pretos) em máscaras/polígonos YOLO-seg
e consolida máscaras preditas. Funções puras, testadas em ``tests/test_common.py``.
"""
import hashlib
import json
import os
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIFFERNET_ROOT = os.path.dirname(HERE)
DATASET_BASE = r'C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset'


def _env(name, default):
    """Caminhos sobrescrevíveis por variável de ambiente ``ABR_<NOME>`` (usado por ``run_all.py``)."""
    return os.environ.get(f'ABR_{name}') or default


ORIG_ROOT = _env('ORIG_ROOT', os.path.join(DATASET_BASE, 'insplad-seg', 'insplad-seg'))
NOBG_ROOT = _env('NOBG_ROOT', os.path.join(DATASET_BASE, 'insplad-seg-no-back-treino'))
DATA_DIR = _env('DATA_DIR', os.path.join(HERE, 'data'))
RUNS_DIR = _env('RUNS_DIR', os.path.join(HERE, 'runs'))
MASKS_DIR = _env('MASKS_DIR', os.path.join(HERE, 'masks'))
ABLATION_ROOT = _env('ABLATION_ROOT', os.path.join(DIFFERNET_ROOT, 'resultado_analise_final', 'ablacao_mascara_objeto'))
REFERENCE_HEAD_RUN = _env('REFERENCE_HEAD_RUN', os.path.join(DIFFERNET_ROOT, 'resultado_analise_final', 'treino_nf_heads',
                                                              'run_20261025_181523'))
BG_ENCODER_DIR = _env('BG_ENCODER_DIR', os.path.join(DIFFERNET_ROOT, 'final_models', 'SEDiffernet'))
NOBG_ENCODER_DIR = _env('NOBG_ENCODER_DIR', os.path.join(DIFFERNET_ROOT, 'final_models', 'SEDiffernet_noBg'))
PYTHON = _env('PYTHON', r'C:\Users\teo-s\.conda\envs\anomaly_attentideffernet\python.exe')


def find_class_encoder(root, class_name):
    """Exatamente um ``.pt`` da classe em ``root`` (pasta da classe ou nome do arquivo)."""
    import glob
    cands = sorted(glob.glob(os.path.join(root, class_name, '*.pt')))
    if not cands:
        cands = sorted(p for p in glob.glob(os.path.join(root, '**', '*.pt'), recursive=True)
                       if class_name in os.path.basename(p) or class_name in os.path.basename(os.path.dirname(p)))
    if len(cands) != 1:
        raise FileNotFoundError(f'Esperava 1 encoder para {class_name} em {root}; achei {len(cands)}: {cands[:4]}')
    return cands[0]

ALL_CLASSES = [
    'glass-insulator',
    'lightning-rod-suspension',
    'polymer-insulator-upper-shackle',
    'vari-grip',
    'yoke-suspension',
]
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def parse_classes(value):
    if value in (None, '', 'all'):
        return list(ALL_CLASSES)
    classes = [c.strip() for c in value.split(',') if c.strip()]
    unknown = [c for c in classes if c not in ALL_CLASSES]
    if unknown:
        raise ValueError(f'Classes desconhecidas: {unknown}. Válidas: {ALL_CLASSES}')
    return classes


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def read_image(path):
    # cv2.imread falha com caminhos não-ASCII no Windows; imdecode não.
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'Não foi possível ler {path}')
    return img


def write_png(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode('.png', array)
    if not ok:
        raise IOError(f'Falha ao codificar {path}')
    buf.tofile(path)


def mask_from_nobg(img_bgr, black_thr=10, close_px=5, min_component_frac=0.001):
    """Máscara binária do objeto a partir de uma imagem com fundo preto.

    ``black_thr`` tolera artefatos JPEG no fundo; o fechamento reconecta pixels escuros
    do objeto; componentes menores que ``min_component_frac`` da imagem são ruído.
    """
    obj = (img_bgr.max(axis=2) > black_thr).astype(np.uint8)
    if close_px > 1:
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(obj, connectivity=8)
    min_area = min_component_frac * obj.size
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def mask_to_polygons(mask, epsilon_px=1.0, min_points=3):
    """Contornos externos -> lista de polígonos normalizados [(x, y), ...] em [0, 1].

    YOLO-seg aceita um polígono por instância e sem furos; cada componente vira uma instância.
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for cnt in contours:
        if epsilon_px > 0:
            cnt = cv2.approxPolyDP(cnt, epsilon_px, True)
        pts = cnt.reshape(-1, 2).astype(np.float64)
        if len(pts) < min_points:
            continue
        pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
        polygons.append(pts)
    return polygons


def polygons_to_yolo_lines(polygons, class_id=0):
    return [
        f'{class_id} ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in poly)
        for poly in polygons
    ]


def rasterize_polygons(polygons, shape):
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    pts = [np.round(p * np.array([w, h])).astype(np.int32) for p in polygons]
    if pts:
        cv2.fillPoly(out, pts, 1)
    return out.astype(bool)


def union_instance_masks(instance_masks, shape):
    """União de máscaras de instância (N, H, W) -> (H, W) bool."""
    if instance_masks is None or len(instance_masks) == 0:
        return np.zeros(shape, dtype=bool)
    union = np.asarray(instance_masks).astype(bool).any(axis=0)
    if union.shape != tuple(shape):
        raise ValueError(f'Máscara {union.shape} difere da imagem {shape}')
    return union


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni else 1.0


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def new_ablation_run(root=ABLATION_ROOT):
    path = os.path.join(root, f'run_{timestamp()}')
    os.makedirs(path, exist_ok=False)
    return path


def latest_ablation_run(root=ABLATION_ROOT):
    cands = [os.path.join(root, d) for d in os.listdir(root)
             if d.startswith('run_') and os.path.isdir(os.path.join(root, d))] if os.path.isdir(root) else []
    if not cands:
        raise FileNotFoundError(f'Nenhuma rodada em {root}; rode 04_audit_masks.py primeiro.')
    return max(cands)


def resolve_run_dir(value, root=ABLATION_ROOT):
    """``latest`` -> rodada mais recente; caminho -> ele mesmo (precisa existir)."""
    if value in (None, '', 'latest'):
        return latest_ablation_run(root)
    if not os.path.isdir(value):
        raise FileNotFoundError(value)
    return os.path.abspath(value)
