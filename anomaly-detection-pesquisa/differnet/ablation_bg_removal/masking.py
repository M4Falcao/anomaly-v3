"""Estratégias de máscara de objeto e materialização de datasets mascarados (D-19).

Convenções:
  * Toda estratégia é definida no espaço 448×448, o mesmo grid em que o SEDifferNet vê a
    imagem (``Resize((448, 448))``). Raios e margens ficam, portanto, comparáveis
    entre imagens de tamanhos nativos diferentes.
  * A máscara final é reamostrada para a resolução nativa e aplicada na imagem
    original antes de qualquer transformação do pipeline.
  * Nome de variante: ``<estratégia>__<preenchimento>[__f<feather>]``, por exemplo
    ``dil32__black`` ou ``bbox10__imagenet_mean__f8``. ``none`` = imagem original.
"""
import os
import re
import shutil

import cv2
import numpy as np

import common as cm

GRID = 448
IMAGENET_MEAN_RGB = (124, 116, 104)  # round(255 * [0.485, 0.456, 0.406])
FILLS = ('black', 'imagenet_mean', 'blur')
DEFAULT_AUDIT_STRATEGIES = ('raw', 'dil8', 'dil16', 'dil32', 'dil64', 'hull', 'hull+dil16', 'bbox10', 'bbox20')
REFERENCE_VARIANT = 'raw__black'  # reproduz o experimento "sem fundo" no inventário completo

_STRATEGY_RE = re.compile(r'^(raw|hull|dil(\d+)|hull\+dil(\d+)|bbox(\d+))$')


def validate_strategy(name):
    if not _STRATEGY_RE.match(name):
        raise ValueError(f'Estratégia inválida {name!r}: use raw, hull, dil<r>, hull+dil<r>, bbox<m%>')
    return name


def parse_variant(variant):
    if variant == 'none':
        return dict(strategy=None, fill=None, feather=0)
    parts = variant.split('__')
    if len(parts) not in (2, 3):
        raise ValueError(f'Variante inválida {variant!r}')
    strategy, fill = validate_strategy(parts[0]), parts[1]
    if fill not in FILLS:
        raise ValueError(f'Preenchimento inválido {fill!r}; use {FILLS}')
    feather = 0
    if len(parts) == 3:
        if not parts[2].startswith('f'):
            raise ValueError(f'Sufixo de feather inválido em {variant!r}')
        feather = int(parts[2][1:])
    return dict(strategy=strategy, fill=fill, feather=feather)


def variant_name(strategy, fill, feather=0):
    validate_strategy(strategy)
    if fill not in FILLS:
        raise ValueError(f'Preenchimento inválido {fill!r}')
    return f'{strategy}__{fill}' + (f'__f{int(feather)}' if feather else '')


def to_grid(mask):
    """Máscara nativa -> 448×448 (vizinho mais próximo, como o GT)."""
    return cv2.resize(mask.astype(np.uint8), (GRID, GRID), interpolation=cv2.INTER_NEAREST).astype(bool)


def _dilate(mask, radius):
    if radius <= 0 or not mask.any():
        return mask.copy()
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return dist <= radius


def _hull(mask):
    out = np.zeros(mask.shape, np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.fillPoly(out, [cv2.convexHull(np.vstack(contours))], 1)
    return out.astype(bool)


def _bbox(mask, margin_pct):
    out = np.zeros(mask.shape, bool)
    if not mask.any():
        return out
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    my, mx = int(round(margin_pct / 100 * h)), int(round(margin_pct / 100 * w))
    out[max(0, ys.min() - my):min(h, ys.max() + 1 + my), max(0, xs.min() - mx):min(w, xs.max() + 1 + mx)] = True
    return out


def strategy_mask(base_grid, strategy):
    """Aplica a estratégia a uma máscara base já no grid 448×448.

    Máscara base vazia (nenhum objeto) vira máscara cheia: sem objeto localizado,
    nada é apagado (mesma política do ``no_detection_policy=full``).
    """
    m = _STRATEGY_RE.match(validate_strategy(strategy))
    base = base_grid.astype(bool)
    if not base.any():
        return np.ones_like(base)
    if strategy == 'raw':
        return base.copy()
    if strategy == 'hull':
        return _hull(base)
    if m.group(2):
        return _dilate(base, int(m.group(2)))
    if m.group(3):
        return _dilate(_hull(base), int(m.group(3)))
    return _bbox(base, int(m.group(4)))


def contour_band(mask_grid, width=8):
    """Faixa de ±``width`` px em torno da borda da máscara (grid 448)."""
    if mask_grid.all() or not mask_grid.any():
        return np.zeros_like(mask_grid, dtype=bool)
    inside = cv2.distanceTransform(mask_grid.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~mask_grid).astype(np.uint8), cv2.DIST_L2, 5)
    return ((inside > 0) & (inside <= width)) | ((outside > 0) & (outside <= width))


def apply_fill(img_bgr, mask_grid, fill, feather=0):
    """Preenche o fundo da imagem nativa fora da máscara (definida no grid 448).

    ``feather`` (px no grid 448) suaviza a transição com um alpha gaussiano.
    """
    h, w = img_bgr.shape[:2]
    alpha = cv2.resize(mask_grid.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    if feather > 0:
        sigma = feather * max(h, w) / GRID / 2.0
        alpha = np.clip(cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma), 0, 1)
        alpha = np.maximum(alpha, cv2.resize(mask_grid.astype(np.float32), (w, h),
                                             interpolation=cv2.INTER_NEAREST))
    if fill == 'black':
        background = np.zeros_like(img_bgr)
    elif fill == 'imagenet_mean':
        background = np.empty_like(img_bgr)
        background[:] = IMAGENET_MEAN_RGB[::-1]
    elif fill == 'blur':
        sigma = 0.05 * max(h, w)
        background = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        raise ValueError(f'Preenchimento inválido {fill!r}')
    a = alpha[..., None]
    return (img_bgr.astype(np.float32) * a + background.astype(np.float32) * (1 - a)).round().astype(np.uint8)


# ─── Cache de máscaras geradas pela etapa 3 ──────────────────────────────────

def find_latest_masks_dir(class_name, masks_root=cm.MASKS_DIR, source='yolo'):
    if not os.path.isdir(masks_root):
        raise FileNotFoundError(f'{masks_root} não existe; rode 03_generate_masks.py.')
    prefix = f'{class_name}_{source}_'
    cands = [os.path.join(masks_root, d) for d in os.listdir(masks_root)
             if d.startswith(prefix) and os.path.isfile(os.path.join(masks_root, d, 'manifest.json'))]
    if not cands:
        raise FileNotFoundError(f'Nenhuma máscara {source} para {class_name} em {masks_root}; '
                                f'rode 03_generate_masks.py --source {source}.')
    return max(cands, key=os.path.getmtime)


def resolve_masks_dir(class_name, value=None, source='yolo'):
    """``value`` explícito (pasta) ou a máscara mais recente da fonte."""
    if value:
        if not os.path.isfile(os.path.join(value, 'manifest.json')):
            raise FileNotFoundError(f'{value} não tem manifest.json de 03_generate_masks.py')
        manifest = cm.read_json(os.path.join(value, 'manifest.json'))
        if manifest.get('class_name') != class_name:
            raise ValueError(f'{value} é da classe {manifest.get("class_name")}, não {class_name}')
        return os.path.abspath(value)
    return find_latest_masks_dir(class_name, source=source)


def load_base_mask(masks_dir, rel_image_path):
    """Máscara base (nativa, bool) para ``<split>/<defeito>/<arquivo>`` relativo à classe."""
    split, defect, fname = rel_image_path.replace('/', os.sep).split(os.sep)[-3:]
    path = os.path.join(masks_dir, split, defect, os.path.splitext(fname)[0] + '.png')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Máscara ausente: {path}')
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) > 127


def load_grid_strategy(masks_dir, rel_image_path, strategy):
    return strategy_mask(to_grid(load_base_mask(masks_dir, rel_image_path)), strategy)


# ─── Materialização ──────────────────────────────────────────────────────────

def _write_like(path, img_bgr, jpeg_quality):
    ext = os.path.splitext(path)[1].lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if ext in ('.jpg', '.jpeg') else []
    ok, buf = cv2.imencode(ext, img_bgr, params)
    if not ok:
        raise IOError(f'Falha ao codificar {path}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf.tofile(path)


def _link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def masked_dataset_root(masks_dir, variant, data_root=cm.DATA_DIR):
    return os.path.join(data_root, 'masked', os.path.basename(os.path.normpath(masks_dir)), variant)


def materialize(class_name, variant, masks_dir, orig_root=cm.ORIG_ROOT, data_root=cm.DATA_DIR,
                jpeg_quality=95, overwrite=False, limit=None):
    """Cria ``<root>/<classe>/{train,test,ground_truth}`` com o fundo preenchido.

    Mesmos nomes e mesma árvore do dataset original, o que preserva a ordem de
    ``AlignedTestDataset`` e o split val/holdout. ``ground_truth`` é linkado (hardlink)
    ou copiado sem alteração. Retorna a raiz do dataset (para ``--dataset``).
    """
    spec = parse_variant(variant)
    if spec['strategy'] is None:
        return orig_root
    root = masked_dataset_root(masks_dir, variant, data_root)
    class_dir = os.path.join(root, class_name)
    manifest_path = os.path.join(class_dir, 'materialize_manifest.json')
    expected = dict(class_name=class_name, variant=variant, masks_dir=os.path.abspath(masks_dir),
                    orig_root=os.path.abspath(orig_root), jpeg_quality=jpeg_quality, limit=limit, **spec)
    if os.path.isfile(manifest_path) and not overwrite:
        import json
        with open(manifest_path, encoding='utf-8') as f:
            found = json.load(f)
        if {k: found.get(k) for k in expected} == expected:
            return root
        raise FileExistsError(f'{class_dir} existe com outros parâmetros; use --overwrite.')
    if os.path.exists(class_dir):
        shutil.rmtree(class_dir)

    counts = {}
    src_class = os.path.join(orig_root, class_name)
    for split in ('train', 'test'):
        split_dir = os.path.join(src_class, split)
        for defect in sorted(os.listdir(split_dir)):
            files = cm.list_images(os.path.join(split_dir, defect))
            if limit:
                files = files[:limit]
            for fname in files:
                rel = os.path.join(split, defect, fname)
                img = cm.read_image(os.path.join(src_class, rel))
                grid = load_grid_strategy(masks_dir, rel, spec['strategy'])
                _write_like(os.path.join(class_dir, rel), apply_fill(img, grid, spec['fill'], spec['feather']),
                            jpeg_quality)
            counts[f'{split}/{defect}'] = len(files)
    gt_src = os.path.join(src_class, 'ground_truth')
    if os.path.isdir(gt_src):
        for dirpath, _, files in os.walk(gt_src):
            for fname in files:
                src = os.path.join(dirpath, fname)
                _link_or_copy(src, os.path.join(class_dir, 'ground_truth', os.path.relpath(src, gt_src)))
    cm.write_json(manifest_path, {**expected, 'counts': counts, 'created_at': cm.timestamp()})
    return root
