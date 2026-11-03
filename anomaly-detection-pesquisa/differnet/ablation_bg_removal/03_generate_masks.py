"""Etapa 3 — gera máscaras de objeto no dataset ORIGINAL com o YOLO26-seg treinado.

Diferenças em relação a ``yolo-segmentation/create_dataset.py`` (causa da perda de GT):
  * máscara = **união de todas as instâncias** (antes: só ``masks.data[0]``);
  * ``retina_masks=True``: máscara na resolução original, sem desalinhamento de letterbox;
  * imagem sem detecção **não é descartada**: recebe máscara cheia (= mantém o fundo)
    e é listada em ``masks_report.csv`` (``--no_detection_policy``);
  * a imagem não é alterada: só a máscara é salva. Dilatação/preenchimento são
    aplicados depois, na ablação, a partir deste cache.

Saída: ``masks/<classe>_<fonte>_<timestamp>/<split>/<defeito>/<stem>.png`` (0/255),
``masks_report.csv``, ``manifest.json`` e ``previews/`` opcionais.

``--source nobg`` não usa YOLO: deriva a máscara dos pixels não-pretos do dataset sem
fundo (máscara antiga). Serve de controle e de smoke test sem GPU.
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402


def find_latest_weights(runs_root, class_name):
    cands = glob.glob(os.path.join(runs_root, 'yolo_seg', f'{class_name}_*', 'weights', 'best.pt'))
    if not cands:
        raise FileNotFoundError(f'Nenhum best.pt para {class_name} em {runs_root}\\yolo_seg; '
                                f'treine com 02_train_yolo_seg.py ou passe --weights.')
    return max(cands, key=os.path.getmtime)


def iter_dataset(orig_root, class_name, splits):
    for split in splits:
        split_dir = os.path.join(orig_root, class_name, split)
        if not os.path.isdir(split_dir):
            continue
        for defect in sorted(os.listdir(split_dir)):
            for fname in cm.list_images(os.path.join(split_dir, defect)):
                yield split, defect, fname, os.path.join(split_dir, defect, fname)


def overlay(img, mask):
    vis = img.copy()
    vis[~mask] = (vis[~mask] * 0.3).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    return vis


def generate_class(class_name, args):
    items = list(iter_dataset(args.orig_root, class_name, args.splits))
    if args.limit:
        items = items[:args.limit]
    out_dir = os.path.join(args.out_root, f'{class_name}_{args.source}_{cm.timestamp()}')
    weights = None
    if args.source == 'yolo':
        weights = args.weights or find_latest_weights(args.runs_root, class_name)
    if args.dry_run:
        print(f'[dry-run] {class_name}: {len(items)} imagens de {args.splits} | fonte={args.source} '
              f'{weights or args.nobg_root} (conf={args.conf}, imgsz={args.imgsz}) -> {out_dir}')
        return None

    if args.source == 'yolo':
        from ultralytics import YOLO
        import ultralytics
        model = YOLO(weights)
        version = ultralytics.__version__
    else:
        model, version = None, None
    rows = []
    previews = {}
    for split, defect, fname, path in items:
        img = cm.read_image(path)
        h, w = img.shape[:2]
        n, confs = 0, []
        if model is not None:
            res = model.predict(img, conf=args.conf, iou=args.iou, imgsz=args.imgsz, retina_masks=True,
                                device=args.device, verbose=False)[0]
            n = 0 if res.masks is None else len(res.masks)
            confs = [] if res.boxes is None else res.boxes.conf.cpu().numpy().tolist()
            mask = cm.union_instance_masks(res.masks.data.cpu().numpy() > 0.5, (h, w)) if n else None
        else:
            nobg_path = os.path.join(args.nobg_root, class_name, split, defect, fname)
            mask = None
            if os.path.isfile(nobg_path):
                nobg = cm.read_image(nobg_path)
                if nobg.shape == img.shape:
                    mask = cm.mask_from_nobg(nobg)
                    n = int(mask.any())
                    mask = mask if n else None
        if mask is not None:
            status = 'ok'
        else:
            mask = np.ones((h, w), bool) if args.no_detection_policy == 'full' else np.zeros((h, w), bool)
            status = 'no_detection'
        stem = os.path.splitext(fname)[0]
        rel = os.path.join(split, defect, stem + '.png')
        cm.write_png(os.path.join(out_dir, rel), mask.astype(np.uint8) * 255)
        rows.append(dict(split=split, defect=defect, file=fname, mask=rel, status=status, n_instances=n,
                         max_conf=round(max(confs), 4) if confs else '', area_frac=round(float(mask.mean()), 5),
                         height=h, width=w))
        key = (split, defect)
        if previews.get(key, 0) < args.previews:
            previews[key] = previews.get(key, 0) + 1
            cm.write_png(os.path.join(out_dir, 'previews', split, defect, stem + '.png'), overlay(img, mask))

    fields = ['split', 'defect', 'file', 'mask', 'status', 'n_instances', 'max_conf', 'area_frac', 'height', 'width']
    with open(os.path.join(out_dir, 'masks_report.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    by_split = {}
    for r in rows:
        s = by_split.setdefault(f'{r["split"]}/{r["defect"]}', dict(n=0, no_detection=0, multi_instance=0))
        s['n'] += 1
        s['no_detection'] += r['status'] == 'no_detection'
        s['multi_instance'] += r['n_instances'] > 1
    manifest = dict(
        class_name=class_name, source=args.source,
        weights=os.path.abspath(weights) if weights else None,
        weights_sha256=cm.sha256_file(weights) if weights and os.path.isfile(weights) else None,
        nobg_root=args.nobg_root if args.source == 'nobg' else None,
        ultralytics_version=version, orig_root=args.orig_root, splits=args.splits,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, retina_masks=True, union_of_instances=True,
        no_detection_policy=args.no_detection_policy, limit=args.limit, counts=by_split,
        created_at=cm.timestamp(),
    )
    cm.write_json(os.path.join(out_dir, 'manifest.json'), manifest)
    print(f'{class_name}: {len(rows)} máscaras -> {out_dir}')
    for k, v in by_split.items():
        print(f'   {k:28s} n={v["n"]:5d} sem_detecção={v["no_detection"]:4d} multi_instância={v["multi_instance"]:4d}')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--classes', default='all')
    p.add_argument('--source', choices=['yolo', 'nobg'], default='yolo',
                   help='yolo: YOLO26-seg treinado (etapa 2). nobg: pixels não-pretos do dataset sem fundo '
                        '(reproduz a máscara antiga; imagens ausentes recebem máscara cheia).')
    p.add_argument('--nobg_root', default=cm.NOBG_ROOT)
    p.add_argument('--weights', default=None, help='best.pt explícito (só com uma classe). Default: mais recente em runs/.')
    p.add_argument('--orig_root', default=cm.ORIG_ROOT)
    p.add_argument('--runs_root', default=cm.RUNS_DIR)
    p.add_argument('--out_root', default=cm.MASKS_DIR)
    p.add_argument('--splits', default='train,test')
    p.add_argument('--conf', type=float, default=0.10, help='Baixo de propósito: perder peça custa mais que incluir fundo.')
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0')
    p.add_argument('--no_detection_policy', choices=['full', 'empty'], default='full')
    p.add_argument('--previews', type=int, default=8, help='Overlays salvos por split/defeito.')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    args.splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    classes = cm.parse_classes(args.classes)
    if args.weights and (len(classes) != 1 or args.source != 'yolo'):
        p.error('--weights exige --source yolo e exatamente uma classe em --classes.')

    for class_name in classes:
        generate_class(class_name, args)


if __name__ == '__main__':
    main()
