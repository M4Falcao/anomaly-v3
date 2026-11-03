"""Etapa 1 — converte o dataset sem fundo em um dataset YOLO-seg por classe.

Para cada imagem de ``<nobg_root>/<classe>/train/good``:
  * a máscara do objeto vem dos pixels não-pretos da versão sem fundo;
  * a imagem de entrada do YOLO é a **original com fundo** (mesmo nome e tamanho),
    pois o segmentador será aplicado ao dataset original.

Só ``train/good`` é usado por padrão: imagens de teste nunca entram no treino do
segmentador. Imagens ausentes no dataset sem fundo (o YOLO antigo não detectou nada)
ficam listadas no relatório como ``missing_in_nobg`` e não viram exemplos negativos,
porque o objeto está presente nelas.

Saída: ``data/yolo_seg/<classe>/{images,labels}/{train,val}``, ``data.yaml`` e
``conversion_report.csv``.
"""
import argparse
import csv
import os
import random
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402


def convert_class(class_name, args):
    out_dir = os.path.join(args.out_root, class_name)
    if os.path.exists(out_dir) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f'{out_dir} já existe; use --overwrite para recriar.')

    rows = []
    sources = []
    for split in args.source_splits:
        nobg_split = os.path.join(args.nobg_root, class_name, split)
        if not os.path.isdir(nobg_split):
            continue
        for defect in sorted(os.listdir(nobg_split)):
            if split != 'train' and defect != 'good':
                continue
            nobg_dir = os.path.join(nobg_split, defect)
            orig_dir = os.path.join(args.orig_root, class_name, split, defect)
            nobg_files = set(cm.list_images(nobg_dir))
            for fname in cm.list_images(orig_dir):
                if fname not in nobg_files:
                    rows.append(dict(split=split, defect=defect, file=fname, status='missing_in_nobg'))
                    continue
                sources.append((split, defect, fname, os.path.join(orig_dir, fname), os.path.join(nobg_dir, fname)))
            for fname in sorted(nobg_files - set(cm.list_images(orig_dir))):
                rows.append(dict(split=split, defect=defect, file=fname, status='missing_in_orig'))

    if args.limit:
        sources = sources[:args.limit]

    rng = random.Random(args.seed)
    order = list(range(len(sources)))
    rng.shuffle(order)
    n_val = max(1, int(round(args.val_frac * len(sources)))) if sources else 0
    val_idx = set(order[:n_val])

    if args.dry_run:
        print(f'[dry-run] {class_name}: {len(sources)} pares orig/sem-fundo '
              f'({len(sources) - n_val} train / {n_val} val); '
              f'{sum(r["status"] == "missing_in_nobg" for r in rows)} sem versão sem fundo -> {out_dir}')
        return None

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    counts = {'train': 0, 'val': 0}
    for i, (split, defect, fname, orig_path, nobg_path) in enumerate(sources):
        row = dict(split=split, defect=defect, file=fname)
        orig = cm.read_image(orig_path)
        nobg = cm.read_image(nobg_path)
        if orig.shape != nobg.shape:
            rows.append({**row, 'status': 'size_mismatch'})
            continue
        mask = cm.mask_from_nobg(nobg, args.black_thr, args.close_px, args.min_component_frac)
        area = float(mask.mean())
        polygons = cm.mask_to_polygons(mask, args.epsilon_px)
        row.update(area_frac=round(area, 5), n_instances=len(polygons))
        if area < args.min_obj_frac or area > args.max_obj_frac or not polygons:
            rows.append({**row, 'status': 'rejected_area'})
            continue
        poly_iou = cm.iou(cm.rasterize_polygons(polygons, mask.shape), mask)
        part = 'val' if i in val_idx else 'train'
        stem = f'{split}__{defect}__{os.path.splitext(fname)[0]}'
        shutil.copy2(orig_path, os.path.join(out_dir, 'images', part, stem + os.path.splitext(fname)[1]))
        with open(os.path.join(out_dir, 'labels', part, stem + '.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(cm.polygons_to_yolo_lines(polygons)) + '\n')
        counts[part] += 1
        rows.append({**row, 'status': f'ok_{part}', 'polygon_iou': round(poly_iou, 4)})

    yaml_path = os.path.join(out_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f'path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: {class_name}\n')

    fields = ['split', 'defect', 'file', 'status', 'area_frac', 'n_instances', 'polygon_iou']
    with open(os.path.join(out_dir, 'conversion_report.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ious = [r['polygon_iou'] for r in rows if 'polygon_iou' in r]
    summary = dict(
        class_name=class_name, orig_root=args.orig_root, nobg_root=args.nobg_root,
        source_splits=args.source_splits, seed=args.seed, val_frac=args.val_frac,
        black_thr=args.black_thr, close_px=args.close_px, min_component_frac=args.min_component_frac,
        epsilon_px=args.epsilon_px, min_obj_frac=args.min_obj_frac, max_obj_frac=args.max_obj_frac,
        counts=counts,
        status_counts={s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})},
        polygon_iou_min=min(ious) if ious else None,
        polygon_iou_mean=float(np.mean(ious)) if ious else None,
        created_at=cm.timestamp(),
    )
    cm.write_json(os.path.join(out_dir, 'conversion_summary.json'), summary)
    print(f'{class_name}: train={counts["train"]} val={counts["val"]} '
          f'status={summary["status_counts"]} IoU polígono médio={summary["polygon_iou_mean"]}')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--classes', default='all', help='Lista separada por vírgula ou "all".')
    p.add_argument('--orig_root', default=cm.ORIG_ROOT)
    p.add_argument('--nobg_root', default=cm.NOBG_ROOT)
    p.add_argument('--out_root', default=os.path.join(cm.DATA_DIR, 'yolo_seg'))
    p.add_argument('--source_splits', default='train',
                   help='Splits do dataset usados como fonte. Default só train; '
                        'incluir test vaza imagens de avaliação para o segmentador.')
    p.add_argument('--val_frac', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--black_thr', type=int, default=10)
    p.add_argument('--close_px', type=int, default=5)
    p.add_argument('--min_component_frac', type=float, default=0.001)
    p.add_argument('--epsilon_px', type=float, default=1.0)
    p.add_argument('--min_obj_frac', type=float, default=0.005)
    p.add_argument('--max_obj_frac', type=float, default=0.98)
    p.add_argument('--limit', type=int, default=None, help='Máximo de imagens por classe (smoke test).')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    args.source_splits = [s.strip() for s in args.source_splits.split(',') if s.strip()]
    if any(s != 'train' for s in args.source_splits):
        print('AVISO: splits além de train entram no treino do segmentador (só good).')

    for class_name in cm.parse_classes(args.classes):
        convert_class(class_name, args)


if __name__ == '__main__':
    main()
