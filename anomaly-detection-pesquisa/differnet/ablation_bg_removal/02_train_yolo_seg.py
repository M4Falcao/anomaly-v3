"""Etapa 2 — treina um YOLO26-seg por classe no dataset gerado pela etapa 1.

Saída: ``runs/yolo_seg/<classe>_<timestamp>/`` (padrão Ultralytics: ``weights/best.pt``,
curvas, ``results.csv``) + ``train_meta.json`` com argumentos e hash do dataset.
Nunca sobrescreve uma rodada anterior.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402


def train_class(class_name, args):
    data_dir = os.path.join(args.data_root, class_name)
    data_yaml = os.path.join(data_dir, 'data.yaml')
    if not os.path.isfile(data_yaml) and not args.dry_run:
        raise FileNotFoundError(f'{data_yaml} não existe; rode 01_build_yolo_dataset.py antes.')
    name = f'{class_name}_{cm.timestamp()}'
    project = os.path.join(args.runs_root, 'yolo_seg')
    train_kwargs = dict(
        data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, seed=args.seed, deterministic=True, device=args.device,
        workers=args.workers, project=project, name=name, exist_ok=False,
        degrees=args.degrees, fliplr=0.5, flipud=args.flipud, plots=True, verbose=False,
    )
    if args.dry_run:
        print(f'[dry-run] {class_name}: YOLO({args.model}).train({json.dumps(train_kwargs, ensure_ascii=False)})')
        return None

    from ultralytics import YOLO
    import ultralytics

    model = YOLO(args.model)
    model.train(**train_kwargs)
    run_dir = os.path.join(project, name)
    summary_path = os.path.join(data_dir, 'conversion_summary.json')
    meta = dict(
        class_name=class_name, model=args.model, ultralytics_version=ultralytics.__version__,
        train_kwargs=train_kwargs,
        conversion_summary=json.load(open(summary_path, encoding='utf-8')) if os.path.isfile(summary_path) else None,
        best_weights=os.path.join(run_dir, 'weights', 'best.pt'),
    )
    cm.write_json(os.path.join(run_dir, 'train_meta.json'), meta)
    print(f'{class_name}: pesos em {meta["best_weights"]}')
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--classes', default='all')
    p.add_argument('--model', default='yolo26s-seg.pt',
                   help='Pesos iniciais Ultralytics (yolo26n/s/m-seg.pt) ou .yaml para treinar do zero.')
    p.add_argument('--data_root', default=os.path.join(cm.DATA_DIR, 'yolo_seg'))
    p.add_argument('--runs_root', default=cm.RUNS_DIR)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8, help='8 cabe na RTX 4060 8 GB com yolo26s-seg@640.')
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--degrees', type=float, default=10.0, help='Rotação de aumento; o SEDifferNet também treina rotacionado.')
    p.add_argument('--flipud', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='0')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    for class_name in cm.parse_classes(args.classes):
        train_class(class_name, args)


if __name__ == '__main__':
    main()
