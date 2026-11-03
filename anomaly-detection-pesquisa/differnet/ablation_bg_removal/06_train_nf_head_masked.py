"""Etapa 6 — braço A: SEDifferNet COM FUNDO congelado + NF head treinada com entrada mascarada.

Para cada variante de ``<run>/<classe>/finalists.json`` (ou ``--variants``):
  1. materializa o dataset mascarado em ``data/masked/<máscaras>/<variante>/<classe>/``
     (mesmos nomes e árvore do original -> mesmo split; GT por hardlink);
  2. chama ``scripts/train/pixel_train_from_pretrained.py --recipe general`` com
     ``--dataset`` apontando para ele e o encoder com fundo de ``final_models/SEDiffernet``;
  3. confere se o ``split_manifest.json`` do treino bate com o do audit e registra a head em
     ``<run>/<classe>/heads_index.json``.

O encoder não muda (D-01). A época é escolhida pelo Pixel AUROC de validação em ``raw`` (D-04).
"""
import argparse
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402
import heads as hd  # noqa: E402
import masking as mk  # noqa: E402


def variants_for(class_dir, override):
    if override:
        return [dict(variant=v, role='candidate' if v != mk.REFERENCE_VARIANT else 'reference_nobg')
                for v in override]
    path = os.path.join(class_dir, 'finalists.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} ausente; rode 05_select_strategies.py ou passe --variants.')
    return cm.read_json(path)['variants']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--classes', default='all')
    p.add_argument('--run_dir', default='latest')
    p.add_argument('--variants', default=None, help='Lista explícita (ex.: dil32__black,raw__black).')
    p.add_argument('--se_checkpoint', default=None, help='Encoder explícito (só com uma classe).')
    p.add_argument('--recipe', default='general', choices=['general', 'best_per_class', 'baseline'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=None, help='Sobrescreve o orçamento da receita (evitar).')
    p.add_argument('--trainer_args', default='', help='Argumentos extras para o treinador (string).')
    p.add_argument('--jpeg_quality', type=int, default=95)
    p.add_argument('--overwrite_data', action='store_true')
    p.add_argument('--materialize_only', action='store_true')
    p.add_argument('--skip_existing', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    args.trainer_args = shlex.split(args.trainer_args, posix=False)
    run_dir = cm.resolve_run_dir(args.run_dir)
    run_cfg = cm.read_json(os.path.join(run_dir, 'run_config.json'))
    args.split_seed, args.val_fraction = run_cfg['split_seed'], run_cfg['val_fraction']
    classes = cm.parse_classes(args.classes)
    if args.se_checkpoint and len(classes) != 1:
        p.error('--se_checkpoint exige exatamente uma classe.')
    override = [v.strip() for v in args.variants.split(',')] if args.variants else None

    for class_name in classes:
        class_dir = os.path.join(run_dir, class_name)
        audit_cfg_path = os.path.join(class_dir, 'audit', 'audit_config.json')
        if not os.path.isfile(audit_cfg_path):
            print(f'{class_name}: sem audit; pulei.')
            continue
        masks_dir = cm.read_json(audit_cfg_path)['masks_dir']
        audit_manifest = os.path.join(class_dir, 'audit', 'split_manifest.json')
        se_ckpt = args.se_checkpoint or hd.default_se_checkpoint(class_name)
        for item in variants_for(class_dir, override):
            variant = item['variant']
            mk.parse_variant(variant)
            if args.dry_run:
                root = mk.masked_dataset_root(masks_dir, variant)
                print(f'[dry-run] materializar {class_name}/{variant} -> {root}')
            else:
                print(f'\n=== {class_name} | braço A | {variant} ===')
                root = mk.materialize(class_name, variant, masks_dir, orig_root=run_cfg['orig_root'],
                                      jpeg_quality=args.jpeg_quality, overwrite=args.overwrite_data)
                print(f'dataset: {root}')
            if args.materialize_only:
                continue
            hd.train_head(class_name, class_dir, 'A', variant, item.get('role', 'candidate'), root, se_ckpt,
                          args, audit_manifest)


if __name__ == '__main__':
    main()
