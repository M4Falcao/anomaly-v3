"""Etapa 7 — braço B: retreina o SEDifferNet com entrada mascarada (nível de imagem).

D-24: o treino do SEDifferNet só otimiza o flow de imagem (``model.nf``); AlexNet e blocos SE
ficam iguais aos do encoder com fundo. Uma NF head treinada sobre o encoder novo sai idêntica à
do braço A, então por padrão o braço B **não** treina head e entra no índice como
``status=ENCODER_ONLY``; a etapa 9 avalia o encoder no nível de imagem. ``--train_head``
restaura o comportamento antigo.

Rodado direto, sem ``--variants``, lê
``<run>/summary.csv`` (etapa 8) e usa as variantes do braço A com ``val_delta_pixel_auroc >= 0``
contra a referência com fundo (no máximo ``--top_k`` por classe). Sem relatório e sem
``--variants`` o script para. O ``run_all.py`` passa todas as candidatas (D-23).

Seleção do encoder (``--encoder_pick``):
  * ``best_test`` (default): ``best_img_auroc.pt``, o mesmo protocolo dos encoders de
    referência em ``final_models/SEDiffernet``, que escolhem a época pelo Image AUROC no
    **teste inteiro**. Isso torna o holdout não independente para métricas de imagem;
    está registrado no índice como ``encoder_selection=test_full``.
  * ``latest``: ``checkpoints/<classe>_<modelname>_latest.pt`` (última época salva).

O SEDifferNet completo custa ~200 épocas (``pre_epochs`` 30 + ``meta_epochs`` 170). Use
``--pre_epochs`` / ``--meta_epochs`` só para smoke test.
"""
import argparse
import csv
import glob
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402
import heads as hd  # noqa: E402
import masking as mk  # noqa: E402


def auto_variants(run_dir, class_name, top_k):
    path = os.path.join(run_dir, 'summary.csv')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['class_name'] == class_name and r['arm'] == 'A'
                and r['role'] == 'candidate' and r['val_delta_pixel_auroc'] not in ('', 'nan')]
    rows = [r for r in rows if float(r['val_delta_pixel_auroc']) >= 0]
    rows.sort(key=lambda r: -float(r['val_pixel_auroc']))
    return [r['variant'] for r in rows[:top_k]]


def find_encoder(enc_dir, class_name, modelname, pick):
    runs = sorted(glob.glob(os.path.join(enc_dir, f'run_*_{class_name}')))
    if not runs:
        return None
    run = runs[-1]
    if pick == 'best_test':
        path = os.path.join(run, 'best_models', 'best_img_auroc.pt')
    else:
        path = os.path.join(run, 'checkpoints', f'{class_name}_{modelname}_latest.pt')
    return path if os.path.isfile(path) else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--classes', default='all')
    p.add_argument('--run_dir', default='latest')
    p.add_argument('--variants', default=None)
    p.add_argument('--top_k', type=int, default=2)
    p.add_argument('--encoder_pick', choices=['best_test', 'latest'], default='best_test')
    p.add_argument('--pre_epochs', type=int, default=None)
    p.add_argument('--meta_epochs', type=int, default=None)
    p.add_argument('--recipe', default='general', choices=['general', 'best_per_class', 'baseline'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=None, help='Épocas da NF head (default: receita).')
    p.add_argument('--trainer_args', default='')
    p.add_argument('--train_head', action='store_true',
                   help='Também treina a NF head sobre o encoder novo. Desligado por padrão (D-24): '
                        'o backbone/SE não muda no retreino, então a head sairia igual à do braço A.')
    p.add_argument('--skip_existing', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    args.trainer_args = shlex.split(args.trainer_args, posix=False)
    run_dir = cm.resolve_run_dir(args.run_dir)
    run_cfg = cm.read_json(os.path.join(run_dir, 'run_config.json'))
    args.split_seed, args.val_fraction = run_cfg['split_seed'], run_cfg['val_fraction']

    for class_name in cm.parse_classes(args.classes):
        class_dir = os.path.join(run_dir, class_name)
        audit_cfg = os.path.join(class_dir, 'audit', 'audit_config.json')
        if not os.path.isfile(audit_cfg):
            print(f'{class_name}: sem audit; pulei.')
            continue
        masks_dir = cm.read_json(audit_cfg)['masks_dir']
        audit_manifest = os.path.join(class_dir, 'audit', 'split_manifest.json')
        variants = ([v.strip() for v in args.variants.split(',') if v.strip()] if args.variants
                    else auto_variants(run_dir, class_name, args.top_k))
        if variants is None:
            print(f'{class_name}: sem summary.csv da etapa 8 e sem --variants; nada a fazer.')
            continue
        if not variants:
            print(f'{class_name}: nenhuma variante do braço A empatou/superou a referência na validação; braço B pulado.')
            continue
        for variant in variants:
            mk.parse_variant(variant)
            enc_dir = os.path.join(class_dir, 'arm_B', variant, 'encoder')
            modelname = f'se_differnet_mask_{variant.replace("+", "p")}'
            root = mk.masked_dataset_root(masks_dir, variant)
            cmd = [cm.PYTHON, hd.ENCODER_RUNNER, '--dataset', root, '--class_name', class_name,
                   '--checkpoint_path', enc_dir, '--modelname', modelname, '--seed', str(args.seed)]
            if args.pre_epochs is not None:
                cmd += ['--pre_epochs', str(args.pre_epochs)]
            if args.meta_epochs is not None:
                cmd += ['--meta_epochs', str(args.meta_epochs)]
            if args.dry_run:
                import subprocess
                print(f'[dry-run] {class_name} B/{variant} encoder: ' + subprocess.list2cmdline(cmd))
                if args.train_head:
                    hd.train_head(class_name, class_dir, 'B', variant, 'candidate', root,
                                  os.path.join(enc_dir, 'run_<ts>_' + class_name, 'best_models', 'best_img_auroc.pt'),
                                  args, audit_manifest)
                continue
            print(f'\n=== {class_name} | braço B | {variant} ===')
            root = mk.materialize(class_name, variant, masks_dir, orig_root=run_cfg['orig_root'])
            encoder = find_encoder(enc_dir, class_name, modelname, args.encoder_pick) if args.skip_existing else None
            minutes = 0.0
            if encoder is None:
                code, minutes = hd.run_logged(cmd, os.path.join(enc_dir, f'train_encoder_{cm.timestamp()}.txt'))
                encoder = find_encoder(enc_dir, class_name, modelname, args.encoder_pick)
                if code != 0 or encoder is None:
                    print(f'{class_name} B/{variant}: treino do encoder falhou (code={code}).')
                    hd.update_index(class_dir, dict(arm='B', variant=variant, role='candidate', status='FAILED_ENCODER',
                                                    returncode=code, minutes=round(minutes, 2)))
                    continue
            enc_info = dict(encoder_selection='test_full' if args.encoder_pick == 'best_test' else 'latest_epoch',
                            encoder_pre_epochs=args.pre_epochs, encoder_meta_epochs=args.meta_epochs)
            if not args.train_head:
                # D-24: backbone/SE não são treinados, então a head B seria idêntica à do braço A.
                hd.update_index(class_dir, dict(arm='B', variant=variant, role='candidate', status='ENCODER_ONLY',
                                                dataset_root=root, se_checkpoint=encoder, minutes=round(minutes, 2),
                                                seed=args.seed, **enc_info))
                print(f'{class_name} B/{variant}: encoder pronto ({encoder}); braço B só no nível de imagem (D-24).')
                continue
            entry = hd.train_head(class_name, class_dir, 'B', variant, 'candidate', root, encoder, args, audit_manifest)
            if entry:
                entry.update(enc_info)
                hd.update_index(class_dir, entry)


if __name__ == '__main__':
    main()
