"""Etapa 5 — poda pré-registrada: escolhe as estratégias finalistas por classe.

Regra (fixada antes de ver qualquer métrica de modelo; D-19):
  1. elegível = ``gt_lost_frac`` na **validação** <= ``--max_gt_lost`` (5%);
  2. entre as elegíveis, mantém as ``--max_finalists`` (2) de **menor área** em train/good
     (= mais fundo removido);
  3. nenhuma elegível -> ``--fallback`` (``dil64,bbox20``) e a classe é marcada
     ``mask_reliable=false``.

Variantes treinadas na etapa 6 = finalistas × ``--fills`` × ``--feathers`` + a referência
``raw__black`` (reproduz o "sem fundo" no inventário completo). A referência com fundo
(``none``) é a head já treinada em ``REFERENCE_HEAD_RUN``; não é retreinada.

Saída: ``<run>/<classe>/finalists.json``.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402
import masking as mk  # noqa: E402


def select(rows, max_gt_lost, max_finalists, fallback):
    val = {r['strategy']: r for r in rows if r['partition'] == 'val'}
    train_area = {r['strategy']: float(r['area_kept']) for r in rows if r['partition'] == 'train_good'}
    table = []
    for s, r in val.items():
        lost = float(r['gt_lost_frac']) if r['gt_lost_frac'] not in ('', None) else float('nan')
        table.append(dict(strategy=s, val_gt_lost_frac=lost, train_area_kept=train_area.get(s, float(r['area_kept'])),
                          val_n_lost50=int(r['n_lost50']), val_n_lost95=int(r['n_lost95']),
                          eligible=bool(lost <= max_gt_lost)))
    eligible = sorted((t for t in table if t['eligible']), key=lambda t: (t['train_area_kept'], t['strategy']))
    if eligible:
        return [t['strategy'] for t in eligible[:max_finalists]], True, table
    return list(fallback), False, table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--classes', default='all')
    p.add_argument('--run_dir', default='latest')
    p.add_argument('--max_gt_lost', type=float, default=0.05)
    p.add_argument('--max_finalists', type=int, default=2)
    p.add_argument('--fallback', default='dil64,bbox20')
    p.add_argument('--fills', default='imagenet_mean', help="D-21: 'black' saiu do default (use 'black,imagenet_mean' para reproduzir D-20).")
    p.add_argument('--feathers', default='0')
    p.add_argument('--no_reference', action='store_true', help='Não incluir raw__black.')
    p.add_argument('--force_strategies', default=None,
                   help='Ignora a regra e usa estas estratégias (registrado como override no JSON).')
    args = p.parse_args()
    run_dir = cm.resolve_run_dir(args.run_dir)
    fills = [f.strip() for f in args.fills.split(',') if f.strip()]
    feathers = [int(f) for f in args.feathers.split(',') if f.strip()]
    fallback = [mk.validate_strategy(s.strip()) for s in args.fallback.split(',') if s.strip()]

    for class_name in cm.parse_classes(args.classes):
        audit_csv = os.path.join(run_dir, class_name, 'audit', 'audit_summary.csv')
        if not os.path.isfile(audit_csv):
            print(f'{class_name}: sem audit ({audit_csv}); pulei.')
            continue
        with open(audit_csv, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        finalists, reliable, table = select(rows, args.max_gt_lost, args.max_finalists, fallback)
        override = None
        if args.force_strategies:
            override = [mk.validate_strategy(s.strip()) for s in args.force_strategies.split(',') if s.strip()]
            finalists = override
        variants = [dict(variant=mk.variant_name(s, fill, fe), role='candidate', strategy=s)
                    for s in finalists for fill in fills for fe in feathers]
        if not args.no_reference and mk.REFERENCE_VARIANT not in [v['variant'] for v in variants]:
            variants.append(dict(variant=mk.REFERENCE_VARIANT, role='reference_nobg', strategy='raw'))
        payload = dict(class_name=class_name, rule=dict(max_gt_lost=args.max_gt_lost, max_finalists=args.max_finalists,
                                                        fallback=fallback, partition='val',
                                                        tie_break='menor área train/good'),
                       finalists=finalists, mask_reliable=reliable, override=override, table=table,
                       variants=variants, created_at=cm.timestamp())
        cm.write_json(os.path.join(run_dir, class_name, 'finalists.json'), payload)
        flag = '' if reliable else '  [máscara NÃO confiável: fallback]'
        print(f'{class_name}: finalistas={finalists}{flag}{"  [override]" if override else ""}')
        for v in variants:
            print(f'   - {v["variant"]:28s} ({v["role"]})')


if __name__ == '__main__':
    main()
