"""Consolidate the October NF-head ablation runs into the tables of the scientific analysis.

Reads only saved artifacts (grid_val.csv, holdout_test_top.csv, ablation_summary.csv);
no inference or training is run.

Usage:
    python scripts/eval/consolidate_nf_head_ablation.py [--out_dir DIR] [--n_boot 5000]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ABLATION_ROOT = PROJECT_ROOT / 'resultado_analise_final' / 'ablacao_nf_head'
RUNS = {
    'run_full_20261007_123558': ['glass-insulator', 'lightning-rod-suspension',
                                 'polymer-insulator-upper-shackle'],
    'run_full_20261010_195638': ['vari-grip', 'yoke-suspension'],
}
SHORT = {'glass-insulator': 'glass', 'lightning-rod-suspension': 'lightning',
         'polymer-insulator-upper-shackle': 'polymer', 'vari-grip': 'vari',
         'yoke-suspension': 'yoke'}
CLASSES = ['glass', 'lightning', 'polymer', 'vari', 'yoke']
FACTORS = ['padcfg', 'tta', 'score_mode', 'levels', 'pos_calib', 'align_corners', 'sigma']
CONTRASTS = [
    ('padcfg', 'pad0', '112reflect'), ('padcfg', 'pad0', '112replicate'),
    ('padcfg', '112reflect', '112replicate'),
    ('tta', 'none', 'hflip'), ('tta', 'none', 'flips'), ('tta', 'hflip', 'flips'),
    ('score_mode', 'per_level_minmax', 'raw'),
    ('score_mode', 'cflow_original', 'raw'), ('score_mode', 'cflow_original', 'per_level_std'),
    ('score_mode', 'cflow_original', 'per_level_minmax'),
    ('score_mode', 'cflow_original', 'per_level_prob'),
    ('levels', '012', '02'), ('levels', '012', '01'), ('levels', '012', '12'),
    ('levels', '012', '0'), ('levels', '012', '1'), ('levels', '012', '2'),
    ('pos_calib', 'none', 'l23'), ('align_corners', True, False),
    ('sigma', 0.0, 2.0), ('sigma', 0.0, 4.0), ('sigma', 0.0, 8.0), ('sigma', 0.0, 12.0),
]


def load():
    grids, holds, summaries = [], [], []
    for run, classes in RUNS.items():
        for cls in classes:
            summary = pd.read_csv(ABLATION_ROOT / run / cls / 'ablation_summary.csv')
            summary['cls'] = SHORT[cls]
            summaries.append(summary)
            for d in sorted((ABLATION_ROOT / run / cls).glob('posthoc_*')):
                head = d.name.replace('posthoc_', '').replace('_s42', '')
                g = pd.read_csv(d / 'grid_val.csv', dtype={'levels': str})
                # image_score does not change pixel metrics: one row per pixel recipe.
                g = g.drop_duplicates(subset=['pad', 'pad_mode', 'tta', 'score_mode', 'levels',
                                              'pos_calib', 'align_corners', 'sigma', 'margin']).copy()
                g['padcfg'] = np.where(g['pad'].astype(int) == 0, 'pad0', '112' + g['pad_mode'].astype(str))
                g['cls'], g['head'] = SHORT[cls], head
                grids.append(g[FACTORS + ['cls', 'head', 'name', 'pixel_auroc', 'aupro']])
                h = pd.read_csv(d / 'holdout_test_top.csv', dtype={'levels': str})
                h['cls'], h['head'] = SHORT[cls], head
                holds.append(h)
    grid = pd.concat(grids, ignore_index=True)
    grid['regret'] = grid.groupby(['cls', 'head']).pixel_auroc.transform('max') - grid.pixel_auroc
    return grid, pd.concat(holds, ignore_index=True), pd.concat(summaries, ignore_index=True)


def paired_effects(grid, n_boot, seed=0):
    """Validation pairs differing only in one factor; class-level hierarchical bootstrap CI."""
    rng = np.random.default_rng(seed)
    rows = []
    for factor, ref, alt in CONTRASTS:
        others = [f for f in FACTORS if f != factor]
        m = grid[grid[factor] == ref].merge(grid[grid[factor] == alt], on=others + ['cls', 'head'],
                                            suffixes=('_ref', '_alt'))
        m['d'] = m.pixel_auroc_alt - m.pixel_auroc_ref
        m['dp'] = m.aupro_alt - m.aupro_ref
        per_head = m.groupby(['cls', 'head']).d.mean()
        per_class = per_head.groupby('cls').mean()
        boots = np.empty(n_boot)
        heads_by_class = {c: per_head.loc[c].to_numpy() for c in per_class.index}
        for b in range(n_boot):
            sampled = rng.choice(per_class.index.to_numpy(), len(per_class), replace=True)
            boots[b] = np.mean([rng.choice(heads_by_class[c], len(heads_by_class[c])).mean()
                                for c in sampled])
        nonzero = per_class.to_numpy()
        p_value = stats.wilcoxon(nonzero).pvalue if np.any(nonzero != 0) else np.nan
        rows.append({'factor': factor, 'contrast': f'{alt} vs {ref}', 'pairs': len(m),
                     'mean_delta_auroc': m.d.mean(), 'frac_pairs_positive': (m.d > 0).mean(),
                     'class_mean_delta_auroc': per_class.mean(),
                     'ci95_low': np.percentile(boots, 2.5), 'ci95_high': np.percentile(boots, 97.5),
                     'classes_positive': int((per_class > 0).sum()),
                     'heads_positive': int((per_head > 0).sum()),
                     'wilcoxon_p_classes': p_value,
                     'mean_delta_aupro_15heads': m.dp.mean(),
                     **{f'delta_{c}': per_class.get(c, np.nan) for c in CLASSES}})
    return pd.DataFrame(rows)


def holdout_ofat(hold):
    o = hold[hold.study == 'ofat']
    agg = o.groupby('role').agg(heads=('delta_auroc_vs_baseline', 'size'),
                                mean_delta_auroc=('delta_auroc_vs_baseline', 'mean'),
                                min_delta=('delta_auroc_vs_baseline', 'min'),
                                max_delta=('delta_auroc_vs_baseline', 'max'),
                                ci_positive=('delta_auroc_ci_low', lambda x: int((x > 0).sum())),
                                ci_negative=('delta_auroc_ci_high', lambda x: int((x < 0).sum())),
                                mean_delta_aupro_15heads=('delta_aupro_vs_baseline', 'mean'))
    per_class = o.groupby(['role', 'cls']).delta_auroc_vs_baseline.mean().unstack()
    agg['classes_positive'] = (per_class > 0).sum(axis=1)
    return agg.join(per_class.add_prefix('delta_')).reset_index().sort_values('mean_delta_auroc')


def regret_tables(grid):
    out = {}
    for factor in ['levels', 'score_mode', 'sigma', 'tta', 'padcfg', 'pos_calib']:
        out[factor] = (grid.groupby([factor, 'cls', 'head']).regret.min()
                       .groupby([factor, 'cls']).mean().unstack().reset_index())
    n_heads = grid.groupby(['cls', 'head']).ngroups
    grid = grid.assign(key=grid[FACTORS].astype(str).agg('|'.join, axis=1))
    common = grid.groupby('key').filter(lambda x: len(x) == n_heads)
    single = (common.groupby('key').agg(name=('name', 'first'), mean_regret=('regret', 'mean'),
                                        max_regret=('regret', 'max'),
                                        mean_val_auroc=('pixel_auroc', 'mean'))
              .sort_values('mean_regret').reset_index(drop=True))
    per_class = (common.groupby(['cls', 'key']).agg(name=('name', 'first'),
                                                    mean_regret=('regret', 'mean'),
                                                    max_regret=('regret', 'max'),
                                                    mean_val_auroc=('pixel_auroc', 'mean'))
                 .reset_index())
    per_class = per_class.loc[per_class.groupby('cls').mean_regret.idxmin()].drop(columns='key')
    return out, single.head(20), per_class


def variant_tables(hold):
    pivot = hold[hold.role.isin(['baseline_cflow_original', 'baseline_producao', 'top1'])].pivot_table(
        index=['cls', 'head'], columns='role', values='pixel_auroc')
    top1 = pivot['top1'].unstack('head')
    fixed = pivot['baseline_cflow_original'].unstack('head')
    fixed_delta = fixed.sub(fixed['baseline'], axis=0)
    friedman_top1 = stats.friedmanchisquare(*[top1[c] for c in top1.columns])
    friedman_fixed = stats.friedmanchisquare(*[fixed[c] for c in fixed.columns])
    tests = pd.DataFrame([{'test': 'friedman_top1_variants', 'statistic': friedman_top1.statistic,
                           'p_value': friedman_top1.pvalue},
                          {'test': 'friedman_fixed_reference_variants',
                           'statistic': friedman_fixed.statistic, 'p_value': friedman_fixed.pvalue}])
    return pivot.reset_index(), top1.reset_index(), fixed_delta.reset_index(), tests


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out_dir', default=str(ABLATION_ROOT / 'analise_consolidada_20261013'))
    parser.add_argument('--n_boot', type=int, default=5000)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    grid, hold, summaries = load()
    paired_effects(grid, args.n_boot).to_csv(os.path.join(args.out_dir, 'paired_effects_validation.csv'), index=False)
    holdout_ofat(hold).to_csv(os.path.join(args.out_dir, 'ofat_holdout.csv'), index=False)
    regrets, single, per_class = regret_tables(grid)
    for factor, table in regrets.items():
        table.to_csv(os.path.join(args.out_dir, f'regret_{factor}.csv'), index=False)
    single.to_csv(os.path.join(args.out_dir, 'single_recipe_ranking.csv'), index=False)
    per_class.to_csv(os.path.join(args.out_dir, 'best_fixed_recipe_per_class.csv'), index=False)
    refs, top1, fixed_delta, tests = variant_tables(hold)
    refs.to_csv(os.path.join(args.out_dir, 'holdout_references_per_head.csv'), index=False)
    top1.to_csv(os.path.join(args.out_dir, 'holdout_top1_class_by_variant.csv'), index=False)
    fixed_delta.to_csv(os.path.join(args.out_dir, 'holdout_fixed_reference_delta_vs_baseline.csv'), index=False)
    tests.to_csv(os.path.join(args.out_dir, 'variant_tests.csv'), index=False)
    hold[hold.aupro == 0].to_csv(os.path.join(args.out_dir, 'holdout_aupro_zero_rows.csv'), index=False)
    summaries.to_csv(os.path.join(args.out_dir, 'ablation_summaries_25heads.csv'), index=False)
    print(f'{grid.groupby(["cls", "head"]).ngroups} heads, {len(grid)} validation recipes -> {args.out_dir}')


if __name__ == '__main__':
    main()
