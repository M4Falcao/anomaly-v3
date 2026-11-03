"""Treina um SEDifferNet via ``main.py`` com ``config`` sobrescrito (uso interno da etapa 7)."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--class_name', required=True)
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--modelname', required=True)
    p.add_argument('--pre_epochs', type=int, default=None)
    p.add_argument('--meta_epochs', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    import config as c
    c.dataset_path = args.dataset
    c.class_name = args.class_name
    c.modelname = args.modelname
    c.checkpoint_path = args.checkpoint_path
    c.seed = args.seed
    c.mlflow_run_name = f'{args.class_name}_{args.modelname}'
    c.mlflow_experiment_name = 'D19_mask_ablation_encoder'
    if args.pre_epochs is not None:
        c.pre_epochs = args.pre_epochs
    if args.meta_epochs is not None:
        c.meta_epochs = args.meta_epochs
    c.set_seed(c.seed)
    print(f'SEDifferNet: dataset={c.dataset_path} classe={c.class_name} pre={c.pre_epochs} '
          f'meta={c.meta_epochs} sub={c.sub_epochs} -> {c.checkpoint_path}')
    from main import main as train_main
    train_main()


if __name__ == '__main__':
    main()
