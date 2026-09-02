"""Main pipeline runner.

Trains/fits each method on a limited subset of `lightning-rod-suspension`
normal images and evaluates pixel-level AUROC on the full test set.
Post-processing A, B, C is applied to every method.

Usage:
    python -m pixel_pipeline.run
"""
import os
import time
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import make_loaders
from .methods import METHODS
from .postproc import apply_postproc


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_method(method, test_loader, img_size, sigma=4.0):
    """Run inference + post-processing on the test set.

    Returns: dict mapping variant -> (scores_flat, labels_flat).
    Variants: 'raw', 'A' (smooth), 'B' (grad), 'C' (ensemble).
    """
    variants = {'raw': [], 'A': [], 'B': [], 'C': []}
    labels_acc = []

    for inputs, _, masks in test_loader:
        inputs = inputs.to(DEVICE)
        raw_map = method.score(inputs)  # (B, H, W)
        out = apply_postproc(method, inputs, raw_map, img_size, sigma=sigma)
        for k in variants:
            variants[k].append(out[k])
        labels_acc.append(masks.squeeze(1).numpy())

    labels = np.concatenate(labels_acc, axis=0).reshape(-1).astype(np.uint8)

    results = {}
    for k, maps in variants.items():
        flat = np.concatenate(maps, axis=0).reshape(-1)
        results[k] = (flat, labels)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default=r"C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg")
    parser.add_argument('--class_name', type=str, default='lightning-rod-suspension')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--out_size', type=int, default=56)
    parser.add_argument('--n_train', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=8, help='for flow-based methods')
    parser.add_argument('--sigma', type=float, default=4.0)
    parser.add_argument('--methods', type=str, default='D_PaDiM,E_PatchCore,F_FastFlow,G_CFLOW,H_DifferNet,I_SEDifferNet')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='./pixel_pipeline_results.csv')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader = make_loaders(
        args.dataset, args.class_name, img_size=args.img_size,
        n_train=args.n_train, batch_size=args.batch_size, seed=args.seed,
    )

    method_names = [m.strip() for m in args.methods.split(',') if m.strip()]
    table = []  # rows: (method, variant, pixel_auroc, fit_time, eval_time)

    for name in method_names:
        if name not in METHODS:
            print(f"Skipping unknown method: {name}")
            continue
        print(f"\n====== {name} ======")
        cls = METHODS[name]

        # Build method with appropriate args
        kw = dict(img_size=args.img_size, out_size=args.out_size)
        if name in ('F_FastFlow', 'G_CFLOW', 'H_DifferNet', 'I_SEDifferNet'):
            kw['epochs'] = args.epochs

        method = cls(**kw)

        t0 = time.time()
        method.fit(train_loader)
        fit_t = time.time() - t0

        t0 = time.time()
        results = evaluate_method(method, test_loader, args.img_size, sigma=args.sigma)
        eval_t = time.time() - t0

        for variant, (scores, labels) in results.items():
            if labels.sum() == 0 or labels.sum() == len(labels):
                auroc = float('nan')
            else:
                auroc = roc_auc_score(labels, scores)
            table.append((name, variant, auroc, fit_t, eval_t))
            print(f"  {name:12s} | {variant:5s} | pixel AUROC = {auroc:.4f}")

        # Cleanup GPU
        del method
        torch.cuda.empty_cache()

    # Print summary
    print("\n====================== SUMMARY ======================")
    print(f"{'Method':14s} {'Variant':8s} {'Pixel AUROC':>12s} {'Fit(s)':>8s} {'Eval(s)':>8s}")
    print("-" * 60)
    for name, variant, auroc, fit_t, eval_t in table:
        print(f"{name:14s} {variant:8s} {auroc:>12.4f} {fit_t:>8.1f} {eval_t:>8.1f}")

    # Save CSV
    with open(args.out, 'w') as f:
        f.write("method,variant,pixel_auroc,fit_seconds,eval_seconds\n")
        for row in table:
            f.write(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.2f},{row[4]:.2f}\n")
    print(f"\nSaved results to {args.out}")


if __name__ == '__main__':
    main()
