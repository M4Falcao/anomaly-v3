# Nível de imagem — SEDifferNet com fundo × sem fundo

Gerado em 20261031_132944. Score DifferNet com 64 rotações fixas. Δ = condição − `bg@orig` nas **mesmas imagens** (interseção), IC95 por bootstrap pareado estratificado (2000).

> Os dois encoders tiveram a época escolhida pelo Image AUROC no próprio teste (o noBg no teste sem fundo): nenhuma partição é limpa e os AUROCs absolutos são otimistas. `val`/`holdout` mostram só a estabilidade do Δ.

## vari-grip

| condição | época ckpt | AUROC ckpt | AUROC próprio teste (n) | AUROC interseção [IC95] | Δ interseção [IC95] | P(Δ≤0) | Δ val | Δ holdout | AP interseção |
|---|---|---|---|---|---|---|---|---|---|
| `bg@orig` | 163 | 0.9770 | 0.9770 (154/40) | 0.9770 [0.9544, 0.9930] | 0.0000 [0.0000, 0.0000] | NaN | 0.0000 | 0.0000 | 0.9401 |
| `nobg@nobg` | 68 | 0.9581 | 0.9544 (154/40) | 0.9544 [0.9204, 0.9809] | -0.0226 [-0.0513, 0.0026] | 0.958 | -0.0258 | -0.0190 | 0.8718 |
| `bg@nobg` | 163 | 0.9770 | 0.5985 (154/40) | 0.5985 [0.4958, 0.6969] | -0.3785 [-0.4776, -0.2829] | 1.000 | -0.4946 | -0.3094 | 0.3603 |
| `nobg@orig` | 68 | 0.9581 | 0.7638 (154/40) | 0.7638 [0.6627, 0.8555] | -0.2132 [-0.3066, -0.1270] | 1.000 | -0.3791 | -0.1134 | 0.6052 |
| `nobg@yolo_raw` | 68 | 0.9581 | 0.9329 (154/40) | 0.9329 [0.8884, 0.9706] | -0.0441 [-0.0857, -0.0083] | 0.994 | -0.0693 | -0.0312 | 0.8307 |
| `nobg@yolo_dil16_mean` | 68 | 0.9581 | 0.6864 (154/40) | 0.6864 [0.5794, 0.7895] | -0.2906 [-0.3952, -0.1881] | 1.000 | -0.3302 | -0.2635 | 0.5468 |

Interseção: 154 imagens (40 anômalas); val 62/16, holdout 92/24.

![vari-grip](roc_vari-grip.png)

