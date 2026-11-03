# Nível de imagem — SEDifferNet com fundo × sem fundo

Gerado em 20261102_192222. Score DifferNet com 64 rotações fixas. Δ = condição − `bg@orig` nas **mesmas imagens** (interseção), IC95 por bootstrap pareado estratificado (1000).

> Os dois encoders tiveram a época escolhida pelo Image AUROC no próprio teste (o noBg no teste sem fundo): nenhuma partição é limpa e os AUROCs absolutos são otimistas. `val`/`holdout` mostram só a estabilidade do Δ.

## vari-grip

| condição | época ckpt | AUROC ckpt | AUROC próprio teste (n) | AUROC interseção [IC95] | Δ interseção [IC95] | P(Δ≤0) | Δ val | Δ holdout | AP interseção |
|---|---|---|---|---|---|---|---|---|---|
| `bg@orig` | 163 | 0.9770 | 0.9770 (154/40) | 0.9770 [0.9526, 0.9930] | 0.0000 [0.0000, 0.0000] | NaN | 0.0000 | 0.0000 | 0.9401 |
| `nobg@nobg` | 68 | 0.9581 | 0.9544 (154/40) | 0.9544 [0.9204, 0.9792] | -0.0226 [-0.0509, 0.0020] | 0.962 | -0.0258 | -0.0190 | 0.8718 |
| `bg@nobg` | 163 | 0.9770 | 0.5985 (154/40) | 0.5985 [0.4919, 0.6963] | -0.3785 [-0.4827, -0.2857] | 1.000 | -0.4946 | -0.3094 | 0.3603 |
| `nobg@orig` | 68 | 0.9581 | 0.7638 (154/40) | 0.7638 [0.6614, 0.8522] | -0.2132 [-0.3068, -0.1298] | 1.000 | -0.3791 | -0.1134 | 0.6052 |
| `nobg@masked_dil16__imagenet_mean` | 68 | 0.9581 | 0.6864 (154/40) | 0.6864 [0.5765, 0.7893] | -0.2906 [-0.4003, -0.1874] | 1.000 | -0.3302 | -0.2635 | 0.5468 |
| `retrained_dil16__imagenet_mean@masked_dil16__imagenet_mean` | 108 | 0.9618 | 0.9605 (154/40) | 0.9605 [0.9320, 0.9840] | -0.0164 [-0.0388, 0.0020] | 0.957 | 0.0014 | -0.0257 | 0.8883 |

Interseção: 154 imagens (40 anômalas); val 62/16, holdout 92/24.

![vari-grip](roc_vari-grip.png)

