# Nível de imagem — SEDifferNet com fundo × sem fundo

Gerado em 20261102_192336. Score DifferNet com 64 rotações fixas. Δ = condição − `bg@orig` nas **mesmas imagens** (interseção), IC95 por bootstrap pareado estratificado (1000).

> Os dois encoders tiveram a época escolhida pelo Image AUROC no próprio teste (o noBg no teste sem fundo): nenhuma partição é limpa e os AUROCs absolutos são otimistas. `val`/`holdout` mostram só a estabilidade do Δ.

## lightning-rod-suspension

| condição | época ckpt | AUROC ckpt | AUROC próprio teste (n) | AUROC interseção [IC95] | Δ interseção [IC95] | P(Δ≤0) | Δ val | Δ holdout | AP interseção |
|---|---|---|---|---|---|---|---|---|---|
| `bg@orig` | 41 | 0.9907 | 0.9907 (163/46) | 0.9904 [0.9778, 0.9984] | 0.0000 [0.0000, 0.0000] | NaN | 0.0000 | 0.0000 | 0.9793 |
| `nobg@nobg` | 74 | 0.9655 | 0.9607 (135/39) | 0.9607 [0.9298, 0.9856] | -0.0296 [-0.0580, -0.0069] | 0.997 | -0.0401 | -0.0225 | 0.9092 |
| `bg@nobg` | 41 | 0.9907 | 0.6442 (135/39) | 0.6442 [0.5336, 0.7439] | -0.3462 [-0.4559, -0.2462] | 1.000 | -0.2509 | -0.4167 | 0.4379 |
| `nobg@orig` | 74 | 0.9655 | 0.3883 (163/46) | 0.3451 [0.2540, 0.4456] | -0.6453 [-0.7375, -0.5432] | 1.000 | -0.5767 | -0.6807 | 0.2189 |
| `nobg@masked_dil16__imagenet_mean` | 74 | 0.9655 | 0.3872 (163/46) | 0.3512 [0.2545, 0.4581] | -0.6392 [-0.7369, -0.5293] | 1.000 | -0.6376 | -0.6335 | 0.2238 |
| `retrained_dil16__imagenet_mean@masked_dil16__imagenet_mean` | 12 | 0.9671 | 0.9610 (163/46) | 0.9602 [0.9236, 0.9880] | -0.0302 [-0.0652, -0.0016] | 0.988 | -0.0296 | -0.0335 | 0.8214 |

Interseção: 135 imagens (39 anômalas); val 55/14, holdout 80/25.

![lightning-rod-suspension](roc_lightning-rod-suspension.png)

