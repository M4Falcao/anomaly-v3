# Nível de imagem — SEDifferNet com fundo × sem fundo

Gerado em 20261102_192728. Score DifferNet com 64 rotações fixas. Δ = condição − `bg@orig` nas **mesmas imagens** (interseção), IC95 por bootstrap pareado estratificado (1000).

> Os dois encoders tiveram a época escolhida pelo Image AUROC no próprio teste (o noBg no teste sem fundo): nenhuma partição é limpa e os AUROCs absolutos são otimistas. `val`/`holdout` mostram só a estabilidade do Δ.

## polymer-insulator-upper-shackle

| condição | época ckpt | AUROC ckpt | AUROC próprio teste (n) | AUROC interseção [IC95] | Δ interseção [IC95] | P(Δ≤0) | Δ val | Δ holdout | AP interseção |
|---|---|---|---|---|---|---|---|---|---|
| `bg@orig` | 146 | 0.9680 | 0.9680 (317/82) | 0.9637 [0.9403, 0.9814] | 0.0000 [0.0000, 0.0000] | NaN | 0.0000 | 0.0000 | 0.8726 |
| `nobg@nobg` | 11 | 0.9624 | 0.9612 (283/63) | 0.9612 [0.9391, 0.9776] | -0.0025 [-0.0203, 0.0165] | 0.625 | -0.0147 | 0.0036 | 0.8714 |
| `bg@nobg` | 146 | 0.9680 | 0.8301 (283/63) | 0.8301 [0.7651, 0.8835] | -0.1336 [-0.2034, -0.0766] | 1.000 | -0.1145 | -0.1399 | 0.7227 |
| `nobg@orig` | 11 | 0.9624 | 0.5962 (317/82) | 0.5798 [0.5015, 0.6479] | -0.3839 [-0.4596, -0.3222] | 1.000 | -0.4158 | -0.3626 | 0.2555 |
| `nobg@masked_dil16__imagenet_mean` | 11 | 0.9624 | 0.7564 (317/82) | 0.7504 [0.6804, 0.8123] | -0.2133 [-0.2858, -0.1517] | 1.000 | -0.2555 | -0.1953 | 0.4797 |
| `retrained_dil16__imagenet_mean@masked_dil16__imagenet_mean` | 19 | 0.9328 | 0.9328 (317/82) | 0.9320 [0.8926, 0.9620] | -0.0317 [-0.0608, -0.0070] | 0.995 | -0.0349 | -0.0322 | 0.8452 |

Interseção: 283 imagens (63 anômalas); val 115/27, holdout 168/36.

![polymer-insulator-upper-shackle](roc_polymer-insulator-upper-shackle.png)

