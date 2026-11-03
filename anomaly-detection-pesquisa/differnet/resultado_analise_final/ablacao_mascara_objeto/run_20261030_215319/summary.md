# Ablação de máscara de objeto (D-19) — relatório

Rodada: `C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\resultado_analise_final\ablacao_mascara_objeto\run_20261030_215319`  
Referência com fundo: `C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\resultado_analise_final\treino_nf_heads\run_20261025_181523`  
Gerado em 20261031_125133. Uma seed (D-13). O teste já foi inspecionado; o holdout **não** é independente (D-18). Métricas de pixel por histograma (65536 bins); IC95 = bootstrap pareado estratificado por imagem (1000 reamostragens, 1024 bins).

**Critério (só validação):** GT perdido ≤ 5%, ΔPixel AUROC ≥ 0 com IC95 superior ≥ 0, ΔAUPRO ≥ −0.02. A recomendada é a aprovada com maior Pixel AUROC de validação; o holdout só confirma.

## vari-grip

| braço | variante | papel | val AUROC | Δ val [IC95] | val AUPRO | ΔAUPRO val | GT perdido val | AUROC no objeto val | contorno/fora FPR val | holdout AUROC | Δ holdout [IC95] | holdout AUPRO | Img AUROC holdout | encoder img AUROC holdout | critério |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | `none` | reference_bg | 0.8837 | 0.0000 [0.0000, 0.0000] | 0.6836 | 0.0000 | 0.000 | 0.7902 | 0.013/0.010 | 0.9134 | 0.0000 [0.0000, 0.0000] | 0.7721 | 0.7286 | 0.9896 | — |
| A | `dil16__black` | candidate | 0.8478 | -0.0359 [-0.0665, -0.0097] | 0.5752 | -0.1084 | 0.021 | 0.6489 | 0.041/0.007 | 0.8563 | -0.0571 [-0.0989, -0.0263] | 0.6387 | 0.6495 | 0.5386 | falhou: delta,aupro |
| A | `dil16__imagenet_mean` | candidate | 0.8820 | -0.0017 [-0.0376, 0.0266] | 0.6806 | -0.0030 | 0.021 | 0.7379 | 0.013/0.010 | 0.8921 | -0.0213 [-0.0672, 0.0107] | 0.7291 | 0.6281 | 0.6483 | falhou: delta |
| A | `raw__black` | reference_nobg | 0.9010 | 0.0173 [-0.0152, 0.0460] | 0.7182 | 0.0347 | 0.063 | 0.7803 | 0.045/0.006 | 0.8993 | -0.0141 [-0.0561, 0.0160] | 0.7269 | 0.5729 | 0.6581 | — |

### Nível de imagem — SEDifferNet (64 rotações)

Braço A = encoder **com fundo** congelado recebendo a imagem mascarada só na inferência (diagnóstico: a máscara sozinha derruba o score de imagem?). Braço B = encoder retreinado com máscara.

| braço | variante | encoder | val Img AUROC | Δ val [IC95] | holdout Img AUROC | Δ holdout [IC95] |
|---|---|---|---|---|---|---|
| ref | `none` | com fundo / imagem original | 0.9606 | NaN [NaN, NaN] | 0.9896 | NaN [NaN, NaN] |
| A | `dil16__black` | com fundo / imagem mascarada | 0.4511 | -0.5095 [-0.6712, -0.3519] | 0.5386 | -0.4510 [-0.5711, -0.3296] |
| A | `dil16__imagenet_mean` | com fundo / imagem mascarada | 0.5516 | -0.4090 [-0.5761, -0.2540] | 0.6483 | -0.3413 [-0.4583, -0.2359] |
| A | `raw__black` | com fundo / imagem mascarada | 0.4538 | -0.5068 [-0.6604, -0.3641] | 0.6581 | -0.3315 [-0.4449, -0.2224] |

![vari-grip](vari-grip/report_vari-grip.png)

