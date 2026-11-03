# Ablação de máscara de objeto (D-19) — relatório

Rodada: `C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\resultado_analise_final\ablacao_mascara_objeto\run_20261101_011735`  
Referência com fundo: `C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\resultado_analise_final\treino_nf_heads\run_20261025_181523`  
Gerado em 20261102_192834. Uma seed (D-13). O teste já foi inspecionado; o holdout **não** é independente (D-18). Métricas de pixel por histograma (65536 bins); IC95 = bootstrap pareado estratificado por imagem (1000 reamostragens, 1024 bins).

**Critério (só validação):** GT perdido ≤ 5%, ΔPixel AUROC ≥ 0 com IC95 superior ≥ 0, ΔAUPRO ≥ −0.02. A recomendada é a aprovada com maior Pixel AUROC de validação; o holdout só confirma.

## vari-grip

| braço | variante | papel | val AUROC | Δ val [IC95] | val AUPRO | ΔAUPRO val | GT perdido val | AUROC no objeto val | contorno/fora FPR val | holdout AUROC | Δ holdout [IC95] | holdout AUPRO | Img AUROC holdout | encoder img AUROC holdout | critério |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | `none` | reference_bg | 0.8837 | 0.0000 [0.0000, 0.0000] | 0.6836 | 0.0000 | 0.000 | 0.7902 | 0.013/0.010 | 0.9134 | 0.0000 [0.0000, 0.0000] | 0.7721 | 0.7286 | 0.9896 | — |
| A | `dil16__imagenet_mean` | candidate | 0.8820 | -0.0017 [-0.0376, 0.0266] | 0.6806 | -0.0030 | 0.021 | 0.7379 | 0.013/0.010 | 0.8921 | -0.0213 [-0.0672, 0.0107] | 0.7291 | 0.6281 | 0.6483 | falhou: delta |
| B | `dil16__imagenet_mean` | candidate | 0.8820 | -0.0017 [-0.0376, 0.0266] | 0.6806 | -0.0030 | 0.021 | 0.7379 | 0.013/0.010 | 0.8921 | -0.0213 [-0.0672, 0.0107] | 0.7291 | 0.6281 | 0.9638 | falhou: delta |

> Braço B: encoder escolhido por Image AUROC no teste inteiro (mesmo protocolo dos encoders de referência); métricas de imagem no holdout não são independentes.

### Nível de imagem — SEDifferNet (64 rotações)

Braço A = encoder **com fundo** congelado recebendo a imagem mascarada só na inferência (diagnóstico: a máscara sozinha derruba o score de imagem?). Braço B = encoder retreinado com máscara.

| braço | variante | encoder | val Img AUROC | Δ val [IC95] | holdout Img AUROC | Δ holdout [IC95] |
|---|---|---|---|---|---|---|
| ref | `none` | com fundo / imagem original | 0.9606 | NaN [NaN, NaN] | 0.9896 | NaN [NaN, NaN] |
| A | `dil16__imagenet_mean` | com fundo / imagem mascarada | 0.5516 | -0.4090 [-0.5761, -0.2540] | 0.6483 | -0.3413 [-0.4583, -0.2359] |
| B | `dil16__imagenet_mean` | retreinado / imagem mascarada | 0.9620 | 0.0014 [-0.0380, 0.0408] | 0.9638 | -0.0257 [-0.0589, -0.0012] |

![vari-grip](vari-grip/report_vari-grip.png)

## lightning-rod-suspension

| braço | variante | papel | val AUROC | Δ val [IC95] | val AUPRO | ΔAUPRO val | GT perdido val | AUROC no objeto val | contorno/fora FPR val | holdout AUROC | Δ holdout [IC95] | holdout AUPRO | Img AUROC holdout | encoder img AUROC holdout | critério |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | `none` | reference_bg | 0.8813 | 0.0000 [0.0000, 0.0000] | 0.6300 | 0.0000 | 0.000 | 0.8316 | 0.014/0.010 | 0.8541 | 0.0000 [0.0000, 0.0000] | 0.5257 | 0.6913 | 0.9857 | — |
| A | `dil16__imagenet_mean` | candidate | 0.9491 | 0.0678 [0.0533, 0.0872] | 0.8419 | 0.2119 | 0.000 | 0.8488 | 0.021/0.009 | 0.9295 | 0.0753 [0.0584, 0.0934] | 0.7607 | 0.6990 | 0.6184 | ★ recomendada |
| B | `dil16__imagenet_mean` | candidate | 0.9491 | 0.0678 [0.0533, 0.0872] | 0.8419 | 0.2119 | 0.000 | 0.8488 | 0.021/0.009 | 0.9295 | 0.0753 [0.0584, 0.0934] | 0.7607 | 0.6990 | 0.9546 | ★ recomendada |

> Braço B: encoder escolhido por Image AUROC no teste inteiro (mesmo protocolo dos encoders de referência); métricas de imagem no holdout não são independentes.

### Nível de imagem — SEDifferNet (64 rotações)

Braço A = encoder **com fundo** congelado recebendo a imagem mascarada só na inferência (diagnóstico: a máscara sozinha derruba o score de imagem?). Braço B = encoder retreinado com máscara.

| braço | variante | encoder | val Img AUROC | Δ val [IC95] | holdout Img AUROC | Δ holdout [IC95] |
|---|---|---|---|---|---|---|
| ref | `none` | com fundo / imagem original | 0.9965 | NaN [NaN, NaN] | 0.9857 | NaN [NaN, NaN] |
| A | `dil16__imagenet_mean` | com fundo / imagem mascarada | 0.6738 | -0.3227 [-0.4658, -0.1855] | 0.6184 | -0.3673 [-0.4792, -0.2484] |
| B | `dil16__imagenet_mean` | retreinado / imagem mascarada | 0.9669 | -0.0296 [-0.0733, 0.0012] | 0.9546 | -0.0311 [-0.0786, 0.0026] |

![lightning-rod-suspension](lightning-rod-suspension/report_lightning-rod-suspension.png)

## polymer-insulator-upper-shackle

| braço | variante | papel | val AUROC | Δ val [IC95] | val AUPRO | ΔAUPRO val | GT perdido val | AUROC no objeto val | contorno/fora FPR val | holdout AUROC | Δ holdout [IC95] | holdout AUPRO | Img AUROC holdout | encoder img AUROC holdout | critério |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | `none` | reference_bg | 0.8756 | 0.0000 [0.0000, 0.0000] | 0.6393 | 0.0000 | 0.000 | 0.7930 | 0.024/0.009 | 0.8598 | 0.0000 [0.0000, 0.0000] | 0.6161 | 0.5405 | 0.9656 | — |
| A | `dil16__imagenet_mean` | candidate | 0.6507 | -0.2248 [-0.3132, -0.1431] | 0.5087 | -0.1307 | 0.529 | 0.7485 | 0.007/0.010 | 0.6738 | -0.1861 [-0.2670, -0.1090] | 0.4522 | 0.7029 | 0.7881 | falhou: gt_lost,delta,aupro |

### Nível de imagem — SEDifferNet (64 rotações)

Braço A = encoder **com fundo** congelado recebendo a imagem mascarada só na inferência (diagnóstico: a máscara sozinha derruba o score de imagem?). Braço B = encoder retreinado com máscara.

| braço | variante | encoder | val Img AUROC | Δ val [IC95] | holdout Img AUROC | Δ holdout [IC95] |
|---|---|---|---|---|---|---|
| ref | `none` | com fundo / imagem original | 0.9720 | NaN [NaN, NaN] | 0.9656 | NaN [NaN, NaN] |
| A | `dil16__imagenet_mean` | com fundo / imagem mascarada | 0.7218 | -0.2502 [-0.3498, -0.1615] | 0.7881 | -0.1774 [-0.2675, -0.0933] |

![polymer-insulator-upper-shackle](polymer-insulator-upper-shackle/report_polymer-insulator-upper-shackle.png)

